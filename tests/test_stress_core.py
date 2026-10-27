"""Stress tests for the three core solvers against the quadprog reference projection.

Dykstra's algorithm reaches the projection only in the limit, so a test at a finite budget
states what it checks: a bound on the squared distance to the reference at that budget,
agreement with plain Dykstra at the corresponding cycle, or invariance of the path and of its
classification under scaling and under a coordinate that no constraint touches. An iterate at a
finite budget is not feasible in general, and what every solver guarantees instead is that each
visit's output lies in its own half-space to within rounding.
"""

import contextlib
import io
from pathlib import Path
import re
import sys
import tempfile
import unittest
import warnings

import matplotlib

matplotlib.use("Agg")

import numpy as np


WORKING_VERSION = Path(__file__).resolve().parents[1] / "Python" / "Working Version"
sys.path.insert(0, str(WORKING_VERSION))

from convex_projection_solver import (
    DykstraMapHybridSolver,
    DykstraProjectionSolver,
    DykstraStallDetectionSolver,
    dykstra_projection,
)
from visualiser import ResultExporter


SOLVERS = (DykstraProjectionSolver, DykstraMapHybridSolver, DykstraStallDetectionSolver)

FAST_FORWARD_LOG = re.compile(
    r"Fast forwarding (-?\d+) rounds to exit stalling at iteration (\d+)"
)
EPS = np.finfo(float).eps

# The box [-1, 1]^2 followed by the line x / 2 + y = 1 as two opposite half-spaces, the
# problem of paper_figure.py and main.py. Its projection is (0, 1) from either start below.
BOX_LINE_A = np.array([
    [1.0, 0.0], [-1.0, 0.0], [0.0, 1.0], [0.0, -1.0], [0.5, 1.0], [-0.5, -1.0],
])
BOX_LINE_B = np.array([1.0, 1.0, 1.0, 1.0, 1.0, -1.0])

# From (-4, 1.4) plain Dykstra holds squared error 0.8 for cycles 1 to 16, and from
# (-2, 1.4) for cycles 1 to 6. Either stall ends in a single fast-forward at iteration 2.
BOX_LINE_STALLS = {
    (-4.0, 1.4): {"last_stalled_cycle": 16, "jumps": [(2, 14)]},
    (-2.0, 1.4): {"last_stalled_cycle": 6, "jumps": [(2, 4)]},
}

# The wedge y <= 0, y <= x tan(20 degrees) projected from (-1, 1). Its first cycle of
# alternating projections lands on a boundary point that is not the projection.
WEDGE_A = np.array([[0.0, 1.0], [-np.tan(np.radians(20.0)), 1.0]])
WEDGE_B = np.zeros(2)
WEDGE_Z = np.array([-1.0, 1.0])

# Dimensions, number of half-spaces, seed, cycle budget, and the bound on the squared error
# at that budget relative to the squared distance from the start to the reference.
RANDOM_CASES = (
    (2, 3, 1000, 60, 1e-28),
    (2, 12, 1001, 60, 1e-28),
    (3, 8, 1002, 60, 1e-28),
    (4, 16, 1003, 120, 1e-28),
    (6, 24, 1004, 300, 1e-28),
    (8, 40, 1005, 400, 1e-28),
    (16, 48, 1006, 400, 1e-4),
    (32, 80, 1007, 300, 1e-4),
    (64, 96, 1008, 300, 1e-16),
    (64, 200, 1009, 60, 1e-2),
    (12, 200, 1010, 60, 1e-4),
)

# Dimensions, number of half-spaces and seed of problems on which plain Dykstra stalls
# within 80 cycles, so that the stall detection solver jumps.
STALLING_CASES = ((4, 18, 5000), (3, 9, 5006), (2, 13, 5013), (4, 19, 5015), (2, 12, 5025))


def run(solver_type, z, A, b, max_iter, **options):
    """Solves, returning the result and the (iteration, skip) pair of every logged jump."""
    log = io.StringIO()
    with contextlib.redirect_stdout(log):
        result = solver_type(z, A, b, max_iter, **options).solve()
    jumps = [(int(iteration), int(skip))
             for skip, iteration in FAST_FORWARD_LOG.findall(log.getvalue())]
    return result, jumps


def row_products(A, x):
    """Returns A @ x, written elementwise.

    numpy 2.1 linked to Apple's Accelerate raises spurious divide-by-zero warnings from
    matrix products of size 16 and above.
    """
    return (A * x).sum(axis=1)


def bounded_problem(rng, num_planes, dimensions):
    """A random bounded polyhedron with a strictly interior anchor, built as Generator 2 of
    the paper's supplement.

    The first dimensions + 1 rows are the axis directions and their negated sum under a
    Haar-random rotation, normals that positively span the space and so bound the polyhedron.
    Every further row has entries s tan(theta), with a random sign s and theta uniform in
    [0, pi/2). Each offset is the row's value at the anchor plus a margin uniform in
    [0.2, 1.5), which keeps the anchor strictly inside.
    """
    basis, upper = np.linalg.qr(rng.standard_normal((dimensions, dimensions)))
    basis = basis * np.sign(np.diag(upper))
    extra = num_planes - (dimensions + 1)
    A = np.vstack([
        basis.T,
        -basis.sum(axis=1),
        rng.choice([-1.0, 1.0], size=(extra, dimensions))
        * np.tan(rng.uniform(0.0, np.pi / 2, size=(extra, dimensions))),
    ])
    anchor = rng.uniform(-2.0, 2.0, size=dimensions)
    b = row_products(A, anchor) + rng.uniform(0.2, 1.5, size=num_planes)
    return A, b, anchor


def infeasible_start(rng, A, b, anchor):
    """A start outside the polyhedron, drawn around its anchor."""
    scale = 4.0
    while True:
        z = anchor + scale * rng.standard_normal(A.shape[1])
        if np.any(row_products(A, z) > b):
            return z
        scale *= 1.5


def random_case(dimensions, num_planes, seed):
    rng = np.random.default_rng(seed)
    A, b, anchor = bounded_problem(rng, num_planes, dimensions)
    return infeasible_start(rng, A, b, anchor), A, b, anchor


def reference_projection(z, A, b):
    return DykstraProjectionSolver(z, A, b, 0).actual_projection


def normalised(A, b):
    norms = np.linalg.norm(A, axis=1)
    return A / norms[:, None], b / norms


def visit_excess(result, A, b):
    """Largest slack of a visit's output in its own half-space, as a fraction of the
    magnitudes summed in that slack and in the projection that produced it.

    It needs errors_for_plotting, whose corrections bound the step of a projection. For a
    projection made without a correction, as in the hybrid's MAP cycles, the input of the
    visit bounds the step instead.
    """
    unit_A, unit_b = normalised(A, b)
    outputs = result.path[1:]
    inputs = np.concatenate([result.path[:-1, -1:], result.path[1:, :-1]], axis=1)
    magnitudes = np.abs(outputs) + np.abs(inputs) + np.abs(result.errors_for_plotting)
    slack = (outputs * unit_A).sum(axis=2) - unit_b
    scale = (magnitudes * np.abs(unit_A)).sum(axis=2) + np.abs(unit_b)
    return np.max(slack / scale)


def stall_mask(result):
    return ~np.isnan(result.stalled_errors)


def converged_mask(result):
    return ~np.isnan(result.converged_errors)


class RandomPolyhedronTests(unittest.TestCase):
    """Seeded random polyhedra in 2 to 64 dimensions with 3 to 200 half-spaces."""

    def test_dykstra_approaches_the_reference_projection(self) -> None:
        """At each case's budget the squared error is below its bound, the reference is
        feasible, and every recorded visit lies in its own half-space to rounding."""
        for dimensions, num_planes, seed, budget, bound in RANDOM_CASES:
            with self.subTest(dimensions=dimensions, planes=num_planes):
                z, A, b, _ = random_case(dimensions, num_planes, seed)
                solver = DykstraProjectionSolver(z, A, b, budget, track_error=True,
                                                 plot_errors=True)
                result = solver.solve()
                unit_A, unit_b = normalised(A, b)

                self.assertTrue(np.isfinite(result.path).all())
                self.assertTrue(np.isfinite(result.errors_for_plotting).all())
                self.assertLessEqual(
                    np.max(row_products(unit_A, solver.actual_projection) - unit_b),
                    1e-12 * np.abs(solver.actual_projection).max())
                self.assertLessEqual(result.squared_errors[-1],
                                     bound * result.squared_errors[0])
                self.assertLessEqual(visit_excess(result, A, b), 1e-12)

    def assert_fast_forward_is_plain_dykstra(self, z, A, b, budget):
        """Checks that every logged skip is positive and that every cycle of the stall
        detection solver is plain Dykstra's at the cycle it stands for, and returns the
        jumps."""
        fast, jumps = run(DykstraStallDetectionSolver, z, A, b, budget, track_error=True,
                          plot_errors=True)
        for iteration, skip in jumps:
            self.assertGreater(skip, 0, f"skip at iteration {iteration}")
        plain, _ = run(DykstraProjectionSolver, z, A, b,
                       budget + sum(skip for _, skip in jumps), track_error=True)
        scale = np.abs(plain.path).max()
        for cycle in range(budget + 1):
            plain_cycle = cycle + sum(skip for i, skip in jumps if i < cycle)
            np.testing.assert_allclose(
                fast.path[cycle], plain.path[plain_cycle], rtol=0.0, atol=1e-9 * scale,
                err_msg=f"cycle {cycle} against {plain_cycle}")
            self.assertAlmostEqual(fast.squared_errors[cycle],
                                   plain.squared_errors[plain_cycle],
                                   delta=1e-9 * scale * np.sqrt(plain.squared_errors[0]))
        self.assertLessEqual(visit_excess(fast, A, b), 1e-12)
        return jumps

    def test_fast_forward_reproduces_plain_dykstra(self) -> None:
        """On these cases the solver runs plain Dykstra's cycles, jumping or not."""
        for dimensions, num_planes, seed, budget, _ in RANDOM_CASES:
            with self.subTest(dimensions=dimensions, planes=num_planes):
                z, A, b, _ = random_case(dimensions, num_planes, seed)
                self.assert_fast_forward_is_plain_dykstra(z, A, b, budget)

    def test_fast_forward_skips_the_stalls_of_random_polyhedra(self) -> None:
        """Each of these problems makes the solver jump at least once, and a jump of N
        cycles lands on plain Dykstra's path to the rounding that N cycles accumulate."""
        for dimensions, num_planes, seed in STALLING_CASES:
            with self.subTest(dimensions=dimensions, planes=num_planes, seed=seed):
                z, A, b, _ = random_case(dimensions, num_planes, seed)
                self.assertTrue(self.assert_fast_forward_is_plain_dykstra(z, A, b, 80))

    def test_hybrid_reaches_the_reference_in_two_and_three_dimensions(self) -> None:
        """The hybrid's limit is the projection once its MAP cycles have reached the
        intersection, which on these cases happens within the budget."""
        for dimensions, num_planes, seed, _, _ in RANDOM_CASES[:3]:
            with self.subTest(dimensions=dimensions, planes=num_planes):
                z, A, b, _ = random_case(dimensions, num_planes, seed)
                result, _ = run(DykstraMapHybridSolver, z, A, b, 400, track_error=True,
                                plot_errors=True)
                self.assertLessEqual(result.squared_errors[-1],
                                     1e-20 * result.squared_errors[0])
                self.assertLessEqual(visit_excess(result, A, b), 1e-12)

    def test_hybrid_visits_stay_in_their_half_spaces(self) -> None:
        """The hybrid's limit is not asserted here, since it alternates MAP and Dykstra
        cycles, but every visit is still a projection onto its own half-space."""
        for dimensions, num_planes, seed, budget, _ in RANDOM_CASES:
            with self.subTest(dimensions=dimensions, planes=num_planes):
                z, A, b, _ = random_case(dimensions, num_planes, seed)
                result, _ = run(DykstraMapHybridSolver, z, A, b, min(budget, 120),
                                track_error=True, plot_errors=True)
                self.assertTrue(np.isfinite(result.path).all())
                self.assertLessEqual(visit_excess(result, A, b), 1e-12)


class DependentNormalTests(unittest.TestCase):
    """Active normals that are linearly dependent or nearly so."""

    CASES = {
        "a box row repeated": (
            [2.0, 0.5],
            [[1.0, 0.0], [-1.0, 0.0], [0.0, 1.0], [0.0, -1.0], [1.0, 0.0]],
            [1.0, 1.0, 1.0, 1.0, 1.0],
            [1.0, 0.5],
            1e-12,
        ),
        "two normals 1e-10 apart through a vertex": (
            # z - (0, 0) bisects the two normals, so both carry a multiplier
            [-np.sin(5e-11), np.cos(5e-11)],
            [[0.0, 1.0], [-np.sin(1e-10), np.cos(1e-10)], [1.0, 0.0], [-1.0, 0.0],
             [0.0, -1.0]],
            [0.0, 0.0, 1.0, 1.0, 1.0],
            [0.0, 0.0],
            # quadprog stops on the first boundary, 5e-11 from the vertex, where the two
            # candidates' squared distances to z agree to rounding
            1e-10,
        ),
        "three lines through one vertex": (
            [3.0, 2.5],
            [[1.0, 0.0], [0.0, 1.0], [1.0, 1.0], [-1.0, 0.0], [0.0, -1.0]],
            [1.0, 1.0, 2.0, 1.0, 1.0],
            [1.0, 1.0],
            1e-12,
        ),
        "a 3-D box vertex with a fourth plane through it": (
            [2.0, 3.0, 2.5],
            [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0], [-1.0, 0.0, 0.0],
             [0.0, -1.0, 0.0], [0.0, 0.0, -1.0], [1.0, 1.0, 1.0]],
            [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 3.0],
            [1.0, 1.0, 1.0],
            1e-12,
        ),
    }

    def test_every_solver_reaches_the_vertex(self) -> None:
        """The reference is the vertex to within each case's tolerance, and after 200
        cycles every solver's squared error is within 1e-20 of the start's."""
        for name, (z, A, b, vertex, reference_atol) in self.CASES.items():
            z, A, b = np.array(z), np.array(A), np.array(b)
            with self.subTest(case=name):
                np.testing.assert_allclose(reference_projection(z, A, b), vertex,
                                           rtol=0.0, atol=reference_atol)
            for solver_type in SOLVERS:
                with self.subTest(case=name, solver=solver_type.__name__):
                    result, _ = run(solver_type, z, A, b, 200, track_error=True,
                                    plot_errors=True)
                    self.assertTrue(np.isfinite(result.path).all())
                    self.assertLessEqual(result.squared_errors[-1],
                                         1e-20 * result.squared_errors[0])
                    self.assertLessEqual(visit_excess(result, A, b), 1e-12)


class RedundantConstraintTests(unittest.TestCase):
    """Constraints implied by others, repeated, or repeated at another scale."""

    def test_redundant_rows_do_not_move_the_projection(self) -> None:
        """With any of these rows added to the box, (2, 2) still projects onto the vertex
        (1, 1), and every solver reaches it within 50 cycles."""
        box_A = BOX_LINE_A[:4]
        z = np.array([2.0, 2.0])
        variants = {
            "implied by the box": (np.vstack([box_A, [[1.0, 1.0]]]), np.append(np.ones(4), 3.0)),
            "a row twice": (np.vstack([box_A, box_A[:1]]), np.ones(5)),
            "a row and the same row times 1e3": (np.vstack([box_A, 1e3 * box_A[:1]]),
                                                 np.append(np.ones(4), 1e3)),
        }
        for name, (A, b) in variants.items():
            with self.subTest(case=name):
                np.testing.assert_allclose(reference_projection(z, A, b), [1.0, 1.0],
                                           rtol=0.0, atol=1e-14)
            for solver_type in SOLVERS:
                with self.subTest(case=name, solver=solver_type.__name__):
                    result, _ = run(solver_type, z, A, b, 50)
                    np.testing.assert_allclose(result.projection, [1.0, 1.0], rtol=0.0,
                                               atol=1e-12)

    def test_a_row_scaled_by_1e3_gives_the_same_path(self) -> None:
        """Normals are normalised before use, so scaling a row and its offset together
        changes nothing beyond rounding."""
        z = np.array([-2.0, 1.4])
        A, b = BOX_LINE_A.copy(), BOX_LINE_B.copy()
        A[4], b[4] = 1e3 * A[4], 1e3 * b[4]
        for solver_type in SOLVERS:
            with self.subTest(solver=solver_type.__name__):
                plain, plain_jumps = run(solver_type, z, BOX_LINE_A, BOX_LINE_B, 30)
                scaled, scaled_jumps = run(solver_type, z, A, b, 30)
                np.testing.assert_allclose(scaled.path, plain.path, rtol=0.0, atol=1e-12)
                self.assertEqual(scaled_jumps, plain_jumps)


class FeasibleStartTests(unittest.TestCase):
    """Starts inside some or all of the half-spaces."""

    def test_a_start_inside_the_intersection_never_moves(self) -> None:
        """The start is its own projection: the error is zero at every cycle, no half-space
        is ever active, and every correction vector stays zero."""
        problems = {"the box": (np.array([0.25, -0.5]), BOX_LINE_A[:4], np.ones(4))}
        for dimensions, num_planes, seed, _, _ in RANDOM_CASES[3:6]:
            _, A, b, anchor = random_case(dimensions, num_planes, seed)
            problems[f"{dimensions}-D anchor"] = (anchor, A, b)
        for name, (z, A, b) in problems.items():
            for solver_type in SOLVERS:
                with self.subTest(case=name, solver=solver_type.__name__):
                    result, jumps = run(solver_type, z, A, b, 10, track_error=True,
                                        plot_errors=True, plot_active_halfspaces=True)
                    np.testing.assert_array_equal(result.projection, z)
                    np.testing.assert_array_equal(
                        result.path, np.broadcast_to(z, result.path.shape))
                    np.testing.assert_array_equal(result.squared_errors, 0.0)
                    np.testing.assert_array_equal(result.active_half_spaces, 0.0)
                    np.testing.assert_array_equal(result.errors_for_plotting, 0.0)
                    self.assertEqual(jumps, [])

    def test_constraints_containing_the_start_still_bound_the_projection(self) -> None:
        """z = (-1, 0) lies in x <= 0 but not in x + y >= 2, and the projection (0, 2)
        lies on both boundaries."""
        z = np.array([-1.0, 0.0])
        A = np.array([[1.0, 0.0], [-1.0, -1.0]])
        b = np.array([0.0, -2.0])
        for solver_type in SOLVERS:
            with self.subTest(solver=solver_type.__name__):
                result, _ = run(solver_type, z, A, b, 80)
                np.testing.assert_allclose(result.projection, [0.0, 2.0], rtol=0.0,
                                           atol=1e-9)


class InvarianceTests(unittest.TestCase):
    """The path, its classification, and the fast-forward under scaling the problem and under
    an extra coordinate that no constraint touches.

    Scaling multiplies every point by the scale and every squared error by its square, so
    min_error is scaled with it. The extra coordinate holds 1e8, which no projection changes.
    """

    def problems(self):
        problems = {f"box and line from {z}": (np.array(z), BOX_LINE_A, BOX_LINE_B, 30)
                    for z in BOX_LINE_STALLS}
        problems["wedge"] = (WEDGE_Z, WEDGE_A, WEDGE_B, 30)
        z, A, b, _ = random_case(3, 8, 1002)
        problems["3-D random"] = (z, A, b, 40)
        return problems

    def assert_same_run(self, reference, reference_jumps, result, jumps, scale):
        np.testing.assert_allclose(result.path / scale, reference.path, rtol=0.0,
                                   atol=1e-9 * np.abs(reference.path).max())
        np.testing.assert_allclose(result.squared_errors / scale ** 2,
                                   reference.squared_errors, rtol=1e-6,
                                   atol=1e-12 * reference.squared_errors[0])
        np.testing.assert_array_equal(stall_mask(result), stall_mask(reference))
        np.testing.assert_array_equal(converged_mask(result), converged_mask(reference))
        self.assertEqual(jumps, reference_jumps)

    def test_scaling_the_problem_scales_the_run(self) -> None:
        for name, (z, A, b, max_iter) in self.problems().items():
            for solver_type in SOLVERS:
                reference, reference_jumps = run(solver_type, z, A, b, max_iter,
                                                 track_error=True)
                for scale in (1e-6, 1e6):
                    with self.subTest(case=name, solver=solver_type.__name__, scale=scale):
                        result, jumps = run(solver_type, scale * z, A, scale * b, max_iter,
                                            track_error=True, min_error=1e-3 * scale ** 2)
                        self.assert_same_run(reference, reference_jumps, result, jumps,
                                             scale)

    def test_an_untouched_coordinate_changes_nothing(self) -> None:
        for name, (z, A, b, max_iter) in self.problems().items():
            extended_z = np.append(z, 1e8)
            extended_A = np.hstack([A, np.zeros((A.shape[0], 1))])
            for solver_type in SOLVERS:
                with self.subTest(case=name, solver=solver_type.__name__):
                    reference, reference_jumps = run(solver_type, z, A, b, max_iter,
                                                     track_error=True)
                    result, jumps = run(solver_type, extended_z, extended_A, b, max_iter,
                                        track_error=True)
                    np.testing.assert_array_equal(result.path[..., -1], 1e8)
                    np.testing.assert_allclose(result.path[..., :-1], reference.path,
                                               rtol=0.0, atol=1e-12)
                    # The path agrees only to rounding, since np.dot sums vectors of
                    # another length in another order. The squared errors may differ by
                    # what that drift moves, and by nothing from the extra coordinate.
                    drift = np.linalg.norm(result.path[:, -1, :-1] - reference.path[:, -1],
                                           axis=1)
                    allowed = ((2 * np.sqrt(reference.squared_errors) + drift) * drift
                               + 8 * EPS * reference.squared_errors)
                    self.assertTrue(np.all(np.abs(result.squared_errors
                                                  - reference.squared_errors) <= allowed))
                    np.testing.assert_array_equal(stall_mask(result), stall_mask(reference))
                    np.testing.assert_array_equal(converged_mask(result),
                                                  converged_mask(reference))
                    self.assertEqual(jumps, reference_jumps)


class FastForwardTests(unittest.TestCase):
    """The stall detection solver's skip count on the paper's example."""

    def test_box_line_stall_is_skipped_in_one_positive_jump(self) -> None:
        """Each start's stall ends in one jump of the documented length. The fast-forwarded
        run is plain Dykstra's at the corresponding cycles, and its squared error never
        rises above the start's."""
        for start, stall in BOX_LINE_STALLS.items():
            with self.subTest(z=start):
                z = np.array(start)
                fast, jumps = run(DykstraStallDetectionSolver, z, BOX_LINE_A, BOX_LINE_B, 30,
                                  track_error=True)
                self.assertEqual(jumps, stall["jumps"])
                self.assertTrue(all(skip > 0 for _, skip in jumps))
                self.assertLessEqual(fast.squared_errors.max(), fast.squared_errors[0])

                skipped = sum(skip for _, skip in jumps)
                plain, _ = run(DykstraProjectionSolver, z, BOX_LINE_A, BOX_LINE_B,
                               30 + skipped, track_error=True)
                self.assertEqual(np.flatnonzero(stall_mask(plain)).tolist(),
                                 list(range(2, stall["last_stalled_cycle"] + 1)))
                for cycle in range(31):
                    plain_cycle = cycle + sum(skip for i, skip in jumps if i < cycle)
                    np.testing.assert_allclose(fast.path[cycle], plain.path[plain_cycle],
                                               rtol=0.0, atol=1e-12)
                    self.assertAlmostEqual(fast.squared_errors[cycle],
                                           plain.squared_errors[plain_cycle], places=12)


class EdgeCaseTests(unittest.TestCase):
    def test_zero_cycles_return_the_start(self) -> None:
        z = np.array([-2.0, 1.4])
        for solver_type in SOLVERS:
            with self.subTest(solver=solver_type.__name__):
                solver = solver_type(z, BOX_LINE_A, BOX_LINE_B, 0, track_error=True,
                                     plot_errors=True, plot_active_halfspaces=True)
                result = solver.solve()
                np.testing.assert_array_equal(result.projection, z)
                self.assertEqual(result.path.shape, (1, 6, 2))
                np.testing.assert_array_equal(result.path[0], np.tile(z, (6, 1)))
                self.assertEqual(result.squared_errors.shape, (1,))
                self.assertAlmostEqual(result.squared_errors[0], 4.16, places=12)
                self.assertEqual(result.errors_for_plotting.shape, (0, 6, 2))
                self.assertEqual(result.active_half_spaces.shape, (6, 1))

    def test_one_cycle_is_one_pass_of_projections(self) -> None:
        """With every correction zero at the start, the first cycle of each solver is the
        same pass of alternating projections."""
        z = np.array([-2.0, 1.4])
        unit_A, unit_b = normalised(BOX_LINE_A, BOX_LINE_B)
        x, outputs = z.copy(), []
        for normal, offset in zip(unit_A, unit_b):
            x = x - max(np.dot(x, normal) - offset, 0.0) * normal
            outputs.append(x)
        for solver_type in SOLVERS:
            with self.subTest(solver=solver_type.__name__):
                result, _ = run(solver_type, z, BOX_LINE_A, BOX_LINE_B, 1, track_error=True)
                np.testing.assert_allclose(result.path[1], outputs, rtol=0.0, atol=1e-15)
                np.testing.assert_allclose(result.projection, outputs[-1], rtol=0.0,
                                           atol=1e-15)
                self.assertEqual(result.squared_errors.shape, (2,))

    def test_a_single_constraint_is_projected_onto_in_one_cycle(self) -> None:
        z = np.array([3.0, 1.0, -2.0])
        A = np.array([[1.0, 2.0, -2.0]])
        b = np.array([-1.0])
        expected = z - (np.dot(A[0], z) - b[0]) / np.dot(A[0], A[0]) * A[0]
        for solver_type in SOLVERS:
            with self.subTest(solver=solver_type.__name__):
                result, jumps = run(solver_type, z, A, b, 5, track_error=True)
                np.testing.assert_allclose(result.path[1:, 0], np.tile(expected, (5, 1)),
                                           rtol=0.0, atol=1e-14)
                self.assertLessEqual(result.squared_errors[1:].max(), 1e-28)
                self.assertEqual(jumps, [])

    def test_an_empty_constraint_set_returns_the_start(self) -> None:
        z = np.array([2.0, -3.0, 0.5])
        for solver_type in SOLVERS:
            with self.subTest(solver=solver_type.__name__):
                result, _ = run(solver_type, z, np.empty((0, 3)), np.empty(0), 4,
                                track_error=True, plot_errors=True,
                                plot_active_halfspaces=True)
                np.testing.assert_array_equal(result.projection, z)
                self.assertEqual(result.path.shape, (5, 0, 3))
                np.testing.assert_array_equal(result.squared_errors, 0.0)
                self.assertTrue(converged_mask(result).all())
                self.assertEqual(result.errors_for_plotting.shape, (4, 0, 3))
                self.assertEqual(result.active_half_spaces.shape, (0, 5))

    def test_min_error_classifies_cycles_without_stopping_the_run(self) -> None:
        """The core solvers run every cycle of the budget whatever min_error is, and a cycle
        counts as converged when its squared error is below min_error."""
        z = np.array([-2.0, 1.4])
        for solver_type in SOLVERS:
            with self.subTest(solver=solver_type.__name__):
                strict, _ = run(solver_type, z, BOX_LINE_A, BOX_LINE_B, 30, track_error=True,
                                min_error=0.0)
                loose, _ = run(solver_type, z, BOX_LINE_A, BOX_LINE_B, 30, track_error=True,
                               min_error=0.5)
                np.testing.assert_array_equal(loose.path, strict.path)
                np.testing.assert_array_equal(converged_mask(loose),
                                              loose.squared_errors < 0.5)
                self.assertFalse(converged_mask(strict).any())
                self.assertNotEqual(loose.squared_errors[-1], loose.squared_errors[-2])

    def test_the_wrapper_returns_the_solver_result(self) -> None:
        z = np.array([-2.0, 1.4])
        result = DykstraProjectionSolver(z, BOX_LINE_A, BOX_LINE_B, 12, track_error=True,
                                         plot_errors=True,
                                         plot_active_halfspaces=True).solve()
        projection, path, errors, corrections, active = dykstra_projection(
            z, BOX_LINE_A, BOX_LINE_B, 12, track_error=True, plot_errors=True,
            plot_active_halfspaces=True)
        np.testing.assert_array_equal(projection, result.projection)
        np.testing.assert_array_equal(path, result.path)
        for returned, expected in zip(errors, (result.squared_errors, result.stalled_errors,
                                               result.converged_errors)):
            np.testing.assert_array_equal(returned, expected)
        np.testing.assert_array_equal(corrections, result.errors_for_plotting)
        np.testing.assert_array_equal(active, result.active_half_spaces)

        projection, path, errors, corrections, active = dykstra_projection(
            z, BOX_LINE_A, BOX_LINE_B, 12)
        np.testing.assert_array_equal(projection, result.projection)
        self.assertIsNone(errors)
        self.assertIsNone(corrections)
        self.assertIsNone(active)

    def test_delete_spaces_warns_and_keeps_every_constraint(self) -> None:
        """delete_spaces=True is a no-op that raises a DeprecationWarning naming the line that
        asked for it, which is what Python's default filters need to show it in a script."""
        z = np.array([-2.0, 1.4])
        reference, _ = run(DykstraProjectionSolver, z, BOX_LINE_A, BOX_LINE_B, 10)
        constructors = {
            solver_type.__name__: (
                lambda solver_type=solver_type: solver_type(
                    z, BOX_LINE_A, BOX_LINE_B, 10, delete_spaces=True))
            for solver_type in SOLVERS
        }
        for name, construct in constructors.items():
            with self.subTest(solver=name):
                with warnings.catch_warnings(record=True) as caught:
                    warnings.simplefilter("always")
                    solver = construct()
                self.assertEqual([w.category for w in caught], [DeprecationWarning])
                self.assertEqual(Path(caught[0].filename).name, Path(__file__).name)
                self.assertEqual(solver.A.shape, BOX_LINE_A.shape)
                with contextlib.redirect_stdout(io.StringIO()):
                    result = solver.solve()
                if name == "DykstraProjectionSolver":
                    np.testing.assert_array_equal(result.path, reference.path)
        with self.subTest(solver="dykstra_projection"):
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                projection, *_ = dykstra_projection(z, BOX_LINE_A, BOX_LINE_B, 10,
                                                    delete_spaces=True)
            self.assertEqual([w.category for w in caught], [DeprecationWarning])
            self.assertEqual(Path(caught[0].filename).name, Path(__file__).name)
            np.testing.assert_array_equal(projection, reference.projection)


if __name__ == "__main__":
    unittest.main()
