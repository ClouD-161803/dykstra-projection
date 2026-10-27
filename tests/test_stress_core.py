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


if __name__ == "__main__":
    unittest.main()
