"""Contract tests for the C solvers; skipped until the libraries in C/ are built."""

import unittest

import numpy as np

from dykstra.c_solvers import (
    MAX_CYCLES,
    accelerated_c_projection,
    dykstra_c_projection,
    libraries_built,
)
from dykstra.convex_projection_solver import DykstraProjectionSolver
from dykstra.paper_figures import bounded_problem, infeasible_start


def box_line_problem() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return a 2-D problem whose active set changes during projection."""
    z = np.array([-2.0, 1.4])
    A = np.array([
        [1.0, 0.0],
        [-1.0, 0.0],
        [0.0, 1.0],
        [0.0, -1.0],
        [0.5, 1.0],
        [-0.5, -1.0],
    ])
    b = np.array([1.0, 1.0, 1.0, 1.0, 1.0, -1.0])
    return z, A, b


def bounded_problems() -> list:
    """Return seeded bounded polyhedra with a start outside each."""
    problems = []
    for num_planes, num_dimensions in ((12, 4), (24, 8)):
        for seed in range(3):
            np.random.seed(1000 * num_planes + seed)
            A, b = bounded_problem(num_planes, num_dimensions)
            problems.append((infeasible_start(A, b, num_dimensions), A, b))
    return problems


@unittest.skipUnless(libraries_built(), "build the C solvers first, see the README")
class CSolverTests(unittest.TestCase):
    def test_plain_dykstra_matches_the_python_iterate_and_corrections(self) -> None:
        for z, A, b in [box_line_problem()] + bounded_problems():
            reference = DykstraProjectionSolver(z, A, b, max_iter=60, plot_errors=True).solve()
            scale = np.abs(z).max()
            for max_iter in (1, 2, 3, 7, 20, 60):
                with self.subTest(constraints=A.shape[0], max_iter=max_iter):
                    result = dykstra_c_projection(z, A, b, max_iter)
                    self.assertEqual(result.exact_cycles, result.cycles)
                    np.testing.assert_allclose(result.projection, reference.path[max_iter, -1],
                                               rtol=0.0, atol=1e-12 * scale)
                    # Dykstra's correction for a row is its multiplier times the row
                    np.testing.assert_allclose(result.multipliers[:, None] * A,
                                               reference.errors_for_plotting[max_iter - 1],
                                               rtol=0.0, atol=1e-12 * scale)

    def test_accelerated_solver_returns_the_iterate_or_the_projection_at_every_budget(self) -> None:
        # Without a recorded history the path is checked by solving at every budget:
        # an unsettled result is Dykstra's own iterate there, a settled one the projection
        for z, A, b in [box_line_problem()] + bounded_problems():
            reference_solver = DykstraProjectionSolver(z, A, b, max_iter=120)
            path = reference_solver.solve().path
            scale = np.abs(z).max()
            for max_iter in range(1, 121):
                with self.subTest(constraints=A.shape[0], max_iter=max_iter):
                    result = accelerated_c_projection(z, A, b, max_iter)
                    expected = reference_solver.actual_projection if result.settled else path[max_iter, -1]
                    np.testing.assert_allclose(result.projection, expected, rtol=0.0, atol=1e-9 * scale)
                    if not result.settled:
                        self.assertEqual(result.cycles, max_iter)
                    # The first cycle is always exact, and no cycle is both exact and skipped
                    self.assertGreaterEqual(result.exact_cycles, 1)
                    self.assertLessEqual(result.exact_cycles + result.cycles_skipped, max_iter)

    def test_seeded_problems_return_the_iterate_or_the_projection(self) -> None:
        # The problems of the LTI suite's seeded test: nearly duplicated rows, an equality
        # written as two half-spaces, and a coordinate of 1e7 that no constraint touches
        rng = np.random.default_rng(20260923)
        for trial in range(30):
            p, n = int(rng.integers(2, 6)), int(rng.integers(1, 9))
            A = rng.normal(size=(n, p))
            if trial % 3 == 1 and n >= 2:
                A[1] = A[0] + 1e-4 * rng.normal(size=p)
            x0 = rng.normal(size=p)
            b = A @ x0 + rng.uniform(0.0, 1.0, size=n)
            if trial % 3 == 2 and n >= 2:
                A[1], b[0] = -A[0], A[0] @ x0
                b[1] = -b[0]
            z = x0 + 3.0 * rng.normal(size=p)
            if trial % 5 == 4:
                z, A = np.append(z, 1e7), np.hstack([A, np.zeros((n, 1))])
            touched = np.any(A != 0.0, axis=0)
            scale = np.abs(z[touched]).max()

            reference_solver = DykstraProjectionSolver(z, A, b, max_iter=300)
            dykstra = reference_solver.solve().projection
            with self.subTest(trial=trial):
                result = accelerated_c_projection(z, A, b, 300)
                expected = reference_solver.actual_projection if result.settled else dykstra
                np.testing.assert_allclose(result.projection[touched], expected[touched],
                                           rtol=0.0, atol=1e-9 * scale)
                np.testing.assert_array_equal(result.projection[~touched], z[~touched])

    def test_an_unrelated_coordinate_does_not_change_the_result(self) -> None:
        # A feasibility tolerance scaled by the largest coordinate once settled this
        # problem at (-0.8, 1.4), which violates the second row by 0.4
        z = np.array([-2.0, 1.4])
        A = np.array([[-1.0, 0.0], [0.0, 1.0], [-0.5, -1.0]])
        b = np.array([1.0, 1.0, -1.0])
        plain = accelerated_c_projection(z, A, b, 200)
        self.assertTrue(plain.settled)
        np.testing.assert_allclose(plain.projection, [0.0, 1.0], rtol=0.0, atol=1e-12)
        for extra in (1e3, 1e9, 1e12):
            with self.subTest(untouched=extra):
                lifted = accelerated_c_projection(np.append(z, extra), np.hstack([A, np.zeros((3, 1))]), b, 200)
                self.assertEqual(lifted.settled, plain.settled)
                self.assertEqual(lifted.cycles, plain.cycles)
                np.testing.assert_allclose(lifted.projection[:2], plain.projection, rtol=0.0, atol=1e-12)
                self.assertEqual(lifted.projection[2], extra)

    def test_an_untouched_coordinate_does_not_change_the_settlement(self) -> None:
        # Every episode of this problem is regular in the plane. A coordinate no constraint
        # touches once made I - A_m singular, so the solver took its deflated path and
        # settled after 1 cycle, where the plane settles after 5
        z = np.array([-1.75, -1.5])
        A = np.array([[-0.5, -1.0], [0.5, -0.25], [-0.5, -0.25]])
        b = np.array([0.375, 0.1875, -0.0625])
        plain = accelerated_c_projection(z, A, b, 200)
        self.assertTrue(plain.settled)
        for position in (0, 2):
            with self.subTest(position=position):
                lifted = accelerated_c_projection(np.insert(z, position, 1e8),
                                                  np.insert(A, position, 0.0, axis=1), b, 200)
                self.assertEqual((lifted.settled, lifted.cycles, lifted.exact_cycles, lifted.cycles_skipped),
                                 (plain.settled, plain.cycles, plain.exact_cycles, plain.cycles_skipped))
                np.testing.assert_allclose(np.delete(lifted.projection, position), plain.projection,
                                           rtol=0.0, atol=1e-12)
                np.testing.assert_allclose(lifted.multipliers, plain.multipliers, rtol=0.0, atol=1e-12)
                self.assertEqual(lifted.projection[position], 1e8)

    def test_certificate_accepts_a_support_row_the_step_does_not_move(self) -> None:
        # In each problem a row through the origin binds at the projection along a
        # coordinate where z is already zero. Least squares leaves about 1e-16 of the
        # rest of the correction's rounding in that coordinate, and a slack floor built
        # from that coordinate alone, about 1e-30, read the row as inactive or violated,
        # so the true support failed. None of the three was ever certified, where the
        # Python certifies the six-row problem on entering cycle 2
        cases = {
            "six rows": (np.array([-5.5, 0.0]),
                         np.array([[0.0, -2.0], [-0.5, 0.5], [-0.5, 0.5], [0.0, 1.0], [1.0, 0.0], [0.5, 1.0]]),
                         np.array([0.0, 0.75, -0.25, 1.5, 1.5, 1.5])),
            "row not tight": (np.array([-4.0, 0.0]),
                              np.array([[1.0, 0.0], [0.0, 1.0], [-0.5, -1.0], [0.0, -0.5]]),
                              np.array([-1.0, 0.0, 1.75, 0.5])),
            "row violated": (np.array([4.5, 0.0, 2.5]),
                             np.array([[-1.5, 1.0, 1.0], [1.5, 0.5, -0.5], [0.0, -0.5, 0.0]]),
                             np.array([0.5, 1.5, 0.0])),
        }
        for case_name, (z, A, b) in cases.items():
            with self.subTest(case=case_name):
                result = accelerated_c_projection(z, A, b, 200)
                self.assertTrue(result.settled)
                np.testing.assert_allclose(result.projection,
                                           DykstraProjectionSolver(z, A, b, max_iter=1).actual_projection,
                                           rtol=0.0, atol=1e-12)

    def test_no_constraints_settle_after_no_cycle(self) -> None:
        # Without constraints z is its own projection, which both C solvers report as
        # settled after no cycle. With no budget the call never reached them and came
        # back unsettled
        z = np.array([1.0, 2.0])
        for solve in (accelerated_c_projection, dykstra_c_projection):
            for max_iter in (0, 5):
                with self.subTest(solver=solve.__name__, max_iter=max_iter):
                    result = solve(z, np.zeros((0, 2)), np.zeros(0), max_iter)
                    self.assertEqual((result.settled, result.cycles), (True, 0))
                    np.testing.assert_array_equal(result.projection, z)

    def test_certificate_accepts_a_tight_row_without_a_multiplier(self) -> None:
        # Rows 1, 4 and 5 are tight at the projection, and row 4, the only one of them
        # that touches the second coordinate, carries no multiplier, so the step there is
        # zero. The rounding least squares left there exceeded a stationarity allowance
        # built from that coordinate alone, so the projection was never certified
        z = np.array([-1.5, 0.0, 1.0])
        A = np.array([[1.5, 0.5, 0.0], [-2.0, 0.0, 0.5], [0.0, 0.0, 0.5], [-0.5, -0.5, 2.5],
                      [-1.0, -0.5, 3.0], [0.5, 0.0, 1.0]])
        b = np.array([0.75, 1.0, 0.5, 0.75, 0.5, -0.25])
        result = accelerated_c_projection(z, A, b, 200)
        self.assertTrue(result.settled)
        np.testing.assert_allclose(result.projection,
                                   DykstraProjectionSolver(z, A, b, max_iter=1).actual_projection,
                                   rtol=0.0, atol=1e-12)

    def test_certificate_dual_exchanges_a_dependent_row(self) -> None:
        # Ten rows in the plane, three of them with the normal (-1, 0), give the dual of
        # the candidate rows a singular Gram matrix. Lawson-Hanson met a row dependent on
        # its support, could not bring it in, and stopped at a support the KKT test
        # rejects, so the projection was certified only after 10 cycles, where the
        # Python certifies it on entering cycle 2
        z = np.array([0.5, -2.5])
        A = np.array([[-0.5, 3.0], [0.5, -0.5], [0.5, -1.0], [1.0, -1.0], [2.5, -2.0],
                      [-1.5, 0.0], [-0.5, 0.0], [-1.0, 0.0], [1.5, -0.5], [-2.0, -0.5]])
        b = np.array([1.0, 0.5, 0.5, 1.5, 3.0, -2.0, -0.5, -1.0, 2.5, -2.75])
        result = accelerated_c_projection(z, A, b, 200)
        self.assertTrue(result.settled)
        self.assertEqual(result.cycles, 1)
        np.testing.assert_allclose(result.projection,
                                   DykstraProjectionSolver(z, A, b, max_iter=1).actual_projection,
                                   rtol=0.0, atol=1e-12)

    def test_the_result_scales_with_the_problem(self) -> None:
        for z, A, b in [box_line_problem()] + bounded_problems()[:2]:
            for max_iter in (5, 400):
                plain = accelerated_c_projection(z, A, b, max_iter)
                for scale in (1e-9, 1e9):
                    with self.subTest(constraints=A.shape[0], max_iter=max_iter, scale=scale):
                        scaled = accelerated_c_projection(scale * z, A, scale * b, max_iter)
                        self.assertEqual(scaled.settled, plain.settled)
                        self.assertEqual(scaled.cycles, plain.cycles)
                        np.testing.assert_allclose(scaled.projection / scale, plain.projection,
                                                   rtol=0.0, atol=1e-9 * np.abs(z).max())

    def test_settled_multipliers_satisfy_the_kkt_conditions(self) -> None:
        for z, A, b in [box_line_problem()] + bounded_problems():
            result = accelerated_c_projection(z, A, b, 100000)
            with self.subTest(constraints=A.shape[0]):
                self.assertTrue(result.settled)
                scale = np.abs(z).max()
                slacks = A @ result.projection - b
                self.assertTrue(np.all(slacks <= 1e-9 * scale))
                self.assertTrue(np.all(result.multipliers >= 0.0))
                np.testing.assert_allclose(A.T @ result.multipliers, z - result.projection,
                                           rtol=0.0, atol=1e-9 * scale)
                # A multiplier is positive only on a row that binds
                self.assertTrue(np.all(np.abs(slacks[result.multipliers > 0.0]) <= 1e-9 * scale))

    def test_degenerate_calls(self) -> None:
        z = np.array([2.0, 0.0])
        A = np.array([[1.0, 0.0]])
        b = np.array([1.0])
        for solve in (accelerated_c_projection, dykstra_c_projection):
            with self.subTest(solver=solve.__name__):
                # No constraints, and no cycles, both leave the point where it is
                np.testing.assert_array_equal(solve(z, np.zeros((0, 2)), np.zeros(0), 10).projection, z)
                unmoved = solve(z, A, b, 0)
                np.testing.assert_array_equal(unmoved.projection, z)
                self.assertFalse(unmoved.settled)
                np.testing.assert_allclose(solve(z, A, b, 10).projection, [1.0, 0.0], rtol=0.0, atol=1e-12)
                for bad in ((z, np.zeros((1, 2)), b, 10), (np.array([np.nan, 0.0]), A, b, 10),
                            (z, A, b, MAX_CYCLES + 1), (z, A, b, -1)):
                    with self.assertRaises(ValueError):
                        solve(*bad)


if __name__ == "__main__":
    unittest.main()
