"""Contract tests for the C solvers; skipped until the libraries in C/ are built."""

from pathlib import Path
import sys
import unittest

import numpy as np


WORKING_VERSION = Path(__file__).resolve().parents[1] / "Python" / "Working Version"
sys.path.insert(0, str(WORKING_VERSION))

from c_solvers import (
    MAX_CYCLES,
    accelerated_c_projection,
    dykstra_c_projection,
    libraries_built,
)
from convex_projection_solver import DykstraProjectionSolver
from paper_figures import bounded_problem, infeasible_start


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
