"""Regression tests for solver behavior that has failed in real examples."""

from pathlib import Path
import sys
import unittest

import numpy as np


WORKING_VERSION = Path(__file__).resolve().parents[1] / "Python" / "Working Version"
sys.path.insert(0, str(WORKING_VERSION))

from convex_projection_solver import DykstraProjectionSolver


class SolverRegressionTests(unittest.TestCase):
    def test_constraints_satisfied_at_the_start_are_retained(self) -> None:
        """Initial membership alone must not change the projection problem."""
        z = np.array([-1.0, 0.0])
        A = np.array([
            [1.0, 0.0],    # x <= 0
            [-1.0, -1.0],  # x + y >= 2
        ])
        b = np.array([0.0, -2.0])

        with self.assertWarns(DeprecationWarning):
            solver = DykstraProjectionSolver(
                z, A, b, max_iter=80, delete_spaces=True
            )
        result = solver.solve()

        self.assertEqual(solver.A.shape[0], 2)
        np.testing.assert_allclose(solver.actual_projection, [0.0, 2.0], atol=1e-10)
        np.testing.assert_allclose(result.projection, [0.0, 2.0], atol=1e-6)

    def test_dimensions_are_inferred_from_the_point(self) -> None:
        z = np.array([2.0, -2.0, 3.0])
        A = np.array([
            [1.0, 0.0, 0.0],
            [0.0, -1.0, 0.0],
            [0.0, 0.0, 1.0],
        ])
        b = np.array([1.0, 1.0, 1.0])

        solver = DykstraProjectionSolver(z, A, b, max_iter=2)

        self.assertEqual(solver.dimensions, 3)
        np.testing.assert_allclose(solver.actual_projection, [1.0, -1.0, 1.0])

    def test_mismatched_explicit_dimensions_are_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "dimensions"):
            DykstraProjectionSolver(
                np.array([0.0, 0.0, 0.0]),
                np.array([[1.0, 0.0, 0.0]]),
                np.array([1.0]),
                max_iter=1,
                dimensions=2,
            )

    def test_zero_iterations_tracks_the_initial_error(self) -> None:
        solver = DykstraProjectionSolver(
            np.array([2.0, 0.0]),
            np.array([[1.0, 0.0]]),
            np.array([1.0]),
            max_iter=0,
            track_error=True,
        )

        result = solver.solve()

        self.assertEqual(result.squared_errors.shape, (1,))
        self.assertAlmostEqual(result.squared_errors[0], 1.0)


if __name__ == "__main__":
    unittest.main()
