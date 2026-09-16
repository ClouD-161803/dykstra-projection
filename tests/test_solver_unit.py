"""Unit tests for half-space projection and solver input handling."""

from pathlib import Path
import sys
import unittest

import numpy as np


WORKING_VERSION = Path(__file__).resolve().parents[1] / "Python" / "Working Version"
sys.path.insert(0, str(WORKING_VERSION))

from convex_projection_solver import ConvexProjectionSolver, DykstraProjectionSolver


class HalfSpaceProjectionTests(unittest.TestCase):
    def test_projection_leaves_a_feasible_point_unchanged(self) -> None:
        point = np.array([-2.0, 1.0])

        projected = ConvexProjectionSolver._project_onto_half_space(
            point, np.array([2.0, 0.0]), 0.0
        )

        np.testing.assert_array_equal(projected, point)

    def test_projection_handles_a_non_unit_normal(self) -> None:
        projected = ConvexProjectionSolver._project_onto_half_space(
            np.array([3.0, 1.0]), np.array([2.0, 0.0]), 2.0
        )

        np.testing.assert_allclose(projected, [1.0, 1.0])


class SolverInputValidationTests(unittest.TestCase):
    def setUp(self) -> None:
        self.z = np.array([0.0, 0.0])
        self.A = np.array([[1.0, 0.0]])
        self.b = np.array([1.0])

    def test_rejects_mismatched_constraint_shapes(self) -> None:
        with self.assertRaisesRegex(ValueError, "same number of constraints"):
            DykstraProjectionSolver(
                self.z, self.A, np.array([1.0, 2.0]), max_iter=1
            )

    def test_rejects_zero_normals(self) -> None:
        with self.assertRaisesRegex(ValueError, "zero-norm"):
            DykstraProjectionSolver(
                self.z, np.array([[0.0, 0.0]]), self.b, max_iter=1
            )

    def test_rejects_negative_iteration_count(self) -> None:
        with self.assertRaisesRegex(ValueError, "non-negative integer"):
            DykstraProjectionSolver(self.z, self.A, self.b, max_iter=-1)

    def test_rejects_negative_error_threshold(self) -> None:
        with self.assertRaisesRegex(ValueError, "non-negative"):
            DykstraProjectionSolver(
                self.z, self.A, self.b, max_iter=1, min_error=-1.0
            )

    def test_accepts_a_and_b_keyword_arguments(self) -> None:
        solver = DykstraProjectionSolver(
            z=np.array([2.0, 0.0]),
            A=self.A,
            b=self.b,
            max_iter=1,
        )

        np.testing.assert_array_equal(solver.A, self.A)
        np.testing.assert_array_equal(solver.b, self.b)


class SolverEdgeCaseTests(unittest.TestCase):
    def test_empty_constraint_set_returns_the_original_point(self) -> None:
        z = np.array([2.0, -3.0])
        solver = DykstraProjectionSolver(
            z, np.empty((0, 2)), np.empty(0), max_iter=2
        )

        result = solver.solve()

        np.testing.assert_array_equal(solver.actual_projection, z)
        np.testing.assert_array_equal(result.projection, z)
        self.assertEqual(result.path.shape, (3, 0, 2))

    def test_legacy_pruning_helper_keeps_every_constraint(self) -> None:
        A = np.array([[1.0, 0.0], [-1.0, -1.0]])
        b = np.array([0.0, -2.0])

        with self.assertWarns(DeprecationWarning):
            retained_A, retained_b = ConvexProjectionSolver._delete_inactive_half_spaces(
                np.array([-1.0, 0.0]), A, b
            )

        np.testing.assert_array_equal(retained_A, A)
        np.testing.assert_array_equal(retained_b, b)
        self.assertIsNot(retained_A, A)
        self.assertIsNot(retained_b, b)


if __name__ == "__main__":
    unittest.main()
