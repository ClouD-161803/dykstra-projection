"""Regression coverage for the LTI Dykstra experiment family."""

from pathlib import Path
import sys
import unittest

import numpy as np


WORKING_VERSION = Path(__file__).resolve().parents[1] / "Python" / "Working Version"
sys.path.insert(0, str(WORKING_VERSION))

from convex_projection_solver import DykstraProjectionSolver
from lti_ver1 import LTIVer1Solver
from lti_ver2 import LTIVer2Solver
from lti_ver3 import LTIVer3Solver
from lti_ver4 import LTIVer4Solver
from lti_ver5 import LTIVer5Solver
from oracle import oracle_lti_projection, record_schedule


LTI_SOLVERS = (
    LTIVer1Solver,
    LTIVer2Solver,
    LTIVer3Solver,
    LTIVer4Solver,
    LTIVer5Solver,
)


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


class LTISolverRegressionTests(unittest.TestCase):
    def test_all_versions_reach_the_qp_reference_on_representative_problems(self) -> None:
        z_2d, A_2d, b_2d = box_line_problem()
        cases = {
            "activity switch": (z_2d, A_2d, b_2d),
            "three-dimensional box": (
                np.array([2.0, -3.0, 4.0]),
                np.vstack([np.eye(3), -np.eye(3)]),
                np.ones(6),
            ),
            "rank-deficient line": (
                np.array([2.0, 3.0]),
                np.array([[2.0, 0.0], [-3.0, 0.0]]),
                np.array([0.0, 0.0]),
            ),
        }

        for case_name, (z, A, b) in cases.items():
            reference_solver = DykstraProjectionSolver(z, A, b, max_iter=200)
            standard_dykstra = reference_solver.solve().projection
            qp_reference = reference_solver.actual_projection
            np.testing.assert_allclose(standard_dykstra, qp_reference, atol=1e-7)

            for solver_type in LTI_SOLVERS:
                with self.subTest(case=case_name, solver=solver_type.__name__):
                    result = solver_type(z, A, b, max_iter=200).solve()
                    np.testing.assert_allclose(result.projection, standard_dykstra, atol=1e-7)
                    np.testing.assert_allclose(result.projection, qp_reference, atol=1e-7)
                    self.assertTrue(np.all(A @ result.projection <= b + 1e-8))

    def test_zero_iterations_returns_the_initial_point_for_every_version(self) -> None:
        z = np.array([2.0, 0.0])
        A = np.array([[1.0, 0.0]])
        b = np.array([1.0])

        for solver_type in LTI_SOLVERS:
            with self.subTest(solver=solver_type.__name__):
                result = solver_type(z, A, b, max_iter=0, track_error=True).solve()
                np.testing.assert_array_equal(result.projection, z)
                self.assertEqual(result.path.shape, (1, 1, 2))
                self.assertEqual(result.squared_errors.shape, (1,))
                self.assertAlmostEqual(result.squared_errors[0], 1.0)

    def test_ver5_rejects_invalid_acceleration_options(self) -> None:
        z = np.array([2.0, 0.0])
        A = np.array([[1.0, 0.0]])
        b = np.array([1.0])

        for block_size in (0, -1, True, 1.5):
            with self.subTest(block_size=block_size):
                with self.assertRaisesRegex(ValueError, "block_size"):
                    LTIVer5Solver(z, A, b, max_iter=1, block_size=block_size)

        for eig_cond_cap in (1.0, 0.0, np.inf, "not a number"):
            with self.subTest(eig_cond_cap=eig_cond_cap):
                with self.assertRaisesRegex(ValueError, "eig_cond_cap"):
                    LTIVer5Solver(z, A, b, max_iter=1, eig_cond_cap=eig_cond_cap)

    def test_oracle_replays_the_recorded_activity_schedule(self) -> None:
        z, A, b = box_line_problem()
        reference = DykstraProjectionSolver(z, A, b, max_iter=200).actual_projection

        schedule, recorded = record_schedule(z, A, b, max_iter=200)
        replayed = oracle_lti_projection(z, A, b, schedule)

        self.assertGreater(len(schedule), 1)
        self.assertLessEqual(sum(cycles for _, cycles in schedule), 200)
        np.testing.assert_allclose(recorded.projection, reference, atol=1e-7)
        np.testing.assert_allclose(replayed.projection, reference, atol=1e-7)

    def test_settlement_is_reported_and_recorded_from_its_cycle(self) -> None:
        # Dykstra reaches (-0.25, 0.25) in two cycles and the projection (0, 0) only in
        # the limit; Ver2-4 prove the active set final on cycle 2
        z, A, b = np.array([2.0, 1.0]), np.array([[1.0, 0.0], [1.0, 1.0]]), np.zeros(2)
        certificates = {LTIVer1Solver: None, LTIVer2Solver: "finality", LTIVer3Solver: "finality",
                        LTIVer4Solver: "finality"}
        for max_iter in (2, 4):
            dykstra = DykstraProjectionSolver(z, A, b, max_iter=max_iter, track_error=True).solve()
            self.assertFalse(dykstra.is_settled())
            for solver_type in certificates:
                with self.subTest(solver=solver_type.__name__, max_iter=max_iter):
                    result = solver_type(z, A, b, max_iter=max_iter, track_error=True).solve()
                    self.assertEqual(result.certificate, certificates[solver_type])
                    if result.certificate is None:
                        self.assertIsNone(result.settled_at)
                        np.testing.assert_allclose(result.projection, dykstra.projection, atol=1e-12)
                        continue

                    # From the settling cycle on, every record is the returned limit
                    self.assertTrue(result.is_settled())
                    first = result.settled_at
                    np.testing.assert_allclose(result.projection, [0.0, 0.0], atol=1e-12)
                    np.testing.assert_allclose(result.path[first:, -1],
                                               np.tile(result.projection, (max_iter + 1 - first, 1)), atol=1e-12)
                    np.testing.assert_allclose(result.squared_errors[first:], 0.0, atol=1e-12)
                    np.testing.assert_allclose(result.squared_errors[:first], dykstra.squared_errors[:first],
                                               atol=1e-12)

    def test_exported_results_carry_the_settlement(self) -> None:
        import contextlib
        import io
        import tempfile
        from visualiser import ResultExporter

        z, A, b = np.array([2.0, 1.0]), np.array([[1.0, 0.0], [1.0, 1.0]]), np.zeros(2)
        for solver_type in (DykstraProjectionSolver, LTIVer5Solver):
            with self.subTest(solver=solver_type.__name__), tempfile.TemporaryDirectory() as directory:
                result = solver_type(z, A, b, max_iter=4, track_error=True).solve()
                self.assertEqual(result.is_settled(), solver_type is LTIVer5Solver)
                path = str(Path(directory) / "result.csv")
                with contextlib.redirect_stdout(io.StringIO()):
                    ResultExporter.export(result, path, solver_type.__name__, z, A, b, 4)
                metadata = ResultExporter.load(path)["metadata"]
                if result.is_settled():
                    self.assertEqual(metadata["settled_at"], result.settled_at)
                    self.assertEqual(metadata["certificate"], result.certificate)
                else:
                    self.assertNotIn("settled_at", metadata)


if __name__ == "__main__":
    unittest.main()
