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


def expected_result(result, solver, dykstra_iterate: np.ndarray) -> np.ndarray:
    """What an LTI solver returns: the projection once settled, else Dykstra's iterate."""
    return solver.actual_projection if result.is_settled() else dykstra_iterate


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

    def test_finality_is_invariant_under_scaling_the_problem(self) -> None:
        # Scaling z and b by c scales every iterate and auxiliary by c, so a draining
        # half-space must not pass the finality test because the problem is small
        box_z, box_A, box_b = box_line_problem()
        cases = {
            "activity switch": (box_z, box_A, box_b, 30),
            "draining vertex": (
                np.array([-1.0, -1.0]),
                np.array([[1.0, -1.0], [0.0, -1.0], [-2.0, 0.0]]),
                np.array([-1.0, 0.0, 0.0]),
                200,
            ),
        }

        for case_name, (z, A, b, max_iter) in cases.items():
            unscaled = {solver_type: solver_type(z, A, b, max_iter=max_iter).solve().projection
                        for solver_type in LTI_SOLVERS}
            for scale in (1e-9, 1e-12):
                reference_solver = DykstraProjectionSolver(scale * z, A, scale * b, max_iter=max_iter)
                standard_dykstra = reference_solver.solve().projection
                qp_reference = reference_solver.actual_projection
                for solver_type in LTI_SOLVERS:
                    with self.subTest(case=case_name, scale=scale, solver=solver_type.__name__):
                        result = solver_type(scale * z, A, scale * b, max_iter=max_iter).solve()
                        projection = result.projection
                        np.testing.assert_allclose(projection / scale, unscaled[solver_type], rtol=0.0, atol=1e-9)

                        # The budgeted Dykstra iterate or, after a settle, the projection
                        off_path = np.max(np.abs(projection - standard_dykstra)) / scale
                        off_qp = np.max(np.abs(projection - qp_reference)) / scale
                        self.assertLess(min(off_path, off_qp), 1e-9)
                        if result.is_settled():
                            self.assertLess(off_qp, 1e-9)

    def test_draining_active_half_space_does_not_pass_the_finality_test(self) -> None:
        # Half-space 0 stays active at the episode limit but drains by about 5e-16 per
        # cycle, so that limit is not the projection, which sits near (5e-4, 0)
        z = np.array([-1.0, 1.0])
        A = np.array([[-1.0, 0.0], [0.0, 1.0], [-1e-6, -1.0]])
        b = np.array([0.0, 0.0, -5e-10])
        dykstra = DykstraProjectionSolver(z, A, b, max_iter=200).solve().projection
        for solver_type in LTI_SOLVERS:
            with self.subTest(solver=solver_type.__name__):
                solver = solver_type(z, A, b, max_iter=200)
                result = solver.solve()
                self.assertNotEqual(result.certificate, "finality")
                # Rows 1 and 2 meet at 1e-6 rad, so the projection is known only to about 1e-10
                np.testing.assert_allclose(result.projection, expected_result(result, solver, dykstra),
                                           rtol=0.0, atol=1e-9 if result.is_settled() else 1e-12)

    def test_closed_form_stays_on_the_dykstra_path_when_the_resolvent_is_ill_conditioned(self) -> None:
        # Two normals 1e-5 rad apart give cond(I - A_m) of about 2.3e10, below the cutoff
        # of the closed form; an explicit inverse loses about cond * eps there
        theta = 1e-5
        z = np.array([3.0, 2.0, 0.0])
        A = np.array([[1.0, 0.0, 0.0], [np.cos(theta), np.sin(theta), 0.0], [1.0, 1.0, -1.0]])
        b = np.array([1.0, 1.0, -1.0])

        dykstra = DykstraProjectionSolver(z, A, b, max_iter=2).solve().projection
        for solver_type in LTI_SOLVERS:
            with self.subTest(solver=solver_type.__name__, max_iter=2):
                solver = solver_type(z, A, b, max_iter=2)
                result = solver.solve()
                np.testing.assert_allclose(result.projection, expected_result(result, solver, dykstra),
                                           rtol=0.0, atol=1e-10)

            with self.subTest(solver=solver_type.__name__, max_iter=2000):
                solver = solver_type(z, A, b, max_iter=2000)
                result = solver.solve()
                np.testing.assert_allclose(result.projection, solver.actual_projection, rtol=0.0, atol=1e-10)

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

    def test_closed_form_fixed_point_is_solved_rather_than_inverted(self) -> None:
        # Rows 0 and 1 are 1.5e-4 rad apart, so cond(I - A_m) is about 5.8e7, inside
        # the closed form; its first cycle x_2 = A_m x_1 + (I - A_m) x_inf returns
        # A_m x_1 + B_m only if x_inf solves the system, and an explicit inverse leaves
        # a residual of about cond * eps that shows in the returned iterate
        z = np.array([4.9807, -2.0211, 4.3121])
        A = np.array([[1.8014, -2.1326, 0.6021],
                      [1.2614, -1.4932, 0.4213],
                      [0.683, 0.4823, 0.8875],
                      [-0.8541, 0.223, -0.7151],
                      [1.2629, 0.8911, 1.6405]])
        b = np.array([6.8518, 4.7968, 2.3995, -0.6644, 4.4364])
        for max_iter in (3, 20):
            dykstra = DykstraProjectionSolver(z, A, b, max_iter=max_iter).solve().projection
            for solver_type in LTI_SOLVERS:
                with self.subTest(solver=solver_type.__name__, max_iter=max_iter):
                    solver = solver_type(z, A, b, max_iter=max_iter)
                    result = solver.solve()
                    np.testing.assert_allclose(result.projection, expected_result(result, solver, dykstra),
                                               rtol=0.0, atol=1e-12)
                    if not result.is_settled():
                        np.testing.assert_allclose(result.projection, result.path[-1, -1],
                                                   rtol=0.0, atol=1e-12)

    def test_oracle_jump_matches_dykstra_when_the_resolvent_is_ill_conditioned(self) -> None:
        # Normals about 1e-4 rad apart give cond(I - A_m) of about 7e7, below the cutoff
        # of the closed form; an explicit inverse loses about 1e-7 in a single jumped cycle
        z = np.array([71.0, -21.5])
        A = np.array([[0.75, -1.5], [0.75016, -1.49988]])
        b = A @ np.array([65.0, -24.0])
        dykstra = DykstraProjectionSolver(z, A, b, max_iter=2).solve().projection

        schedule, recorded = record_schedule(z, A, b, max_iter=2)
        replayed = oracle_lti_projection(z, A, b, schedule)

        self.assertEqual([cycles for _, cycles in schedule], [2])
        np.testing.assert_array_equal(recorded.projection, dykstra)
        np.testing.assert_allclose(replayed.projection, dykstra, rtol=0.0, atol=1e-12 * np.abs(z).max())


if __name__ == "__main__":
    unittest.main()
