"""Regression coverage for the LTI Dykstra experiment family."""

from pathlib import Path
import signal
import sys
import threading
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


def spy_exact_cycles(solver) -> list:
    """Cycles the solver runs as exact Dykstra cycles."""
    cycles = []
    run_exact_cycle = solver._dykstra_cycle
    solver._dykstra_cycle = lambda cycle: (cycles.append(cycle), run_exact_cycle(cycle))[1]
    return cycles


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

    def test_frozen_drain_far_past_the_budget_returns_promptly(self) -> None:
        # In the three-planes case the draining auxiliary would cross about 2.5e30 cycles
        # out, where refining the crossing one cycle at a time never terminates; the
        # narrow wedge reached the same loop through Ver5 until Ver5 certified it at entry
        if not hasattr(signal, "setitimer") or threading.current_thread() is not threading.main_thread():
            self.skipTest("the hang guard needs SIGALRM on the main thread")

        def unit(degrees: float) -> np.ndarray:
            return np.array([np.cos(np.radians(degrees)), np.sin(np.radians(degrees)), 0.0])

        cases = {
            "far point, three planes": (
                np.array([-1e30, 1.0, 0.0]),
                np.array([[-1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [-0.5, -1.0, 0.0]]),
                np.array([0.0, 0.0, -1.0]),
            ),
            "far point, narrow wedge": (
                1e25 * np.array([-3.0, -2.6, 0.0]),
                np.array([unit(70.0), -unit(68.0), -unit(68.0)]),
                np.array([-0.3, 0.6, -0.6]),
            ),
        }

        def on_alarm(signum, frame):
            raise TimeoutError("solve() did not return within 2 s")

        previous = signal.signal(signal.SIGALRM, on_alarm)
        previous_timer = signal.getitimer(signal.ITIMER_REAL)
        try:
            for case_name, (z, A, b) in cases.items():
                dykstra = DykstraProjectionSolver(z, A, b, max_iter=2).solve().projection
                scale = float(np.abs(dykstra).max())
                for solver_type in LTI_SOLVERS:
                    with self.subTest(case=case_name, solver=solver_type.__name__):
                        signal.setitimer(signal.ITIMER_REAL, 2.0)
                        try:
                            solver = solver_type(z, A, b, max_iter=2)
                            result = solver.solve()
                        except TimeoutError as error:
                            self.fail(str(error))
                        finally:
                            signal.setitimer(signal.ITIMER_REAL, 0.0)
                        # The QP reference is unusable at this magnitude, so a settled
                        # point is checked for feasibility, to within 1e-9 of its step
                        if result.is_settled():
                            step = np.linalg.norm(z - result.projection)
                            self.assertTrue(np.all(A @ result.projection - b <= 1e-9 * step))
                        else:
                            np.testing.assert_allclose(result.projection, dykstra, rtol=1e-9, atol=1e-9 * scale)
        finally:
            signal.signal(signal.SIGALRM, previous)
            signal.setitimer(signal.ITIMER_REAL, *previous_timer)

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

    def test_every_version_ignores_unrelated_coordinates_and_problem_size(self) -> None:
        # A coordinate no normal touches, or a uniform rescaling, leaves the projection
        # unchanged in the problem's own length scale, the last entry of each case
        cases = {
            "unrelated coordinate of 1e9": (
                np.array([-2.0, 1.4, 1e9]),
                np.array([[-1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [-0.5, -1.0, 0.0]]),
                np.array([1.0, 1.0, -1.0]),
                1.0,
            ),
            "unrelated coordinate of 1e6": (
                np.array([0.5, -1.0, 1e6]),
                np.array([[-0.5, -2.0, 0.0], [2.0, -1.0, 0.0], [2.0, -2.0, 0.0]]),
                np.array([-0.5, 0.0, -0.5]),
                1.0,
            ),
            "scaled by 1e-9": (
                1e-9 * np.array([-2.0, 1.4, 0.0]),
                np.array([[-1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [-0.5, -1.0, 0.0]]),
                1e-9 * np.array([1.0, 1.0, -1.0]),
                1e-9,
            ),
            "limit of one half-space, scaled by 1e-12": (
                np.zeros(2),
                np.array([[-1.0, -1.0], [0.0, 1.0]]),
                1e-12 * np.array([0.0, -1.0]),
                1e-12,
            ),
        }

        for case_name, (z, A, b, length) in cases.items():
            reference_solver = DykstraProjectionSolver(z, A, b, max_iter=200)
            standard_dykstra = reference_solver.solve().projection
            qp_reference = reference_solver.actual_projection
            np.testing.assert_allclose(standard_dykstra, qp_reference, rtol=1e-13, atol=1e-7 * length)

            for solver_type in LTI_SOLVERS:
                with self.subTest(case=case_name, solver=solver_type.__name__):
                    result = solver_type(z, A, b, max_iter=200).solve()
                    self.assertLessEqual(float(np.max(A @ result.projection - b)), 1e-8 * length)
                    np.testing.assert_allclose(result.projection, standard_dykstra,
                                               rtol=1e-13, atol=1e-7 * length)
                    np.testing.assert_allclose(result.projection, qp_reference,
                                               rtol=1e-13, atol=1e-7 * length)

    def test_frozen_stall_test_is_invariant_under_scaling_the_problem(self) -> None:
        # At scale 1e-9 cycle 2 of the first problem moves the state by 0.25 * scale,
        # which is not a frozen stall however small it is in absolute terms
        cases = {
            "moving wedge": (
                np.array([1.0, 0.0, 0.0]),
                np.array([[1.0, 1.0, 0.0], [1.0, 0.0, 0.0]]),
                np.zeros(2),
            ),
            "rank-deficient episodes": (
                np.array([-2.5773951085302205, -4.631028893964539, 0.7036526127117877]),
                np.array([[1.903719540005617, 2.1684157123550465, 0.7132322831037413],
                          [0.504002187870548, 0.24362088066698953, -0.2981776943387448],
                          [-0.4672237847253826, -0.22584320592790263, 0.27641886131141086],
                          [-1.9996431856170793, -1.3159613447991658, 0.4179761229367063]]),
                np.array([-2.5984110466213823, 0.23570619892985528, 0.8680844100049375,
                          2.1886709836203315]),
            ),
        }

        for case_name, (z, A, b) in cases.items():
            for scale in (1.0, 1e-9):
                for max_iter in (3, 20, 300):
                    reference_solver = DykstraProjectionSolver(scale * z, A, scale * b, max_iter=max_iter)
                    dykstra = reference_solver.solve().projection
                    with self.subTest(case=case_name, scale=scale, max_iter=max_iter):
                        solver = LTIVer4Solver(scale * z, A, scale * b, max_iter=max_iter)
                        result = solver.solve()
                        np.testing.assert_allclose(result.projection / scale,
                                                   expected_result(result, solver, dykstra) / scale,
                                                   rtol=0.0, atol=1e-10)

    def test_nearly_parallel_rank_deficient_episode_stays_on_the_dykstra_path(self) -> None:
        # Rows 1 and 2 are about 6.5e-6 rad apart, so I - A_m + P_1 has condition about
        # 7e10, where a deflated closed form with an explicit inverse once left the path
        # by 9e-8; Ver5 now certifies the projection at the episode's entry, before any
        # deflation, and Ver1-4 step the singular episode
        A = np.array([
            [1.061165117248386, -0.4797851068903314, 1.4493092864891945, -0.644978051185903],
            [0.2640002985508877, 0.2966456083995408, 0.19636328836145112, -0.6327615294046715],
            [0.26400116371678334, 0.296648512460822, 0.19635930251866132, -0.6327611382963289],
        ])
        b = np.array([1.5782483174150457, -0.49228872262211826, -0.49229626709978036])
        z = np.array([0.1, -1.0, 2.0, -0.8])

        dykstra = DykstraProjectionSolver(z, A, b, max_iter=2).solve().projection
        for solver_type in LTI_SOLVERS:
            with self.subTest(solver=solver_type.__name__, max_iter=2):
                solver = solver_type(z, A, b, max_iter=2)
                result = solver.solve()
                np.testing.assert_allclose(result.projection, expected_result(result, solver, dykstra),
                                           rtol=0.0, atol=1e-12)

            with self.subTest(solver=solver_type.__name__, max_iter=2000):
                solver = solver_type(z, A, b, max_iter=2000)
                result = solver.solve()
                np.testing.assert_allclose(result.projection, solver.actual_projection, rtol=0.0, atol=1e-12)

    def test_episode_above_the_resolvent_cutoff_stays_on_the_dykstra_path(self) -> None:
        # Two normals 1e-5 rad apart give cond(I - A_m) of about 2.3e10, above the cutoff,
        # so the episode is stepped; a closed form with an explicit inverse once left the
        # path here by 3e-7
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

    def test_correction_history_satisfies_the_dykstra_identity(self) -> None:
        # Dykstra's auxiliary update telescopes to x^t = z - sum_m e_m^t after every cycle
        z, A, b = box_line_problem()
        max_iter = 40
        atol = 1e-9 * (1.0 + np.abs(z).max())
        reference = DykstraProjectionSolver(z, A, b, max_iter=max_iter, plot_errors=True).solve()
        np.testing.assert_allclose(reference.path[1:, -1] + reference.errors_for_plotting.sum(axis=1),
                                   np.tile(z, (max_iter, 1)), atol=atol)

        for solver_type in LTI_SOLVERS:
            with self.subTest(solver=solver_type.__name__):
                result = solver_type(z, A, b, max_iter=max_iter, plot_errors=True).solve()
                corrections = result.errors_for_plotting
                np.testing.assert_allclose(result.path[1:, -1] + corrections.sum(axis=1),
                                           np.tile(z, (max_iter, 1)), atol=atol)
                np.testing.assert_allclose(result.projection + corrections[-1].sum(axis=0), z, atol=atol)

    def test_converged_correction_history_matches_standard_dykstra(self) -> None:
        # One half-space: Dykstra is exact after one cycle, so every correction is known
        z, A, b = np.array([2.0, 1.0]), np.array([[1.0, 0.0]]), np.array([0.0])
        reference = DykstraProjectionSolver(z, A, b, max_iter=3, plot_errors=True).solve()
        for solver_type in LTI_SOLVERS:
            with self.subTest(solver=solver_type.__name__):
                result = solver_type(z, A, b, max_iter=3, plot_errors=True).solve()
                np.testing.assert_allclose(result.errors_for_plotting, reference.errors_for_plotting, atol=1e-12)

    def test_every_cycle_is_recorded_without_constraints(self) -> None:
        z = np.array([1.0, 2.0])
        A, b = np.empty((0, 2)), np.empty(0)
        for min_error in (1e-3, 0.0):
            reference = DykstraProjectionSolver(z, A, b, max_iter=5, track_error=True,
                                                min_error=min_error).solve()
            for solver_type in LTI_SOLVERS:
                with self.subTest(solver=solver_type.__name__, min_error=min_error):
                    result = solver_type(z, A, b, max_iter=5, track_error=True,
                                         min_error=min_error).solve()
                    np.testing.assert_array_equal(result.projection, z)
                    np.testing.assert_array_equal(result.squared_errors, reference.squared_errors)
                    np.testing.assert_array_equal(result.converged_errors, reference.converged_errors)
                    np.testing.assert_array_equal(result.stalled_errors, reference.stalled_errors)

    def test_history_matches_standard_dykstra_until_settlement(self) -> None:
        # Each case skips cycles a different way: closed-form scanning, an envelope jump
        # (Ver3), a frozen-stall jump (Ver4) and the modal scan (Ver5); a skipped cycle
        # must still be recorded as Dykstra's own iterate, half-space by half-space
        box_z, box_A, box_b = box_line_problem()
        cases = {
            "closed form": (box_z, box_A, box_b),
            "envelope jump": (
                [1.5, -4.5],
                [[-0.5, -0.75], [0.25, -2.5], [3.0, -0.75], [-0.75, 0.75]],
                [1.5, 6.5, 3.0, 0.25],
            ),
            "frozen stall": (
                [0.5, 0.0, 6.0],
                [[-1.0, 0.0, 0.75], [-1.0, 0.0, 0.75], [1.25, -0.25, -1.75]],
                [2.125, 1.875, -0.625],
            ),
            "modal scan": (
                [1.0, 2.5, 5.5],
                [[-1.0, -0.25, 1.5], [-0.25, -0.25, 0.5], [0.5, -0.25, 1.0],
                 [0.75, 0.25, -0.25], [-1.5, 0.5, -0.5], [0.25, -0.25, 1.75]],
                [-2.0, 0.5, 0.25, 3.75, -1.5, 0.0],
            ),
        }

        max_iter = 60
        for case_name, problem in cases.items():
            z, A, b = (np.asarray(item, dtype=float) for item in problem)
            atol = 1e-10 * np.abs(z).max()
            dykstra = DykstraProjectionSolver(z, A, b, max_iter=max_iter, track_error=True,
                                              plot_errors=True).solve()
            for solver_type in LTI_SOLVERS:
                with self.subTest(case=case_name, solver=solver_type.__name__):
                    result = solver_type(z, A, b, max_iter=max_iter, track_error=True,
                                         plot_errors=True).solve()
                    end = result.settled_at if result.is_settled() else max_iter + 1
                    np.testing.assert_allclose(result.path[:end], dykstra.path[:end], rtol=0.0, atol=atol)
                    np.testing.assert_allclose(result.squared_errors[:end], dykstra.squared_errors[:end],
                                               rtol=0.0, atol=1e-9)
                    np.testing.assert_allclose(result.errors_for_plotting[:end - 1],
                                               dykstra.errors_for_plotting[:end - 1], rtol=0.0, atol=atol)

    def test_settlement_is_reported_and_recorded_from_its_cycle(self) -> None:
        # Dykstra reaches (-0.25, 0.25) in two cycles and the projection (0, 0) only in
        # the limit; Ver2-4 prove the active set final on cycle 2, and Ver5 certifies the
        # projection at that cycle's entry
        z, A, b = np.array([2.0, 1.0]), np.array([[1.0, 0.0], [1.0, 1.0]]), np.zeros(2)
        certificates = {LTIVer1Solver: None, LTIVer2Solver: "finality", LTIVer3Solver: "finality",
                        LTIVer4Solver: "finality", LTIVer5Solver: "kkt"}
        for max_iter in (2, 4):
            dykstra = DykstraProjectionSolver(z, A, b, max_iter=max_iter, track_error=True).solve()
            self.assertFalse(dykstra.is_settled())
            for solver_type in LTI_SOLVERS:
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

    def test_equality_pair_does_not_force_an_exact_cycle_every_cycle(self) -> None:
        # Rows 2 and 3 are one equality written as two half-spaces; rounding leaves the
        # idle member a slack of a few ulps above zero, which once predicted a switch
        # on every cycle although Dykstra's active set changes once in 300 cycles
        z = np.array([2.4007125994569645, -7.7558065657026365])
        A = np.array([[-0.6178466870477588, -0.4973069762069178],
                      [0.32830760276717613, 1.6568144692019835],
                      [0.2117931803349737, -1.979476762616457],
                      [-0.3626095681006677, 3.389047809860636],
                      [-1.0189481378840013, -1.3299508294844882]])
        b = np.array([-0.24332332165456616, 3.2900231051990105, -2.330544570041443,
                      3.990108456964962, -1.6600198362466738])
        dykstra = DykstraProjectionSolver(z, A, b, max_iter=300).solve().projection
        for solver_type in LTI_SOLVERS:
            with self.subTest(solver=solver_type.__name__):
                solver = solver_type(z, A, b, max_iter=300)
                exact_cycles = spy_exact_cycles(solver)
                result = solver.solve()
                self.assertLessEqual(len(exact_cycles), 5)
                np.testing.assert_allclose(result.projection, expected_result(result, solver, dykstra),
                                           rtol=0.0, atol=1e-12)

    def test_slow_drains_are_jumped_rather_than_scanned(self) -> None:
        # Both episodes drain an active auxiliary with no transient left to wait for: one
        # crosses about 5e7 cycles out, past the envelope's bracketing cap, and in the
        # other the active normals are e_1, e_2 and e_1 again, so A_m = 0 and rho = 0
        angles = np.deg2rad([80.0, 90.0, 100.0])
        cases = {
            "far crossing": (np.array([0.0, 10.0]), np.column_stack([np.cos(angles), np.sin(angles)]),
                             np.array([0.0, -1e-7, 0.0])),
            "orthogonal normals": (np.ones(2), np.array([[1.0, 0.0], [0.0, 1.0], [1.0, 0.0]]),
                                   np.array([0.0, 0.0, -1e-5])),
        }
        for case_name, (z, A, b) in cases.items():
            dykstra = DykstraProjectionSolver(z, A, b, max_iter=2000).solve().projection
            for solver_type in (LTIVer3Solver, LTIVer4Solver):
                with self.subTest(case=case_name, solver=solver_type.__name__):
                    solver = solver_type(z, A, b, max_iter=2000)
                    jumps = []
                    jump = solver._jump
                    solver._jump = lambda *args: (lambda result: (jumps.append(result[0]), result)[1])(jump(*args))
                    result = solver.solve()
                    np.testing.assert_allclose(result.projection, expected_result(result, solver, dykstra),
                                               rtol=0.0, atol=1e-12)
                    self.assertGreater(max(jumps, default=0), 1000)

    def test_frozen_stall_is_fast_forwarded_whatever_the_budget(self) -> None:
        # The first state stays frozen while an auxiliary drains towards a crossing about
        # 2e6 cycles away, past the budget; the second freezes only after a few stepped
        # cycles, because its first and last active normals are orthogonal
        r2 = np.sqrt(2.0)
        cases = {
            "crossing past the budget": (np.array([1e6, 0.0]),
                                         np.array([[1.0, 0.0], [-1.0, 0.0], [1.0, 0.0]]),
                                         np.array([1.0, 0.0, 0.5])),
            "frozen part-way through": (np.array([10.0, 3.0, 0.5]),
                                      np.array([[1.0, 0.0, 0.0], [1 / r2, 1 / r2, 0.0], [0.0, 1.0, 0.0]]),
                                      np.array([1.0, r2 - 1e-4, 1.0])),
        }
        for case_name, (z, A, b) in cases.items():
            reference = LTIVer1Solver(z, A, b, max_iter=20000).solve().projection
            with self.subTest(case=case_name):
                solver = LTIVer4Solver(z, A, b, max_iter=20000)
                steps = []
                advance = solver._advance_cycle
                solver._advance_cycle = lambda: (steps.append(1), advance())
                result = solver.solve()
                np.testing.assert_allclose(result.projection, expected_result(result, solver, reference),
                                           rtol=0.0, atol=1e-12 * np.abs(z).max())
                self.assertLess(len(steps), 100)

    def test_ill_conditioned_episode_returns_the_iterate_or_the_projection(self) -> None:
        # A wedge 1.5e-6 rad wide gives cond(I - A_m) of about 4e11, where the fixed point
        # of the rounded cycle map misses the apex by about 3e-6
        theta = 1.5e-6
        A = np.array([[0.0, 1.0], [np.sin(theta), -np.cos(theta)]])
        apex = np.array([0.3, 0.7])
        z, b = apex + np.array([1.0, 0.5]), A @ apex
        for max_iter in (5, 200):
            reference_solver = DykstraProjectionSolver(z, A, b, max_iter=max_iter)
            dykstra = reference_solver.solve().projection
            for solver_type in LTI_SOLVERS:
                with self.subTest(solver=solver_type.__name__, max_iter=max_iter):
                    projection = solver_type(z, A, b, max_iter=max_iter).solve().projection
                    self.assertLess(min(np.abs(projection - dykstra).max(),
                                        np.abs(projection - reference_solver.actual_projection).max()), 1e-8)

    def test_ver5_certifies_only_the_projection(self) -> None:
        # Each case once returned settled=True away from the projection: a slab whose
        # walls are 9e-8 rad from parallel and whose limit violates a row by 1.7e-9, rows
        # 5e-7 inside the loose tight band carrying multipliers, and a wedge whose limit
        # the resolvent loses
        theta = 1.5e-6
        wedge_A = np.array([[0.0, 1.0], [np.sin(theta), -np.cos(theta)]])
        wedge_apex = np.array([0.3, 0.7])
        cases = {
            "thin slab": (
                np.array([-2.214017695513702, 1.420323277826624]),
                np.array([[-1.17684612741209, 1.2390550702799208],
                          [0.9855651633523589, 0.16929651144143015],
                          [-0.9855651788783955, -0.1692964210560769]]),
                np.array([7.201498328866286, -0.5652622410478404, 0.56526240210934]),
            ),
            "loose tight band": (
                np.array([-0.20027373229037335, -3.7900904061063447, 4.619121844997523]),
                np.array([[0.6523873274568803, -0.75788572685707, 0.0],
                          [-0.7136344148070796, 0.7005183238166988, 0.0],
                          [-0.7569148072721577, 0.6535135610927691, 0.0]]),
                np.array([-0.0001124406713453392, 0.015503330040274278, 0.02715164951644004]),
            ),
            "loose tight band, second": (
                np.array([7.266888334528822, -5.520340862055513, 2.5878978909372736]),
                np.array([[0.09710188189914234, 0.9952744468394861, 0.0],
                          [0.3123082375331488, -0.9499808233690501, 0.0],
                          [0.41005948939534226, -0.9120587783453604, 0.0]]),
                np.array([-2.1844715877611156, 1.8529686415093019, 1.7158369137491483]),
            ),
            "ill-conditioned wedge": (wedge_apex + np.array([1.0, 0.5]), wedge_A, wedge_A @ wedge_apex),
        }

        for case_name, (z, A, b) in cases.items():
            for max_iter in (200, 2000):
                with self.subTest(case=case_name, max_iter=max_iter):
                    solver = LTIVer5Solver(z, A, b, max_iter=max_iter)
                    result = solver.solve()
                    if solver.settled:
                        np.testing.assert_allclose(result.projection, solver.actual_projection,
                                                   rtol=0.0, atol=1e-8 * np.abs(z).max())

    def test_ver5_certificate_exits_crawls_and_degenerate_limits(self) -> None:
        # Near-parallel wedges crawl for about 1 / theta**2 cycles, an equality written as
        # two half-spaces leaves an inactive row exactly tight, and a feasible z is its own
        # projection; the KKT test settles each without running the crawl
        cases = {
            "equality pair": (np.array([2.0, 1.0]),
                              np.array([[1.0, 1.0], [-1.0, -1.0], [1.0, 0.0]]),
                              np.array([1.0, -1.0, 0.2])),
            "feasible start": (np.array([0.1, 0.1]), np.eye(2), np.ones(2)),
        }
        for theta in (1e-3, 1e-5, 1e-6):
            cases[f"wedge of half-angle {theta:g}"] = (
                np.array([-1.0, 0.3, 0.7, 0.5]),
                np.array([[-np.sin(theta), np.cos(theta), 0.0, 0.0],
                          [-np.sin(theta), -np.cos(theta), 0.0, 0.0]]),
                np.zeros(2),
            )

        for case_name, (z, A, b) in cases.items():
            with self.subTest(case=case_name):
                solver = LTIVer5Solver(z, A, b, max_iter=200, plot_errors=True)
                result = solver.solve()
                self.assertTrue(solver.settled)
                np.testing.assert_allclose(result.projection, solver.actual_projection,
                                           rtol=0.0, atol=1e-9)
                # The multipliers written back as corrections reproduce z
                np.testing.assert_allclose(result.projection + result.errors_for_plotting[-1].sum(axis=0),
                                           z, rtol=0.0, atol=1e-9)

    def test_stall_jump_does_not_skip_a_small_reactivation(self) -> None:
        # Rows 1 and 3 hold the state at x = 0.5 * scale while row 1 drains; row 0 is
        # violated there by only 5e-13, and Dykstra reactivates it on cycle 2
        scale = 1e-6
        z = scale * np.array([5.0, 0.0])
        A = np.array([[-1.0, 1.0], [1.0, 0.0], [-1.0, 0.0], [1.0, 0.0]])
        b = np.array([-0.5 * scale - 5e-13, scale, 0.0, 0.5 * scale])
        dykstra = DykstraProjectionSolver(z, A, b, max_iter=9).solve().projection
        result = LTIVer4Solver(z, A, b, max_iter=9, plot_active_halfspaces=True).solve()
        self.assertFalse(result.is_settled())
        np.testing.assert_array_equal(result.active_half_spaces[:, 2], [1, 1, 0, 1])
        np.testing.assert_allclose(result.projection / scale, dykstra / scale, rtol=0.0, atol=1e-12)

    def test_modal_scan_keeps_a_draining_row_watched_at_any_scale(self) -> None:
        # No normal touches the third coordinate, so the episode with all three rows
        # active is deflated and, with the certificate set aside, scanned modally; row 2
        # drains by 0.29 * scale per cycle and Dykstra drops it on cycle 7. A drain
        # tolerance fixed at 1e-9 cleared it at small scales and jumped the budget with
        # it still active
        z = np.array([1.0, -1.5, 0.0])
        A = np.array([[1.5, -1.0, 0.0], [-0.5, 1.0, 0.0], [0.5, 0.5, 0.0]])
        b = np.array([-0.5, -2.0, -2.0])
        for scale in (1.0, 1e-9, 1e-12):
            with self.subTest(scale=scale):
                dykstra = DykstraProjectionSolver(scale * z, A, scale * b, max_iter=20).solve()
                solver = LTIVer5Solver(scale * z, A, scale * b, max_iter=20, block_size=4)
                solver._kkt_certificate = lambda hint, active: None
                result = solver.solve()
                self.assertFalse(result.is_settled())
                np.testing.assert_allclose(result.path / scale, dykstra.path / scale, rtol=0.0, atol=1e-12)
                np.testing.assert_allclose(result.projection / scale, dykstra.projection / scale,
                                           rtol=0.0, atol=1e-12)

    def test_deflated_episode_switching_on_its_first_cycle_keeps_the_dykstra_state(self) -> None:
        # Rows 1-3 span three of the four coordinates, so their episode is deflated,
        # and their cycle map is nearly defective (eigenvector condition about 6e7).
        # Row 0 reactivates on the episode's first cycle; rebuilding the state from
        # the modes there would move it off Dykstra's path by about cond * eps
        z = np.array([4.0, -6.0, -1.0, 1.0])
        A = np.array([
            [-1.0, 0.4, 0.1, 0.0],
            [0.18881711923692265, -0.19839032737660414, 0.9617636786063786, 0.0],
            [0.16021416297716448, -0.818128926665578, 0.5522648652001644, 0.0],
            [0.6640385974516226, 0.5503881193550134, -0.5060885882603295, 0.0],
            [0.5, -0.8, 0.0, 0.2],
        ])
        b = np.array([-4.4, 0.0, 0.0, 0.0, 8.0])
        max_iter = 2

        dykstra = DykstraProjectionSolver(z, A, b, max_iter=max_iter, track_error=True,
                                          plot_errors=True).solve()
        result = LTIVer5Solver(z, A, b, max_iter=max_iter, track_error=True, plot_errors=True).solve()
        end = max_iter + 1 if result.settled_at is None else result.settled_at
        self.assertGreater(end, 2)
        atol = 1e-10 * np.abs(z).max()
        np.testing.assert_allclose(result.path[:end], dykstra.path[:end], rtol=0.0, atol=atol)
        np.testing.assert_allclose(result.errors_for_plotting[:end - 1],
                                   dykstra.errors_for_plotting[:end - 1], rtol=0.0, atol=atol)

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

    def test_deflated_episode_stays_on_the_dykstra_path_when_the_resolvent_is_ill_conditioned(self) -> None:
        # Rows 0 and 1 are 4.4e-4 rad apart and the last coordinate is free, so the
        # episode of rows 0, 1 and 3 is deflated with cond(I - A_m + P_1) about 1.2e7;
        # it cannot be certified, scans one cycle and switches on cycle 4, and an
        # explicit inverse moves the state it switches from by about 1e-9
        z = np.array([-6.0, 0.0, 37.0, -33.0])
        A = np.array([
            [2.269, 1.999, 0.634, 0.0],
            [2.267, 1.999, 0.634, 0.0],
            [-0.997, 1.345, -0.725, 0.0],
            [-0.222, -1.105, 0.245, 0.0],
        ])
        b = np.array([23.01, 23.0, -12.42, 2.96])
        dykstra = DykstraProjectionSolver(z, A, b, max_iter=5).solve()

        result = LTIVer5Solver(z, A, b, max_iter=5).solve()
        end = result.settled_at if result.is_settled() else 6
        self.assertGreaterEqual(end, 5)
        np.testing.assert_allclose(result.path[:end], dykstra.path[:end], rtol=0.0,
                                   atol=1e-12 * np.abs(z).max())

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

    def test_ver5_multipliers_only_on_rows_tight_to_rounding(self) -> None:
        # Row 0 is 1.9e-7 inside the vertex of rows 1-3 and nearly dependent on rows 1
        # and 2; a multiplier on it makes the vertex pass the KKT test with multipliers
        # near 1e4, although the projection lies 5e-3 away
        z = np.array([-2.767835, -0.105477, 2.268443])
        A = np.array([[-0.018279, 0.186721, -0.163899], [-2.25957, 0.403857, 0.5084],
                      [0.56094, -0.98625, 0.685253], [-0.552181, 0.012463, 0.13444]])
        b = np.array([-0.083882, -1.237567, 0.664005, -0.294348])
        for max_iter in (2, 200):
            reference_solver = DykstraProjectionSolver(z, A, b, max_iter=max_iter)
            dykstra = reference_solver.solve().projection
            with self.subTest(max_iter=max_iter):
                solver = LTIVer5Solver(z, A, b, max_iter=max_iter)
                result = solver.solve()
                expected = reference_solver.actual_projection if solver.settled else dykstra
                np.testing.assert_allclose(result.projection, expected, rtol=0.0, atol=1e-9 * np.abs(z).max())

    def test_ver5_certifies_a_crawl_whose_limit_dormant_rows_cut_off(self) -> None:
        # Rows 0 and 1 form a wedge 0.02 rad wide that Dykstra crawls along towards the
        # origin; rows 2 and 3 both cut the origin off but stay inactive for thousands of
        # cycles, and only row 2 binds at the projection, so neither the active set nor
        # every candidate row is its support
        theta = 1e-2
        z = np.array([-1.0, 0.3, 0.0])
        A = np.array([[-np.sin(theta), np.cos(theta), 0.0],
                      [-np.sin(theta), -np.cos(theta), 0.0],
                      [1.0, 0.0, 1.0],
                      [1.0, 0.0, 2.0]])
        b = np.array([0.0, 0.0, -0.3, -0.1])
        dykstra = DykstraProjectionSolver(z, A, b, max_iter=200).solve().projection
        solver = LTIVer5Solver(z, A, b, max_iter=200, plot_errors=True)
        result = solver.solve()
        self.assertGreater(np.abs(dykstra - solver.actual_projection).max(), 0.5)
        self.assertTrue(solver.settled)
        self.assertEqual(result.certificate, "kkt")
        np.testing.assert_allclose(result.projection, solver.actual_projection, rtol=0.0, atol=1e-9)
        np.testing.assert_allclose(result.projection + result.errors_for_plotting[-1].sum(axis=0),
                                   z, rtol=0.0, atol=1e-9)

    def test_ver5_does_not_certify_a_limit_the_kkt_test_rejects(self) -> None:
        # Row 0 drains by about 1.4e-11 per cycle, below the rounding level of the drifts
        # at an offset of 1000, so the finality test passes at the corner (1000, 1000);
        # the projection lies 1e-3 further along row 1, where row 2 meets it
        offset = np.array([1000.0, 1000.0])
        A = np.array([[-1.0, 0.0], [0.0, 1.0], [-1.2e-4, -1.0]])
        b = A @ offset - np.array([0.0, 0.0, 1.2e-7])
        z = offset + np.array([-1.0, 1.0])
        atol = 1e-9 * np.abs(z).max()
        for max_iter in (3, 200):
            reference_solver = DykstraProjectionSolver(z, A, b, max_iter=max_iter)
            dykstra = reference_solver.solve().projection
            with self.subTest(max_iter=max_iter):
                result = LTIVer5Solver(z, A, b, max_iter=max_iter).solve()
                if result.certificate == "kkt":
                    np.testing.assert_allclose(result.projection, reference_solver.actual_projection,
                                               rtol=0.0, atol=atol)
                else:
                    self.assertIsNone(result.settled_at)
                    np.testing.assert_allclose(result.projection, dykstra, rtol=0.0, atol=atol)

    def test_active_idle_member_of_an_equality_does_not_force_exact_cycles(self) -> None:
        # Dykstra leaves the idle member of the equality active with an auxiliary of a
        # few ulps, whose one-cycle prediction rounding can push to or below zero; only a
        # value beyond rounding may deactivate it
        z, A, b = np.array([3.7, 0.6]), np.array([[1.4, 0.9], [-1.4, -0.9]]), np.array([0.6, -0.6])
        dykstra = DykstraProjectionSolver(z, A, b, max_iter=300).solve().projection
        for solver_type in (LTIVer1Solver, LTIVer2Solver, LTIVer3Solver):
            with self.subTest(solver=solver_type.__name__):
                solver = solver_type(z, A, b, max_iter=300)
                exact_cycles = spy_exact_cycles(solver)
                result = solver.solve()
                self.assertEqual(len(exact_cycles), 1)
                np.testing.assert_allclose(result.projection, expected_result(result, solver, dykstra),
                                           rtol=0.0, atol=1e-12)

    def test_idle_member_of_an_equality_does_not_end_a_closed_form_episode(self) -> None:
        # Rows 1 and 2 are one equality and, with row 0, span the plane, so the episode
        # runs in closed form; the idle member's slack of a few ulps once predicted a
        # switch on every cycle, so the active set was never proven final
        z = np.array([2.7, 8.1])
        A = np.array([[1.0, 1.2], [-2.0, -2.9], [2.0, 2.9]])
        b = np.array([-1.7, 1.9, -1.9])
        for solver_type in (LTIVer2Solver, LTIVer3Solver, LTIVer4Solver):
            with self.subTest(solver=solver_type.__name__):
                solver = solver_type(z, A, b, max_iter=100, track_error=True)
                exact_cycles = spy_exact_cycles(solver)
                result = solver.solve()
                self.assertLessEqual(len(exact_cycles), 5)
                self.assertEqual(result.certificate, "finality")
                np.testing.assert_allclose(result.projection, solver.actual_projection,
                                           rtol=0.0, atol=1e-9)

    def test_idle_member_of_an_equality_does_not_end_a_modal_scan(self) -> None:
        # Rows 2 and 3 are one equality; the active normals lie in the first two
        # coordinates, so Ver5 scans the episode in its deflated modal form, where the
        # idle member's slack of a few ulps once predicted a switch on every cycle
        z = np.array([-1.3, -2.0, 0.0])
        A = np.array([[-0.9, -0.5, 0.0], [0.2, -1.9, 0.0], [-0.4, 1.7, 0.0], [0.4, -1.7, 0.0]])
        b = np.array([0.5, 2.1, -2.1, 2.1])
        dykstra = DykstraProjectionSolver(z, A, b, max_iter=20).solve().projection
        solver = LTIVer5Solver(z, A, b, max_iter=20)
        exact_cycles = spy_exact_cycles(solver)
        result = solver.solve()
        self.assertLessEqual(len(exact_cycles), 5)
        np.testing.assert_allclose(result.projection, expected_result(result, solver, dykstra),
                                   rtol=0.0, atol=1e-12)

    def test_exactly_tight_idle_member_of_an_equality_pair_allows_finality(self) -> None:
        # Rows 0 and 1 are the equality x = 0, whose idle member ends exactly tight; it
        # reactivates only on a slack above rounding, so the active set is final on cycle 2
        z, A, b = np.array([2.0, 1.0]), np.array([[1.0, 0.0], [-1.0, 0.0], [0.0, 1.0]]), np.zeros(3)
        max_iter = 50
        dykstra = DykstraProjectionSolver(z, A, b, max_iter=max_iter, track_error=True).solve()
        for solver_type in (LTIVer2Solver, LTIVer3Solver, LTIVer4Solver):
            with self.subTest(solver=solver_type.__name__):
                result = solver_type(z, A, b, max_iter=max_iter, track_error=True).solve()
                self.assertEqual(result.certificate, "finality")
                self.assertEqual(result.settled_at, 2)
                np.testing.assert_allclose(result.projection, [0.0, 0.0], rtol=0.0, atol=1e-15)
                np.testing.assert_allclose(result.path[:2], dykstra.path[:2], rtol=0.0, atol=1e-15)
                np.testing.assert_allclose(result.path[2:, -1], 0.0, rtol=0.0, atol=1e-15)


if __name__ == "__main__":
    unittest.main()
