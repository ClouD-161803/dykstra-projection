"""Regression tests for solver behavior that has failed in real examples."""

import contextlib
import io
from pathlib import Path
import re
import sys
import unittest

import numpy as np


WORKING_VERSION = Path(__file__).resolve().parents[1] / "Python" / "Working Version"
sys.path.insert(0, str(WORKING_VERSION))

from convex_projection_solver import (
    DykstraMapHybridSolver,
    DykstraProjectionSolver,
    DykstraStallDetectionSolver,
)


FAST_FORWARD_LOG = re.compile(
    r"Fast forwarding (-?\d+) rounds to exit stalling at iteration (\d+)"
)

BOX_A = np.array([[1.0, 0.0], [-1.0, 0.0], [0.0, 1.0], [0.0, -1.0]])
BOX_B = np.ones(4)
BOX_LINE_A = np.vstack([BOX_A, [[0.5, 1.0], [-0.5, -1.0]]])
BOX_LINE_B = np.hstack([BOX_B, [1.0, -1.0]])


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

    def _fast_forward(self, z, A, b, max_iter):
        """Run the stall detection solver, returning its result and logged jumps."""
        log = io.StringIO()
        with contextlib.redirect_stdout(log):
            result = DykstraStallDetectionSolver(
                z, A, b, max_iter, track_error=True
            ).solve()
        jumps = [(int(iteration), int(skip))
                 for skip, iteration in FAST_FORWARD_LOG.findall(log.getvalue())]
        return result, jumps

    def _assert_matches_plain_dykstra(self, fast, jumps, z, A, b, max_iter):
        """Every fast-forwarded cycle must be a cycle plain Dykstra also reaches."""
        plain = DykstraProjectionSolver(
            z, A, b, max_iter + sum(skip for _, skip in jumps), track_error=True
        ).solve()
        for cycle in range(max_iter + 1):
            # A jump logged at iteration i is applied before cycle i + 1 ends.
            plain_cycle = cycle + sum(skip for i, skip in jumps if i < cycle)
            self.assertAlmostEqual(
                fast.squared_errors[cycle], plain.squared_errors[plain_cycle],
                places=9, msg=f"cycle {cycle} against plain cycle {plain_cycle}"
            )
            np.testing.assert_allclose(
                fast.path[cycle, -1], plain.path[plain_cycle, -1], atol=1e-9,
                err_msg=f"cycle {cycle} against plain cycle {plain_cycle}"
            )

    def test_fast_forward_skips_the_box_line_stall_without_rewinding(self) -> None:
        """The paper's example: from (-4, 1.4) plain Dykstra sits at squared
        error 0.8 for cycles 1-16. Half-spaces inactive in the stall once drove
        the skip count negative, rewinding the auxiliaries until the error grew
        to about 1e9."""
        A = np.vstack([BOX_A, [[0.5, 1.0], [-0.5, -1.0]]])
        b = np.hstack([BOX_B, [1.0, -1.0]])

        for z in ([-4.0, 1.4], [-2.0, 1.4]):
            with self.subTest(z=z):
                z = np.array(z)
                fast, jumps = self._fast_forward(z, A, b, max_iter=40)

                for iteration, skip in jumps:
                    self.assertGreaterEqual(
                        skip, 0, f"rewound {-skip} cycles at iteration {iteration}"
                    )
                self.assertGreater(sum(skip for _, skip in jumps), 0)
                self._assert_matches_plain_dykstra(fast, jumps, z, A, b, max_iter=40)

    def test_fast_forward_waits_for_a_fully_frozen_cycle(self) -> None:
        """The paper's box and line in another row order: from (3.2, 4.5) plain
        Dykstra holds squared error 0.05 for cycles 1-9. Flagging a stall when
        a single active half-space repeated its output jumped ten cycles out
        of the partly frozen cycle 2, even with the skip count fixed, and the
        error rose to 149."""
        A = np.array([
            [1.0, 0.0], [-0.5, -1.0], [0.0, 1.0], [0.5, 1.0], [-1.0, 0.0], [0.0, -1.0],
        ])
        b = np.array([1.0, -1.0, 1.0, 1.0, 1.0, 1.0])
        z = np.array([3.2, 4.5])

        fast, jumps = self._fast_forward(z, A, b, max_iter=40)

        for iteration, skip in jumps:
            self.assertGreaterEqual(
                skip, 0, f"rewound {-skip} cycles at iteration {iteration}"
            )
        self.assertGreater(sum(skip for _, skip in jumps), 0)
        self._assert_matches_plain_dykstra(fast, jumps, z, A, b, max_iter=40)

    def test_fast_forward_does_not_jump_from_a_converged_fixed_point(self) -> None:
        """Plain Dykstra reaches the projection at cycle 3 and every later cycle
        repeats. Its active slack is then rounding noise, and dividing by it
        asked for a jump of about 1.7e16 cycles on every cycle."""
        A = np.vstack([BOX_A, [[-0.7, -1.0]]])  # 0.7 x + y >= 0
        b = np.hstack([BOX_B, [0.0]])
        z = np.array([-2.0, -2.0])

        fast, jumps = self._fast_forward(z, A, b, max_iter=40)

        self.assertEqual(jumps, [])
        self._assert_matches_plain_dykstra(fast, jumps, z, A, b, max_iter=40)

    def test_squared_errors_keep_their_value_at_small_scales(self) -> None:
        """The box and line from (-2, 1.4), scaled by 1e-6, starts at squared error
        4.16e-12. Rounding each squared error to ten decimals stored it as zero, and every
        cycle was classified as converged however small min_error was."""
        scale = 1e-6
        result = DykstraProjectionSolver(
            scale * np.array([-2.0, 1.4]), BOX_LINE_A, scale * BOX_LINE_B, 10,
            track_error=True, min_error=1e-3 * scale ** 2,
        ).solve()

        self.assertAlmostEqual(result.squared_errors[0] / scale ** 2, 4.16, places=9)
        self.assertAlmostEqual(result.squared_errors[7] / scale ** 2, 0.512, places=9)
        self.assertFalse((~np.isnan(result.converged_errors)).any())

    def test_stall_classification_does_not_depend_on_the_scale(self) -> None:
        """From (-4, 1.4) plain Dykstra holds squared error 0.8 for cycles 1 to 16, so
        cycles 2 to 16 are stalled. Scaled by 1e6 the squared errors carry rounding noise
        of about 1e-4, which an absolute rounding to ten decimals kept, and only cycles 2
        to 4 and 13 to 16 were classified as stalled."""
        for scale in (1.0, 1e-6, 1e6):
            with self.subTest(scale=scale):
                result = DykstraProjectionSolver(
                    scale * np.array([-4.0, 1.4]), BOX_LINE_A, scale * BOX_LINE_B, 30,
                    track_error=True, min_error=1e-3 * scale ** 2,
                ).solve()
                self.assertEqual(np.flatnonzero(~np.isnan(result.stalled_errors)).tolist(),
                                 list(range(2, 17)))

    def test_hybrid_switches_to_dykstra_corrections_on_a_boundary(self) -> None:
        """On the wedge y <= 0, y <= x tan(20 degrees), the hybrid's first cycle from
        (-1, 1) uses MAP corrections and lands on a boundary point that is not the
        projection. Rounding that point to ten decimals before testing membership put it
        outside, so the hybrid kept its zero MAP corrections and held squared error 0.117
        at every later cycle."""
        A = np.array([[0.0, 1.0], [-np.tan(np.radians(20.0)), 1.0]])
        b = np.zeros(2)
        z = np.array([-1.0, 1.0])
        hybrid = DykstraMapHybridSolver(z, A, b, 20, track_error=True).solve()
        plain = DykstraProjectionSolver(z, A, b, 20, track_error=True).solve()

        self.assertLess(hybrid.squared_errors[-1], 1e-20)
        np.testing.assert_allclose(hybrid.projection, plain.projection, rtol=0.0, atol=1e-12)


if __name__ == "__main__":
    unittest.main()
