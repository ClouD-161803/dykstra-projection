"""LTI Ver4: when the active normals do not span the space and a cycle leaves
the state unchanged, the auxiliary variables drift along straight lines, so the
episode jumps directly to the cycle before the first one reaches zero."""

import numpy as np
from lti_ver3 import LTIVer3Solver

# A cycle that moves the state by less than this leaves it frozen
_MOTION_TOL = 1e-9

# Auxiliary increments delta_m smaller than this count as zero
_DELTA_TOL = 1e-12

# Motion of the state times jump length allowed while treating the state as frozen
_MOTION_BUDGET = 1e-9


class LTIVer4Solver(LTIVer3Solver):
    """Frozen-stall fast-forward."""

    def _is_stalled(self, x: np.ndarray | None = None) -> bool:
        """Cycle leaves the state unchanged."""
        x = self.x if x is None else x
        x_next = self.A_m @ x + self.B_m
        return float(np.linalg.norm(x_next - x)) < _MOTION_TOL

    def _stall_crossing(self, y: np.ndarray, delta: np.ndarray, active: np.ndarray) -> float:
        """Cycles until the first crossing."""
        crossing = np.inf

        # A draining active auxiliary reaches zero after ceil(y_m / -delta_m) cycles
        for m in np.where(active)[0]:
            if delta[m] < -_DELTA_TOL and y[m] > 0.0:
                k = int(np.floor(y[m] / (-delta[m])))
                while y[m] + k * delta[m] > 0.0:
                    k += 1
                while k > 1 and y[m] + (k - 1) * delta[m] <= 0.0:
                    k -= 1
                crossing = min(crossing, max(k, 1))

        # A positive increment on an inactive half-space reactivates it at once
        for j in np.where(~active)[0]:
            if delta[j] > _DELTA_TOL:
                crossing = min(crossing, 1 if y[j] + delta[j] > 0.0 else 2)
        return crossing

    def _stall_episode(self, start_cycle: int, x: np.ndarray | None = None,
                       y: np.ndarray | None = None,
                       active: np.ndarray | None = None) -> int | None:
        """Fast-forward a frozen stall."""
        x = self.x if x is None else x
        y = self.y if y is None else y
        active = np.asarray(self.active if active is None else active)

        # Inactive auxiliaries are zero; every auxiliary moves by delta per cycle
        y = np.where(active, y, 0.0)
        motion = float(np.linalg.norm(self.A_m @ x + self.B_m - x))
        delta = self.R @ x + self.s

        # No jump when the crossing lies beyond the budget or the motion of the
        # state over the jump would exceed the budget
        crossing = self._stall_crossing(y, delta, active)
        if crossing > self.max_iter - start_cycle + 1:
            return None
        k = int(crossing) - 1
        if motion * max(k, 1) > _MOTION_BUDGET:
            return None

        # Apply the k increments at once, the state staying frozen, then switch
        if k >= 1:
            self._set_state(x, y + k * delta, active)
            for cycle in range(start_cycle, start_cycle + k):
                self._record_cycle(cycle, active)
        return start_cycle + k

    def _accelerate(self, start_cycle: int) -> None:
        """Run episodes to the budget."""
        p = len(self.x)
        cycle = start_cycle
        while cycle <= self.max_iter:
            # Closed form when I - A_m is invertible, stall fast-forward when it is
            # singular and the state is frozen, exact stepping otherwise
            IA = np.eye(p) - self.A_m
            if np.linalg.cond(IA) < 1e12:
                switch_cycle = self._closed_form_episode(cycle, np.linalg.inv(IA))
            else:
                switch_cycle = self._stall_episode(cycle) if self._is_stalled() else None
                if switch_cycle is None:
                    switch_cycle = self._step_episode(cycle)
            if switch_cycle is None:
                return
            self._perform_switch(switch_cycle)
            cycle = switch_cycle + 1
