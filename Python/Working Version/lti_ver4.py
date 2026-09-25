"""LTI Ver4: when the active normals do not span the space and a cycle leaves
the state unchanged, the auxiliary variables drift along straight lines, so the
episode jumps directly to the cycle before the first one reaches zero."""

import numpy as np
from lti_ver2 import _RESOLVENT_COND_CAP
from lti_ver3 import LTIVer3Solver


class LTIVer4Solver(LTIVer3Solver):
    """Frozen-stall fast-forward."""

    def _is_stalled(self, x: np.ndarray | None = None) -> bool:
        """Cycle leaves the state unchanged."""
        # Frozen means fixed to rounding, coordinate by coordinate: holding a state
        # that still moves, however slowly, would leave Dykstra's path
        x = self.x if x is None else x
        motion = np.abs(self.A_m @ x + self.B_m - x)
        floor = self._rounding_floor(np.abs(self.A_m) @ np.abs(x) + np.abs(self.B_m) + np.abs(x))
        return bool(np.all(motion <= floor))

    def _stall_crossing(self, y: np.ndarray, delta: np.ndarray, delta_floor: np.ndarray,
                        active: np.ndarray) -> float:
        """Cycles until the first crossing."""
        crossing = np.inf

        # A draining active auxiliary reaches zero after ceil(y_m / -delta_m) cycles
        for m in np.where(active)[0]:
            if delta[m] < -delta_floor[m] and y[m] > 0.0:
                # A crossing past the budget is never taken, and above 2**53 refining it
                # one cycle at a time takes about ratio / 2**53 steps, which never
                # finishes at ratios like 1e30
                ratio = y[m] / (-delta[m])
                if not ratio <= self.max_iter + 1:
                    crossing = min(crossing, ratio)
                    continue
                k = int(np.floor(ratio))
                while y[m] + k * delta[m] > 0.0:
                    k += 1
                while k > 1 and y[m] + (k - 1) * delta[m] <= 0.0:
                    k -= 1
                crossing = min(crossing, max(k, 1))

        # A positive increment on an inactive half-space reactivates it at once
        for j in np.where(~active)[0]:
            if delta[j] > delta_floor[j]:
                crossing = min(crossing, 1 if y[j] + delta[j] > 0.0 else 2)
        return crossing

    def _stall_episode(self, start_cycle: int, x: np.ndarray | None = None,
                       y: np.ndarray | None = None,
                       active: np.ndarray | None = None) -> int | None:
        """Fast-forward a frozen stall."""
        x = self.x if x is None else x
        y = self.y if y is None else y
        active = np.asarray(self.active if active is None else active)

        # Inactive auxiliaries are zero; every auxiliary moves by delta per cycle,
        # which is zero only up to rounding
        y = np.where(active, y, 0.0)
        delta = self.R @ x + self.s
        delta_floor = self._rounding_floor(np.abs(self.R) @ np.abs(x) + self.s_scale)

        # No jump when the crossing lies beyond the budget
        crossing = self._stall_crossing(y, delta, delta_floor, active)
        if crossing > self.max_iter - start_cycle + 1:
            return None
        k = int(crossing) - 1

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
            if np.linalg.cond(IA) < _RESOLVENT_COND_CAP:
                switch_cycle = self._closed_form_episode(cycle, IA)
            else:
                switch_cycle = self._stall_episode(cycle) if self._is_stalled() else None
                if switch_cycle is None:
                    switch_cycle = self._step_episode(cycle)
            if switch_cycle is None:
                return
            self._perform_switch(switch_cycle)
            cycle = switch_cycle + 1
