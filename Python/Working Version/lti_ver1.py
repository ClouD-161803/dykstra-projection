"""LTI Ver1: runs Dykstra's algorithm through its cycle map while the set of
active half-spaces is constant, at one matrix-vector product per cycle, and
executes an exact Dykstra cycle whenever the active set changes."""

import numpy as np
from convex_projection_solver import ConvexProjectionSolver
from projection_result import ProjectionResult

# Rounding allowed, in units of eps times the magnitudes that cancel, before a
# quantity that is zero in exact arithmetic counts as nonzero
_ROUNDING_UNITS = 64


class LTIVer1Solver(ConvexProjectionSolver):
    """Dykstra via its cycle map."""

    def __init__(self, *args, **kwargs) -> None:
        """Solver without a settlement yet."""
        super().__init__(*args, **kwargs)
        self.settled_at = None
        self.certificate = None

    def _update_error(self, m: int, x_temp: np.ndarray, x: np.ndarray, index: int) -> None:
        """Dykstra's auxiliary update."""
        self.e[m] = self.e[index] + (x_temp - x)

    @staticmethod
    def _rounding_floor(magnitude: np.ndarray) -> np.ndarray:
        """Rounding level of a sum."""
        return _ROUNDING_UNITS * np.finfo(float).eps * magnitude

    def _prepare(self) -> None:
        """Normalise the half-spaces."""
        self.unit_A = np.empty_like(self.A)
        self.unit_b = np.empty_like(self.b)
        for index, (row, offset) in enumerate(zip(self.A, self.b)):
            self.unit_A[index], self.unit_b[index] = self._normalise(row, offset)
        self.y = np.zeros(self.n)
        self._recorded_y = np.zeros(self.n)

    def _sync_y_from_e(self) -> None:
        """Scalar auxiliaries from vectors."""
        for m in range(self.n):
            self.y[m] = float(np.dot(self.e[m], self.unit_A[m]))

    def _sync_e_from_y(self) -> None:
        """Vector auxiliaries from scalars."""
        for m in range(self.n):
            self.e[m] = self.y[m] * self.unit_A[m]

    def _dykstra_cycle(self, cycle: int) -> tuple:
        """Run one exact Dykstra cycle."""
        for m, (normal, offset) in enumerate(zip(self.A, self.b)):
            index = (m - self.n) % self.n
            x_temp = self.x.copy()

            # Shift by the stored auxiliary e_{m-n}, then project onto H_m
            shifted = x_temp + self.e[index]
            self.x = self._project_onto_half_space(shifted, normal, offset)

            # Update e_m, zero if the shifted point was already inside H_m
            if self._is_in_half_space(shifted, self.unit_A[m], self.unit_b[m]):
                self.e[m] = np.zeros_like(self.x)
            else:
                self._update_error(m, x_temp, self.x, index)

            # Store historical data for path plotting
            self.x_historical[cycle][m] = self.x.copy()
            if self.plot_errors:
                self.errors_for_plotting[cycle - 1][m] = self.e[m].copy()

        # The active set collects the half-spaces with y_m > 0
        self._sync_y_from_e()
        self._recorded_y = self.y.copy()
        return tuple(self.y > 0.0)

    def _build_cycle_map(self, active: tuple) -> None:
        """Build the cycle map."""
        p = len(self.x)

        # Prefix map carrying the cycle start x_0 to the point entering half-space m
        P = np.eye(p)
        q = np.zeros(p)
        R = np.zeros((self.n, p))
        s = np.zeros(self.n)
        s_scale = np.zeros(self.n)

        for m in range(self.n):
            unit_normal, unit_offset = self.unit_A[m], self.unit_b[m]

            # Row m of the auxiliary update y_m += a_m . x_{m-1} - b_m
            R[m] = P.T @ unit_normal
            s[m] = float(unit_normal @ q) - unit_offset
            s_scale[m] = float(np.abs(unit_normal) @ np.abs(q)) + abs(unit_offset)

            # Advance the prefix map through the projector M_m = I - a_m a_m^T
            if active[m]:
                M = np.eye(p) - np.outer(unit_normal, unit_normal)
                P = M @ P
                q = M @ q + unit_offset * unit_normal

        # After all n half-spaces the prefix map is the cycle map A_m, B_m
        self.A_m, self.B_m, self.R, self.s = P, q, R, s
        self.s_scale = s_scale
        self.active = tuple(active)

    def _advance_cycle(self) -> None:
        """Advance one cycle by the map."""
        self.x = self.A_m @ self.x + self.B_m

    def _activity_check(self) -> tuple:
        """Predict the next active set."""
        # Increment of every auxiliary over one cycle from the current state
        delta = self.R @ self.x + self.s
        y_next = self.y + delta
        return tuple(y_next > 0.0), y_next

    def _replay_cycle(self, cycle: int, active: tuple | np.ndarray) -> None:
        """Record an episode cycle as Dykstra runs it."""
        # Within an episode an active half-space projects the point onto its boundary
        # and an inactive one leaves it alone, so the intermediate points and the
        # auxiliaries of a cycle the solver skipped follow from the previous row
        x = self.x_historical[cycle - 1][-1].copy()
        y = np.where(active, self._recorded_y, 0.0)
        for m in range(self.n):
            if active[m]:
                slack = float(self.unit_A[m] @ x) - self.unit_b[m]
                x = x - slack * self.unit_A[m]
                y[m] += slack
            self.x_historical[cycle][m] = x
        self._recorded_y = y
        if self.plot_errors:
            self.errors_for_plotting[cycle - 1] = y[:, None] * self.unit_A
        self._track_error_at(cycle, x)

    def _track_error_at(self, cycle: int, x: np.ndarray) -> None:
        """Track the error of a recorded point."""
        # The base tracker reads self.x, which after a jump is the landing state
        state, self.x = self.x, x
        try:
            self._track_error(cycle)
        finally:
            self.x = state

    def _record_activity(self, cycle: int, active: tuple) -> None:
        """Store the active set."""
        for m in range(self.n):
            self.active_half_spaces[m][cycle] = 1 if active[m] else 0

    def solve(self) -> ProjectionResult:
        """Project the point."""
        self._prepare()

        # Track error and activity at the initial point
        self._track_error(0)
        self._track_activity(0)
        if self.max_iter == 0:
            return self._format_output()

        # One exact cycle sets x, e, y and the first active set
        active = self._dykstra_cycle(1)
        self._track_error(1)
        if self.plot_active_halfspaces:
            self._record_activity(1, active)
        if self.n == 0:
            # Without constraints the state never moves, so every cycle repeats cycle 1
            for cycle in range(2, self.max_iter + 1):
                self._track_error(cycle)
            return self._format_output()
        self._build_cycle_map(active)

        # Main body: one matrix product per cycle, an exact cycle on every activity change
        for cycle in range(2, self.max_iter + 1):
            active_next, y_next = self._activity_check()
            if active_next != self.active:
                self._sync_e_from_y()
                self._build_cycle_map(self._dykstra_cycle(cycle))
                self._track_error(cycle)
            else:
                self.y = np.where(np.array(active_next), y_next, 0.0)
                self._sync_e_from_y()
                self._advance_cycle()
                self._replay_cycle(cycle, self.active)

            # Track the activity after each complete cycle
            if self.plot_active_halfspaces:
                self._record_activity(cycle, self.active)

        return self._format_output()

    def _format_output(self) -> ProjectionResult:
        """Pack the ProjectionResult."""
        return ProjectionResult(
            projection=self.x,
            path=self.x_historical,
            squared_errors=self.squared_errors if self.track_error else None,
            stalled_errors=self.stalled_errors if self.track_error else None,
            converged_errors=self.converged_errors if self.track_error else None,
            errors_for_plotting=self.errors_for_plotting if self.plot_errors else None,
            active_half_spaces=self.active_half_spaces if self.plot_active_halfspaces else None,
            settled_at=self.settled_at,
            certificate=self.certificate
        )
