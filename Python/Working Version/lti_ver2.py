"""LTI Ver2: while the active set is constant, evaluates the auxiliary variables
and slacks in closed form to find the first cycle on which the active set changes,
jumps the state there, and settles at the fixed point once no change can follow."""

from collections import namedtuple
import numpy as np
from lti_ver1 import LTIVer1Solver
from projection_result import ProjectionResult

# A drift beta_m or floor Gamma_j smaller than this counts as zero
_BETA_TOL = 1e-9

# Constants of one episode: the cycle map A_m, its fixed point x_inf, the matrix
# RIA = R (I - A_m)^-1, the levels G and drifts beta of the auxiliaries (floors
# Gamma on inactive rows), the row norms of RIA and R, and the activity masks
_CF = namedtuple("_CF", "A_m x_inf RIA G beta row_RIA row_R active inactive")


class LTIVer2Solver(LTIVer1Solver):
    """Closed-form activity episodes."""

    def _commit_auxiliaries(self, y: np.ndarray, active: np.ndarray) -> None:
        """Set the active auxiliaries."""
        self.y = np.where(active, y, 0.0)
        self._sync_e_from_y()

    def _set_state(self, x: np.ndarray, y: np.ndarray, active: np.ndarray) -> None:
        """Set the state and auxiliaries."""
        self.x = x
        self._commit_auxiliaries(y, active)

    def _setup_closed_form(self, IA_inv: np.ndarray) -> _CF:
        """Constants of the closed form."""
        # Fixed point x_inf = (I - A_m)^-1 B_m and the transient z_0 = x - x_inf
        x_inf = IA_inv @ self.B_m
        RIA = self.R @ IA_inv
        active = np.array(self.active)
        z_0 = self.x - x_inf

        # Level G_m of each auxiliary without its transient, drift beta_m per cycle
        G = self.y + RIA @ z_0
        beta = self.R @ x_inf + self.s
        return _CF(A_m=self.A_m, x_inf=x_inf, RIA=RIA, G=G, beta=beta,
                   row_RIA=np.linalg.norm(RIA, axis=1), row_R=np.linalg.norm(self.R, axis=1),
                   active=active, inactive=~active)

    @staticmethod
    def _active_auxiliaries(cf: _CF, t: int, z_t: np.ndarray) -> np.ndarray:
        """Closed-form active auxiliaries."""
        # Auxiliary y_m at cycle t: level plus drift, minus the decaying transient
        return cf.G + t * cf.beta - cf.RIA @ z_t

    def _active_set_is_final(self, cf: _CF, t: int, z_t: np.ndarray) -> bool:
        """No further activity change."""
        # Every later transient is bounded by the current ||z_t||
        z_norm = float(np.linalg.norm(z_t))
        active_ok = not cf.active.any() or (
            np.all(cf.beta[cf.active] >= -_BETA_TOL)
            and np.all(cf.G[cf.active] + t * cf.beta[cf.active]
                       - cf.row_RIA[cf.active] * z_norm > 0.0))
        inactive_ok = not cf.inactive.any() or (
            np.all(cf.beta[cf.inactive] <= _BETA_TOL)
            and np.all(cf.beta[cf.inactive] + cf.row_R[cf.inactive] * z_norm < 0.0))
        return active_ok and inactive_ok

    def _settle_at_fixed_point(self, cf: _CF, cycle: int, t: int) -> None:
        """Jump to the fixed point."""
        self._set_state(cf.x_inf, cf.G + t * cf.beta, cf.active)
        for later_cycle in range(cycle + 1, self.max_iter + 1):
            self._record_cycle(later_cycle, cf.active)

    def _closed_form_episode(self, start_cycle: int, IA_inv: np.ndarray) -> int | None:
        """Scan an episode to its switch."""
        cf = self._setup_closed_form(IA_inv)
        z_prev = self.x - cf.x_inf
        for t in range(1, self.max_iter - start_cycle + 2):
            cycle = start_cycle + t - 1

            # Transient z_t = A_m z_{t-1}, auxiliaries y_m and slacks g_j at cycle t
            z_t = cf.A_m @ z_prev
            y_t = self._active_auxiliaries(cf, t, z_t)
            g_t = cf.beta + self.R @ z_prev
            active_t = np.where(cf.active, y_t, g_t) > 0.0

            # Stop just before the cycle on which the active set changes
            if not np.array_equal(active_t, cf.active):
                self._set_state(cf.x_inf + z_prev,
                                self._active_auxiliaries(cf, t - 1, z_prev), cf.active)
                return cycle

            self._set_state(cf.x_inf + z_t, y_t, cf.active)
            self._record_cycle(cycle, active_t)
            if self._active_set_is_final(cf, t, z_t):
                self._settle_at_fixed_point(cf, cycle, t)
                return None
            z_prev = z_t
        return None

    def _step_episode(self, start_cycle: int) -> int | None:
        """Step a singular episode."""
        for cycle in range(start_cycle, self.max_iter + 1):
            active_next, y_next = self._activity_check()
            if tuple(active_next) != self.active:
                self._sync_e_from_y()
                return cycle
            self._commit_auxiliaries(y_next, np.array(active_next))
            self._advance_cycle()
            self._record_cycle(cycle, active_next)
        return None

    def _accelerate(self, start_cycle: int) -> None:
        """Run episodes to the budget."""
        p = len(self.x)
        cycle = start_cycle
        while cycle <= self.max_iter:
            # The closed form needs I - A_m invertible, i.e. active normals spanning the space
            IA = np.eye(p) - self.A_m
            if np.linalg.cond(IA) < 1e12:
                switch_cycle = self._closed_form_episode(cycle, np.linalg.inv(IA))
            else:
                switch_cycle = self._step_episode(cycle)
            if switch_cycle is None:
                return
            self._perform_switch(switch_cycle)
            cycle = switch_cycle + 1

    def _perform_switch(self, switch_cycle: int) -> None:
        """Run the switching cycle."""
        self.active = self._dykstra_cycle(switch_cycle)
        self._record_cycle(switch_cycle, self.active, record_path=False)
        self._build_cycle_map(self.active)

    def _record_cycle(self, cycle: int, active: tuple | np.ndarray, record_path: bool = True) -> None:
        """Record one cycle."""
        if record_path:
            self.x_historical[cycle][:, :] = self.x
        self._track_error(cycle)
        if self.plot_active_halfspaces:
            self._record_activity(cycle, tuple(active))

    def solve(self) -> ProjectionResult:
        """Project the point."""
        self._prepare()

        # Track error and activity at the initial point
        self._track_error(0)
        self._track_activity(0)

        # One exact cycle sets x, e, y and the first active set
        active = self._dykstra_cycle(1)
        self._record_cycle(1, active, record_path=False)
        if self.n == 0:
            return self._format_output()

        # Episodes of constant activity, each ended by an exact switching cycle
        self._build_cycle_map(active)
        self._accelerate(2)
        return self._format_output()
