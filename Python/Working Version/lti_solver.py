"""LTI solvers: between changes of the active set, Dykstra's iteration is a linear
time-invariant system, so an episode of constant activity can be stepped through its
cycle map, evaluated in closed form or skipped, with one exact Dykstra cycle whenever
the active set changes. LTISolver runs one of five presets, each adding one technique
to the one before, so that their effect can be compared:

    cycle_map       steps every cycle through the episode's cycle map (Ver1)
    closed_form     evaluates regular episodes in closed form and settles once the
                    active set is proven final (Ver2)
    envelope        also jumps the runs of cycles an envelope bound certifies
                    switch-free (Ver3)
    frozen_stall    also fast-forwards frozen stalls, in which only the auxiliaries
                    move (Ver4)
    deflated_modal  also scans singular episodes in their deflated modal form, and
                    settles only on a point that passes the KKT test (Ver5)

Every preset returns Dykstra's own iterate after max_iter cycles unless it settles;
then it returns the limit, from the cycle recorded in settled_at on."""

from collections import namedtuple
import numpy as np
from convex_projection_solver import ConvexProjectionSolver
from lti_numerics import (RESOLVENT_COND_CAP, RHO_CAP, ClosedForm, active_auxiliaries,
                          active_set_is_final, advance, auxiliary_floor, closed_form,
                          closed_form_activity, cycle_map, deflated_closed_form, exact_cycle,
                          is_frozen, jump_length, kkt_certificate, predicted_activity,
                          rounding_floor, stall_crossing, unit_constraints)
from projection_result import ProjectionResult

PRESETS = ("cycle_map", "closed_form", "envelope", "frozen_stall", "deflated_modal")

# How an episode ended: an active-set change at cycle, the budget running out, or a
# settlement on the limit
_Outcome = namedtuple("_Outcome", "kind cycle")
_BUDGET_EXHAUSTED = _Outcome("budget exhausted", None)
_SETTLED = _Outcome("settled", None)


def _switch(cycle: int) -> _Outcome:
    """An active-set change at cycle."""
    return _Outcome("switch", cycle)


class LTISolver(ConvexProjectionSolver):
    """Dykstra through its episodes."""

    PRESET = "deflated_modal"

    def __init__(self, *args, preset: str | None = None, eig_cond_cap: float = 1e8,
                 block_size: int = 4096, **kwargs) -> None:
        """
        Initialise the solver; the other arguments are those of ConvexProjectionSolver.

        Args:
            preset: One of PRESETS; defaults to the class's PRESET.
            eig_cond_cap: Largest condition number of the modal basis the
                deflated_modal scan accepts before stepping the episode instead.
            block_size: Cycles the deflated_modal scan evaluates at once.
        """
        preset = self.PRESET if preset is None else preset
        if preset not in PRESETS:
            raise ValueError(f"preset must be one of {', '.join(PRESETS)}.")
        try:
            eig_cond_cap = float(eig_cond_cap)
        except (TypeError, ValueError) as error:
            raise ValueError("eig_cond_cap must be a finite value greater than one.") from error
        if not np.isfinite(eig_cond_cap) or eig_cond_cap <= 1.0:
            raise ValueError("eig_cond_cap must be a finite value greater than one.")
        if (isinstance(block_size, (bool, np.bool_)) or
                not isinstance(block_size, (int, np.integer)) or block_size < 1):
            raise ValueError("block_size must be a positive integer.")

        super().__init__(*args, **kwargs)
        self.preset = preset
        level = PRESETS.index(preset)
        self._closed_forms, self._jumps, self._stalls, self._deflation = (
            level >= 1, level >= 2, level >= 3, level >= 4)
        self.eig_cond_cap = eig_cond_cap
        self.block_size = int(block_size)
        self.settled_at = None
        self.certificate = None

    @property
    def settled(self) -> bool:
        """Result proven to be the projection."""
        # Unlike result.is_settled(), false for a limit proven only final
        return self.certificate == "kkt"

    def _update_error(self, m: int, x_temp: np.ndarray, x: np.ndarray, index: int) -> None:
        """Not used."""
        # The state carries one scalar auxiliary per half-space, not correction vectors
        raise NotImplementedError("LTISolver keeps scalar auxiliaries")

    # ---- solve: one exact cycle, then episodes separated by exact switching cycles

    def solve(self) -> ProjectionResult:
        """Project the point."""
        self.unit_A, self.unit_b = unit_constraints(self.A, self.b)
        self.y = np.zeros(self.n)
        self._recorded_y = np.zeros(self.n)

        # Track error and activity at the initial point
        self._track_error(0)
        self._track_activity(0)
        if self.max_iter == 0:
            return self._format_output()

        active = self._exact_cycle(1)
        if self.n == 0:
            # Without constraints the state never moves, so every cycle repeats cycle 1
            for cycle in range(2, self.max_iter + 1):
                self._track_error(cycle)
            return self._format_output()

        self.cmap = cycle_map(self.unit_A, self.unit_b, active)
        cycle = 2
        while cycle <= self.max_iter:
            outcome = self._episode(cycle)
            if outcome.kind != "switch":
                break
            self.cmap = cycle_map(self.unit_A, self.unit_b, self._exact_cycle(outcome.cycle))
            cycle = outcome.cycle + 1
        return self._format_output()

    def _exact_cycle(self, cycle: int) -> tuple:
        """Run and record one exact cycle."""
        self.x, self.y, points = exact_cycle(self.x, self.y, self.unit_A, self.unit_b)
        active = tuple(self.y > 0.0)
        self.x_historical[cycle] = points
        self._recorded_y = self.y.copy()
        if self.plot_errors:
            self.errors_for_plotting[cycle - 1] = self.y[:, None] * self.unit_A
        self._track_error(cycle)
        self._record_activity(cycle, active)
        return active

    def _episode(self, start_cycle: int) -> _Outcome:
        """Run an episode of constant activity."""
        active = np.array(self.cmap.active)

        # A point that passes the KKT test is the projection whatever the episode
        # would do next, and finding it needs no resolvent
        if self._deflation:
            certified = kkt_certificate(self.z, self.unit_A, self.unit_b, self.x, active)
            if certified is not None:
                return self._settle_certified(*certified, start_cycle)

        # Closed form when I - A_m is well conditioned, which needs the active normals
        # to span the space; otherwise the deflated form or exact stepping
        if self._closed_forms:
            IA = np.eye(len(self.x)) - self.cmap.A_m
            if np.linalg.cond(IA) < RESOLVENT_COND_CAP:
                return self._closed_form_episode(start_cycle, closed_form(self.cmap, self.x, self.y, IA))
            if self._deflation:
                return self._deflated_episode(start_cycle)
        return self._step_episode(start_cycle)

    # ---- episodes

    def _step_episode(self, start_cycle: int) -> _Outcome:
        """Step an episode through its cycle map."""
        # A state can freeze part-way through an episode; probing at offsets 0, 1, 2,
        # 4, ... keeps the probes logarithmic in the episode length
        next_probe = start_cycle
        for cycle in range(start_cycle, self.max_iter + 1):
            if self._stalls and cycle == next_probe:
                next_probe += max(1, cycle - start_cycle)
                if is_frozen(self.cmap, self.x):
                    return self._stall_episode(cycle, self.x, self.y)
            active_next, y_next = predicted_activity(self.cmap, self.x, self.y)
            if active_next != self.cmap.active:
                return _switch(cycle)
            self.y = y_next
            self.x = advance(self.cmap, self.x)
            self._record_cycle(cycle)
        return _BUDGET_EXHAUSTED

    def _stall_episode(self, start_cycle: int, x: np.ndarray, y: np.ndarray) -> _Outcome:
        """Fast-forward a frozen stall."""
        # Inactive auxiliaries are zero; every auxiliary moves by delta per cycle, which
        # is zero only up to rounding
        active = np.array(self.cmap.active)
        y = np.where(active, y, 0.0)
        delta = self.cmap.R @ x + self.cmap.s
        delta_floor = rounding_floor(self.cmap.R_scale @ np.abs(x) + self.cmap.s_scale)

        # Jump to the cycle before the crossing, or through the rest of the budget when
        # the crossing lies beyond it, applying the k increments at once while the
        # state stays frozen
        remaining = self.max_iter - start_cycle + 1
        crossing = stall_crossing(y, delta, delta_floor, active, self.max_iter + 1)
        k = int(crossing) - 1 if crossing <= remaining else remaining
        if k >= 1:
            self._set_state(x, y + k * delta)
            for cycle in range(start_cycle, start_cycle + k):
                self._record_cycle(cycle)
        return _switch(start_cycle + k) if start_cycle + k <= self.max_iter else _BUDGET_EXHAUSTED

    def _closed_form_episode(self, start_cycle: int, cf: ClosedForm) -> _Outcome:
        """Scan a regular episode in closed form."""
        rho = min(float(np.linalg.svd(cf.A_m, compute_uv=False)[0]), RHO_CAP) if self._jumps else None
        z_prev = self.x - cf.x_inf
        t = 1
        while start_cycle + t - 1 <= self.max_iter:
            cycle = start_cycle + t - 1

            # Transient z_t = A_m z_{t-1}, auxiliaries y_m and slacks g_j at cycle t
            z_t = cf.A_m @ z_prev
            y_t = active_auxiliaries(cf, t, z_t)

            # Stop just before the cycle on which the active set changes
            if not np.array_equal(closed_form_activity(cf, self.cmap, t, y_t, z_t, z_prev), cf.active):
                self._set_state(cf.x_inf + z_prev, active_auxiliaries(cf, t - 1, z_prev))
                return _switch(cycle)

            self._set_state(cf.x_inf + z_t, y_t)
            self._record_cycle(cycle)
            if active_set_is_final(cf, t, z_t):
                return self._settle_final(cf, cycle, t)

            # Jump the cycles the envelope certifies switch-free
            if self._jumps:
                k, z_t = self._jump(cf, t, z_t, cycle, rho)
                t += k
            t += 1
            z_prev = z_t
        return _BUDGET_EXHAUSTED

    def _jump(self, cf: ClosedForm, t: int, z_t: np.ndarray, cycle: int, rho: float) -> tuple:
        """Jump and verify the landing."""
        k = min(jump_length(cf, t, z_t, rho), self.max_iter - cycle)
        # Land at z_{t+k} = A_m^k z_t; halve the jump whenever the active set did not hold
        while k >= 1:
            z_before = np.linalg.matrix_power(cf.A_m, k - 1) @ z_t
            z_after = cf.A_m @ z_before
            y_after = active_auxiliaries(cf, t + k, z_after)
            if np.array_equal(closed_form_activity(cf, self.cmap, t + k, y_after, z_after, z_before),
                              cf.active):
                self._set_state(cf.x_inf + z_after, y_after)
                for later_cycle in range(cycle + 1, cycle + k + 1):
                    self._record_cycle(later_cycle)
                return k, z_after
            k //= 2
        return 0, z_t

    def _deflated_episode(self, start_cycle: int) -> _Outcome:
        """Scan a singular episode in modal form."""
        active = np.array(self.cmap.active)
        cf = deflated_closed_form(self.cmap, self.x, self.y) if active.any() else None
        if cf is None:
            return self._step_episode(start_cycle)
        certified = kkt_certificate(self.z, self.unit_A, self.unit_b, cf.x_inf, active)
        if certified is not None:
            return self._settle_certified(*certified, start_cycle)

        # Modes lambda_i of the contracting block T = Q^T A_m Q, and the modal
        # coefficients mu (auxiliaries) and nu (slacks) of the transient
        Q = self.cmap.span
        lam, V = np.linalg.eig(Q.T @ self.cmap.A_m @ Q)
        if np.max(np.abs(lam)) >= 1.0 - 1e-12 or np.linalg.cond(V) >= self.eig_cond_cap:
            return self._step_episode(start_cycle)
        coords = np.linalg.solve(V, Q.T @ (self.x - cf.x_inf))
        mu = (cf.RIA @ Q) @ V * coords[None, :]
        nu = (self.cmap.R @ Q) @ V * coords[None, :]
        abs_lam, env_y, env_g = np.abs(lam), np.abs(mu), np.abs(nu)
        env_y_scale = (cf.RIA_scale @ np.abs(Q)) @ np.abs(V) * np.abs(coords)[None, :]

        def modal_state(t: int) -> tuple:
            """State and auxiliaries at cycle t."""
            lam_t = lam ** t
            return cf.x_inf + (Q @ (V @ (lam_t * coords))).real, cf.G + t * cf.beta - (mu @ lam_t).real

        # Activity changes only on a value beyond rounding, as in closed_form_activity
        g_floor = cf.beta_floor + rounding_floor(self.cmap.R_scale @ np.abs(Q) @ (np.abs(V) @ np.abs(coords)))

        # Scan in blocks; a half-space cleared by its envelope leaves the watch set
        watch = np.ones(self.n, dtype=bool)
        t = 0
        while start_cycle + t <= self.max_iter:
            # A frozen state is fast-forwarded straight to its switching cycle
            x_t, y_t = modal_state(t)
            if is_frozen(self.cmap, x_t):
                return self._stall_episode(start_cycle + t, x_t, y_t)

            # Auxiliaries y_m and slacks g_j at every cycle of the block
            block = min(self.block_size, self.max_iter - (start_cycle + t) + 1)
            t_range = np.arange(t + 1, t + block + 1)
            lam_powers = lam[None, :] ** t_range[:, None]
            lam_powers_prev = lam[None, :] ** (t_range[:, None] - 1)
            y_vals = cf.G[None, :] + t_range[:, None] * cf.beta[None, :] - (lam_powers @ mu.T).real
            g_vals = cf.beta[None, :] + (lam_powers_prev @ nu.T).real
            y_floor = auxiliary_floor(cf, t_range[:, None]) + rounding_floor(np.abs(lam_powers) @ env_y_scale.T)
            signs = np.where(active[None, :], y_vals > -y_floor, g_vals > g_floor[None, :])
            flip_rows = np.where(((signs != active[None, :]) & watch[None, :]).any(axis=1))[0]

            # Stop just before the first cycle whose signs differ from the active set;
            # the state at cycle t is already committed, and rebuilding it from the modes
            # would move it by about eps times the condition of V
            if flip_rows.size:
                t_switch = t + 1 + int(flip_rows[0])
                if t_switch - 1 > t:
                    self._set_state(*modal_state(t_switch - 1))
                for cycle in range(start_cycle + t, start_cycle + t_switch - 1):
                    self._record_cycle(cycle)
                return _switch(start_cycle + t_switch - 1)

            t += block
            self._set_state(*modal_state(t))
            for cycle in range(start_cycle + t - block, start_cycle + t):
                self._record_cycle(cycle)

            # Clear every half-space whose envelope rules out a sign change at every
            # later cycle; once all are cleared the active set is final
            decay = abs_lam ** t
            clear_inactive = cf.inactive & (cf.beta + env_g @ decay <= g_floor)
            clear_active = (cf.active & (cf.beta >= -cf.beta_floor)
                            & (cf.G + (t + 1) * cf.beta - env_y @ (decay * abs_lam) > 0.0))
            watch &= ~(clear_inactive | clear_active)
            if not watch.any():
                return self._settle_final(cf, start_cycle + t - 1, t)
        return _BUDGET_EXHAUSTED

    # ---- settlement

    def _settle_final(self, cf: ClosedForm, cycle: int, t: int) -> _Outcome:
        """Settle once the active set is final."""
        # Before the deflated_modal preset the limit is the fixed point of the cycle
        # map; with it the KKT test decides, since that fixed point can still miss the
        # projection
        if not self._deflation:
            self._set_state(cf.x_inf, cf.G + t * cf.beta)
            self.settled_at, self.certificate = cycle, "finality"
            self._record_limit(cycle)
            return _SETTLED
        certified = kkt_certificate(self.z, self.unit_A, self.unit_b, cf.x_inf, cf.active)
        if certified is not None:
            return self._settle_certified(*certified, cycle)

        # No switch can follow, so the budget's iterate is one closed-form jump away
        k = self.max_iter - cycle
        if k >= 1:
            z_end = np.linalg.matrix_power(cf.A_m, k) @ (self.x - cf.x_inf)
            self._set_state(cf.x_inf + z_end, active_auxiliaries(cf, t + k, z_end))
            for later_cycle in range(cycle + 1, self.max_iter + 1):
                self._record_cycle(later_cycle)
        return _BUDGET_EXHAUSTED

    def _settle_certified(self, x: np.ndarray, multipliers: np.ndarray, first_cycle: int) -> _Outcome:
        """Settle on the projection."""
        # The multipliers are the auxiliaries of the limit, x + sum y_m a_m = z, and
        # their support its active set
        self.x, self.y = x, multipliers
        self.settled_at, self.certificate = first_cycle, "kkt"
        self._record_limit(first_cycle, multipliers > 0.0)
        return _SETTLED

    def _set_state(self, x: np.ndarray, y: np.ndarray) -> None:
        """Set the state within the episode."""
        # An active auxiliary kept within rounding of zero stays nonnegative, as in Dykstra
        self.x = x
        self.y = np.where(self.cmap.active, np.maximum(y, 0.0), 0.0)

    # ---- recording

    def _record_cycle(self, cycle: int) -> None:
        """Record an episode cycle as Dykstra runs it."""
        # Within an episode an active half-space projects the point onto its boundary
        # and an inactive one leaves it alone, so the intermediate points and the
        # auxiliaries of a cycle the solver skipped follow from the previous row
        active = self.cmap.active
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

        # The base tracker reads self.x, which after a jump is the landing state
        state, self.x = self.x, x
        try:
            self._track_error(cycle)
        finally:
            self.x = state
        self._record_activity(cycle, active)

    def _record_limit(self, first_cycle: int, active: tuple | np.ndarray | None = None) -> None:
        """Record the limit for every remaining cycle."""
        active = self.cmap.active if active is None else active
        self.x_historical[first_cycle:] = self.x
        if self.plot_errors:
            self.errors_for_plotting[first_cycle - 1:] = self.y[:, None] * self.unit_A
        for cycle in range(first_cycle, self.max_iter + 1):
            self._track_error(cycle)
        if self.plot_active_halfspaces:
            self.active_half_spaces[:, first_cycle:] = np.asarray(active, dtype=float)[:, None]
        self._recorded_y = self.y.copy()

    def _record_activity(self, cycle: int, active: tuple | np.ndarray) -> None:
        """Store the active set."""
        if self.plot_active_halfspaces:
            self.active_half_spaces[:, cycle] = np.asarray(active, dtype=float)

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


class LTIVer1Solver(LTISolver):
    """Cycle-map stepping."""
    PRESET = "cycle_map"


class LTIVer2Solver(LTISolver):
    """Closed-form episodes."""
    PRESET = "closed_form"


class LTIVer3Solver(LTISolver):
    """Envelope-bounded jumps."""
    PRESET = "envelope"


class LTIVer4Solver(LTISolver):
    """Frozen-stall fast-forward."""
    PRESET = "frozen_stall"


class LTIVer5Solver(LTISolver):
    """Deflated modal episodes."""
    PRESET = "deflated_modal"
