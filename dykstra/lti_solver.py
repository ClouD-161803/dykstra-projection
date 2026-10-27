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
then it returns the limit, from the cycle recorded in settled_at on.

A solver built with profile=True also leaves, after each solve, the wall time and the
flops of every stage in STAGES and the counts in COUNTS in its profile attribute, as
SolveProfile.as_dict returns them. Profiling changes neither the path nor any output."""

from collections import namedtuple
from contextlib import nullcontext
from functools import wraps
import time
import numpy as np
from dykstra.convex_projection_solver import ConvexProjectionSolver
from dykstra.lti_numerics import (RESOLVENT_COND_CAP, RHO_CAP, ClosedForm, CycleMap, KKTMemo,
                                  KKTTally, active_auxiliaries, active_set_is_final, advance,
                                  auxiliary_floor, budget_jump_flops, closed_form,
                                  closed_form_activity, closed_form_cycle_flops,
                                  closed_form_flops, cycle_map, cycle_map_flops,
                                  deflated_closed_form, deflated_closed_form_flops, exact_cycle,
                                  exact_cycle_flops, frozen_probe_flops, ignore_blas_flags,
                                  is_frozen, jump_landing_flops, jump_length, jump_length_flops,
                                  kkt_certificate, modal_block_flops, modal_form_flops,
                                  predicted_activity, replay_flops, resolvent_flops,
                                  rounding_floor, stall_crossing, stall_flops,
                                  stepped_cycle_flops, touched_resolvent, unit_constraints)
from dykstra.projection_result import ProjectionResult

PRESETS = ("cycle_map", "closed_form", "envelope", "frozen_stall", "deflated_modal")

# How an episode ended: an active-set change at cycle, the budget running out, or a
# settlement on the limit
_Outcome = namedtuple("_Outcome", "kind cycle")
_BUDGET_EXHAUSTED = _Outcome("budget exhausted", None)
_SETTLED = _Outcome("settled", None)

# Stages of a profiled solve. Each is charged with its own wall time, the stages nested in it
# excluded, and "other" with the time spent between stages, so that the times sum to the solve
STAGES = ("exact_cycles", "cycle_maps", "resolvent", "closed_form_setup", "deflation",
          "closed_form_evaluation", "stepped_evaluation", "modal_scan", "envelope_jumps",
          "budget_jumps", "stall_fast_forwards", "kkt_test", "history", "other")

# Counts of a profiled solve. exact_cycles + evaluated_cycles + skipped_cycles is max_iter
# unless the solve settled, and otherwise the last cycle it ran: settled_at - 1 after a KKT
# test passed at an episode's entry or at a deflated limit, and settled_at after a settlement
# at a final active set. scanned_cycles counts every cycle the modal scan evaluated, those of
# its last block after the switch included, and the skipped cycles are split into those of the
# envelope jumps, the stall fast-forwards and the jump to the budget
COUNTS = ("episodes", "regular_episodes", "singular_episodes", "modal_episodes",
          "stepped_episodes", "exact_cycles", "evaluated_cycles", "skipped_cycles",
          "scanned_cycles", "envelope_jumps", "budget_jumps", "stall_fast_forwards",
          "kkt_attempts", "kkt_passes", "envelope_skipped_cycles", "stall_skipped_cycles",
          "budget_skipped_cycles")

# Stands in for SolveProfile.stage in a solve that is not profiled
_UNPROFILED = nullcontext()


def _switch(cycle: int) -> _Outcome:
    """An active-set change at cycle."""
    return _Outcome("switch", cycle)


class SolveProfile:
    """Wall time, flops and counts of one solve, by stage, kept by a profiled LTISolver."""

    def __init__(self) -> None:
        self.seconds = dict.fromkeys(STAGES, 0.0)
        self.flops = dict.fromkeys(STAGES, 0)
        self.entries = dict.fromkeys(STAGES, 0)
        self.counts = dict.fromkeys(COUNTS, 0)
        # One record per KKT test, as LTISolver._kkt writes it
        self.kkt_tests = []
        self.settled_by = None
        # Sizes of the current episode: its active half-spaces, the rank of their normals,
        # and the coordinates some constraint touches
        self.active = self.rank = self.touched = 0
        self._stack = ["other"]
        self._entering = None
        self._mark = time.perf_counter()

    def stage(self, name: str) -> "SolveProfile":
        """A context that charges the wall time inside it to stage name, apart from the
        stages nested in it."""
        self._entering = name
        return self

    def __enter__(self) -> None:
        self._charge()
        self._stack.append(self._entering)
        self.entries[self._entering] += 1

    def __exit__(self, *exception) -> bool:
        self._charge()
        self._stack.pop()
        return False

    def _charge(self) -> None:
        """Charges the time since the last charge to the innermost open stage."""
        now = time.perf_counter()
        self.seconds[self._stack[-1]] += now - self._mark
        self._mark = now

    def count(self, stage: str | None = None, flops: int = 0, **counts: int) -> None:
        """Adds flops to stage, and every keyword to the count it names."""
        if stage is not None:
            self.flops[stage] += flops
        for name, value in counts.items():
            self.counts[name] += value

    def as_dict(self) -> dict:
        """The record so far: seconds, flops and entries by stage, the counts, the KKT tests in
        the order they ran, and how the solve settled, one of None, "episode entry", "deflated
        limit", "final active set", "finality" and "no constraints"."""
        self._charge()
        return {"seconds": dict(self.seconds), "flops": dict(self.flops),
                "entries": dict(self.entries), "counts": dict(self.counts),
                "kkt_tests": [dict(test) for test in self.kkt_tests],
                "settled_by": self.settled_by}


def _profiled(stage: str):
    """Charges the wall time of an LTISolver method to stage while its solve is profiled."""
    def decorate(method):
        @wraps(method)
        def charged(self, *args, **kwargs):
            if self._profile is None:
                return method(self, *args, **kwargs)
            with self._profile.stage(stage):
                return method(self, *args, **kwargs)
        return charged
    return decorate


class LTISolver(ConvexProjectionSolver):
    """Dykstra through its episodes."""

    PRESET = "deflated_modal"

    def __init__(self, *args, preset: str | None = None, eig_cond_cap: float = 1e8,
                 block_size: int = 4096, profile: bool = False, **kwargs) -> None:
        """
        Initialise the solver; the other arguments are those of ConvexProjectionSolver.

        Args:
            preset: One of PRESETS; defaults to the class's PRESET.
            eig_cond_cap: Largest condition number of the modal basis the
                deflated_modal scan accepts before stepping the episode instead.
            block_size: Cycles the deflated_modal scan evaluates at once.
            profile: Whether each solve leaves its SolveProfile.as_dict record in
                self.profile, which is None otherwise.
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
        if not isinstance(profile, (bool, np.bool_)):
            raise ValueError("profile must be True or False.")

        super().__init__(*args, **kwargs)
        self.preset = preset
        level = PRESETS.index(preset)
        self._closed_forms, self._jumps, self._stalls, self._deflation = (
            level >= 1, level >= 2, level >= 3, level >= 4)
        self.eig_cond_cap = eig_cond_cap
        self.block_size = int(block_size)
        self.settled_at = None
        self.certificate = None
        self._profiling = bool(profile)
        self._profile = None
        self.profile = None

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

    @ignore_blas_flags
    def solve(self) -> ProjectionResult:
        """Project the point."""
        # Every call starts from z with records of its own, so that a second call neither
        # continues from where the first stopped nor writes into the arrays it returned
        self._profile = SolveProfile() if self._profiling else None
        self._reset_state()
        self.settled_at, self.certificate = None, None
        self.unit_A, self.unit_b = unit_constraints(self.A, self.b)
        self.y = np.zeros(self.n)
        self._recorded_y = np.zeros(self.n)
        self._kkt_memo = KKTMemo()

        # Track error and activity at the initial point
        self._track_error(0)
        self._track_activity(0)
        if self.n == 0 and self._deflation:
            # Without constraints z is its own projection, as the KKT test proves with no
            # multipliers, and the C solver reports it as settled after no cycle as well
            self.settled_at, self.certificate = 0, "kkt"
            if self._profile is not None:
                self._profile.settled_by = "no constraints"
        if self.max_iter == 0:
            return self._format_output()

        active = self._exact_cycle(1)
        if self.n == 0:
            # Without constraints the state never moves, so every cycle repeats cycle 1
            for cycle in range(2, self.max_iter + 1):
                self._track_error(cycle)
            if self._profile is not None:
                self._profile.count(skipped_cycles=self.max_iter - 1)
            return self._format_output()

        self.cmap = self._cycle_map(active)
        cycle = 2
        while cycle <= self.max_iter:
            outcome = self._episode(cycle)
            if outcome.kind != "switch":
                break
            self.cmap = self._cycle_map(self._exact_cycle(outcome.cycle))
            cycle = outcome.cycle + 1
        return self._format_output()

    def _exact_cycle(self, cycle: int) -> tuple:
        """Run and record one exact cycle."""
        with self._stage("exact_cycles"):
            self.x, self.y, points = exact_cycle(self.x, self.y, self.unit_A, self.unit_b)
        active = tuple(self.y > 0.0)
        with self._stage("history"):
            self.x_historical[cycle] = points
            self._recorded_y = self.y.copy()
            if self.plot_errors:
                self.errors_for_plotting[cycle - 1] = self.y[:, None] * self.unit_A
            self._track_error(cycle)
            self._record_activity(cycle, active)
        if self._profile is not None:
            self._profile.count("exact_cycles", exact_cycle_flops(self.n, self.dimensions, sum(active)),
                                exact_cycles=1)
        return active

    def _cycle_map(self, active: tuple) -> CycleMap:
        """Build the cycle map of an active set."""
        with self._stage("cycle_maps"):
            cmap = cycle_map(self.unit_A, self.unit_b, active)
        profile = self._profile
        if profile is not None:
            profile.active, profile.rank = sum(active), cmap.span.shape[1]
            profile.touched = int(np.count_nonzero(cmap.touched))
            profile.count("cycle_maps", cycle_map_flops(self.n, self.dimensions, profile.active))
        return cmap

    def _episode(self, start_cycle: int) -> _Outcome:
        """Run an episode of constant activity."""
        active = np.array(self.cmap.active)
        profile = self._profile
        if profile is not None:
            profile.count(episodes=1)

        # A point that passes the KKT test is the projection whatever the episode
        # would do next, and finding it needs no resolvent
        if self._deflation:
            certified = self._kkt(self.x, active, start_cycle, "episode entry")
            if certified is not None:
                return self._settle_certified(*certified, start_cycle)

        # Closed form when I - A_m is well conditioned on the touched coordinates, which
        # needs the active normals to span them, and otherwise the deflated form or exact
        # stepping
        if self._closed_forms:
            with self._stage("resolvent"):
                IA = touched_resolvent(self.cmap)
                regular = np.linalg.cond(IA) < RESOLVENT_COND_CAP
            if profile is not None:
                profile.count("resolvent", resolvent_flops(profile.touched),
                              regular_episodes=int(regular), singular_episodes=int(not regular))
            if regular:
                with self._stage("closed_form_setup"):
                    cf = closed_form(self.cmap, self.x, self.y, IA)
                if profile is not None:
                    profile.count("closed_form_setup", closed_form_flops(self.n, profile.touched, self._jumps))
                return self._closed_form_episode(start_cycle, cf)
            if self._deflation:
                return self._deflated_episode(start_cycle)
        return self._step_episode(start_cycle)

    def _kkt(self, hint: np.ndarray, active: np.ndarray, cycle: int, at: str) -> tuple | None:
        """KKT test at hint, which a profiled solve records with its cost, at cycle and where
        the solver tried it, one of "episode entry", "deflated limit" and "final active set"."""
        profile = self._profile
        if profile is None:
            return kkt_certificate(self.z, self.unit_A, self.unit_b, hint, active, self._kkt_memo)
        tally, before = KKTTally(), profile.seconds["kkt_test"]
        with profile.stage("kkt_test"):
            certified = kkt_certificate(self.z, self.unit_A, self.unit_b, hint, active,
                                        self._kkt_memo, tally=tally)
        passed = certified is not None
        profile.count("kkt_test", tally.flops, kkt_attempts=1, kkt_passes=int(passed))
        profile.kkt_tests.append({
            "cycle": cycle, "at": at, "candidates": tally.candidates, "verified": tally.verified,
            "skipped": tally.skipped, "solves": tally.solves, "flops": tally.flops,
            "flops_without_memo": tally.flops + tally.reused,
            "seconds": profile.seconds["kkt_test"] - before, "passed": passed})
        if passed:
            profile.settled_by = at
        return certified

    # ---- episodes

    @_profiled("stepped_evaluation")
    def _step_episode(self, start_cycle: int) -> _Outcome:
        """Step an episode through its cycle map."""
        profile = self._profile
        if profile is not None:
            profile.count(stepped_episodes=1)
            cycle_flops = stepped_cycle_flops(self.n, self.dimensions, profile.rank)
        # A state can freeze part-way through an episode; probing at offsets 0, 1, 2,
        # 4, ... keeps the probes logarithmic in the episode length
        next_probe = start_cycle
        for cycle in range(start_cycle, self.max_iter + 1):
            if self._stalls and cycle == next_probe:
                next_probe += max(1, cycle - start_cycle)
                if profile is not None:
                    profile.count("stepped_evaluation", frozen_probe_flops(self.dimensions))
                if is_frozen(self.cmap, self.x):
                    return self._stall_episode(cycle, self.x, self.y)
            active_next, y_next = predicted_activity(self.cmap, self.x, self.y)
            if active_next != self.cmap.active:
                return _switch(cycle)
            self.y = y_next
            self.x = advance(self.cmap, self.x)
            self._record_cycle(cycle)
            if profile is not None:
                profile.count("stepped_evaluation", cycle_flops, evaluated_cycles=1)
        return _BUDGET_EXHAUSTED

    @_profiled("stall_fast_forwards")
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
        if self._profile is not None:
            self._profile.count("stall_fast_forwards", stall_flops(self.n, self.dimensions),
                                stall_fast_forwards=int(k >= 1), skipped_cycles=k,
                                stall_skipped_cycles=k)
        if k >= 1:
            self._set_state(x, y + k * delta)
            for cycle in range(start_cycle, start_cycle + k):
                self._record_cycle(cycle)
        return _switch(start_cycle + k) if start_cycle + k <= self.max_iter else _BUDGET_EXHAUSTED

    @_profiled("closed_form_evaluation")
    def _closed_form_episode(self, start_cycle: int, cf: ClosedForm) -> _Outcome:
        """Scan a regular episode in closed form."""
        # The transient has no component on an untouched coordinate, where A_m is the
        # identity and would hold the contraction rate at one
        touched = np.ix_(self.cmap.touched, self.cmap.touched)
        with self._stage("closed_form_setup"):
            rho = min(float(np.linalg.svd(cf.A_m[touched], compute_uv=False)[0]), RHO_CAP) if self._jumps else None
        profile = self._profile
        if profile is not None:
            cycle_flops = closed_form_cycle_flops(self.n, self.dimensions)
        z_prev = self.x - cf.x_inf
        t = 1
        while start_cycle + t - 1 <= self.max_iter:
            cycle = start_cycle + t - 1

            z_t = cf.A_m @ z_prev
            y_t = active_auxiliaries(cf, t, z_t)
            if profile is not None:
                profile.count("closed_form_evaluation", cycle_flops)

            # The cycle that changes the active set is run exactly, so the episode ends
            # in the state before it
            if not np.array_equal(closed_form_activity(cf, self.cmap, t, y_t, z_t, z_prev), cf.active):
                self._set_state(cf.x_inf + z_prev, active_auxiliaries(cf, t - 1, z_prev))
                return _switch(cycle)

            self._set_state(cf.x_inf + z_t, y_t)
            self._record_cycle(cycle)
            if profile is not None:
                profile.count(evaluated_cycles=1)
            if active_set_is_final(cf, t, z_t):
                return self._settle_final(cf, cycle, t)

            # Jump the cycles the envelope certifies switch-free
            if self._jumps:
                k, z_t = self._jump(cf, t, z_t, cycle, rho)
                t += k
            t += 1
            z_prev = z_t
        return _BUDGET_EXHAUSTED

    @_profiled("envelope_jumps")
    def _jump(self, cf: ClosedForm, t: int, z_t: np.ndarray, cycle: int, rho: float) -> tuple:
        """Jump and verify the landing."""
        k = min(jump_length(cf, t, z_t, rho), self.max_iter - cycle)
        profile = self._profile
        if profile is not None:
            # every horizon counted, which overstates a jump an inactive half-space blocks first
            profile.count("envelope_jumps", jump_length_flops(self.n, profile.active))
        # The envelope proves every cycle up to the landing switch-free, reading drifts and
        # slacks within rounding as zero as closed_form_activity does; the landing state is
        # computed with rounding, so its signs are checked, and a shorter jump stays inside
        # the proven run
        while k >= 1:
            z_before = np.linalg.matrix_power(cf.A_m, k - 1) @ z_t
            z_after = cf.A_m @ z_before
            y_after = active_auxiliaries(cf, t + k, z_after)
            if profile is not None:
                profile.count("envelope_jumps", jump_landing_flops(self.n, self.dimensions, k - 1))
            if np.array_equal(closed_form_activity(cf, self.cmap, t + k, y_after, z_after, z_before),
                              cf.active):
                self._set_state(cf.x_inf + z_after, y_after)
                for later_cycle in range(cycle + 1, cycle + k + 1):
                    self._record_cycle(later_cycle)
                if profile is not None:
                    profile.count(envelope_jumps=1, skipped_cycles=k, envelope_skipped_cycles=k)
                return k, z_after
            k //= 2
        return 0, z_t

    @_profiled("modal_scan")
    def _deflated_episode(self, start_cycle: int) -> _Outcome:
        """Scan a singular episode in modal form."""
        active = np.array(self.cmap.active)
        profile = self._profile
        with self._stage("deflation"):
            cf = deflated_closed_form(self.cmap, self.x, self.y) if active.any() else None
        if profile is not None and active.any():
            profile.count("deflation", deflated_closed_form_flops(self.n, profile.touched, profile.rank))
        if cf is None:
            return self._step_episode(start_cycle)
        certified = self._kkt(cf.x_inf, active, start_cycle, "deflated limit")
        if certified is not None:
            return self._settle_certified(*certified, start_cycle)

        with self._stage("deflation"):
            # Modes lambda_i of the contracting block T = Q^T A_m Q, and the modal
            # coefficients mu (auxiliaries) and nu (slacks) of the transient
            Q = self.cmap.span
            modes, V = np.linalg.eig(Q.T @ self.cmap.A_m @ Q)
            if np.max(np.abs(modes)) >= 1.0 - 1e-12 or np.linalg.cond(V) >= self.eig_cond_cap:
                return self._step_episode(start_cycle)
            coords = np.linalg.solve(V, Q.T @ (self.x - cf.x_inf))
            mu = (cf.RIA @ Q) @ V * coords[None, :]
            nu = (self.cmap.R @ Q) @ V * coords[None, :]
            moduli, env_y, env_g = np.abs(modes), np.abs(mu), np.abs(nu)
            env_y_scale = (cf.RIA_scale @ np.abs(Q)) @ np.abs(V) * np.abs(coords)[None, :]

            def modal_state(t: int) -> tuple:
                """State and auxiliaries at cycle t."""
                powers = modes ** t
                return cf.x_inf + (Q @ (V @ (powers * coords))).real, cf.G + t * cf.beta - (mu @ powers).real

            # Activity changes only on a value beyond rounding, as in closed_form_activity
            g_floor = cf.beta_floor + rounding_floor(self.cmap.R_scale @ np.abs(Q) @ (np.abs(V) @ np.abs(coords)))
        if profile is not None:
            profile.count("deflation", modal_form_flops(self.n, self.dimensions, profile.rank),
                          modal_episodes=1)

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
            if profile is not None:
                profile.count("modal_scan", modal_block_flops(self.n, self.dimensions, profile.rank, block),
                              scanned_cycles=block)
            t_range = np.arange(t + 1, t + block + 1)
            powers = modes[None, :] ** t_range[:, None]
            powers_prev = modes[None, :] ** (t_range[:, None] - 1)
            y_vals = cf.G[None, :] + t_range[:, None] * cf.beta[None, :] - (powers @ mu.T).real
            g_vals = cf.beta[None, :] + (powers_prev @ nu.T).real
            y_floor = auxiliary_floor(cf, t_range[:, None]) + rounding_floor(np.abs(powers) @ env_y_scale.T)
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
                if profile is not None:
                    profile.count(evaluated_cycles=t_switch - 1 - t)
                return _switch(start_cycle + t_switch - 1)

            t += block
            self._set_state(*modal_state(t))
            for cycle in range(start_cycle + t - block, start_cycle + t):
                self._record_cycle(cycle)
            if profile is not None:
                profile.count(evaluated_cycles=block)

            # Clear every half-space whose envelope rules out a sign change at every
            # later cycle; once all are cleared the active set is final
            decay = moduli ** t
            clear_inactive = cf.inactive & (cf.beta + env_g @ decay <= g_floor)
            clear_active = (cf.active & (cf.beta >= -cf.beta_floor)
                            & (cf.G + (t + 1) * cf.beta - env_y @ (decay * moduli) > 0.0))
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
            if self._profile is not None:
                self._profile.settled_by = "finality"
            self._record_limit(cycle)
            return _SETTLED
        certified = self._kkt(cf.x_inf, cf.active, cycle, "final active set")
        if certified is not None:
            return self._settle_certified(*certified, cycle)

        # No switch can follow, so the budget's iterate is one closed-form jump away
        k = self.max_iter - cycle
        if k >= 1:
            with self._stage("budget_jumps"):
                z_end = np.linalg.matrix_power(cf.A_m, k) @ (self.x - cf.x_inf)
                self._set_state(cf.x_inf + z_end, active_auxiliaries(cf, t + k, z_end))
            for later_cycle in range(cycle + 1, self.max_iter + 1):
                self._record_cycle(later_cycle)
            if self._profile is not None:
                self._profile.count("budget_jumps", budget_jump_flops(self.n, self.dimensions, k),
                                    budget_jumps=1, skipped_cycles=k, budget_skipped_cycles=k)
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

    @_profiled("history")
    def _record_cycle(self, cycle: int) -> None:
        """Record an episode cycle as Dykstra runs it."""
        # Within an episode an active half-space projects the point onto its boundary
        # and an inactive one leaves it alone, so the intermediate points and the
        # auxiliaries of a cycle the solver skipped follow from the previous row
        active = self.cmap.active
        x = self.x_historical[cycle - 1][-1].copy()
        y = np.where(active, self._recorded_y, 0.0)
        row = self.x_historical[cycle]
        start = 0
        # The inactive half-spaces between two active ones all store the same point
        for m in np.flatnonzero(active):
            row[start:m] = x
            slack = float(self.unit_A[m] @ x) - self.unit_b[m]
            x = x - slack * self.unit_A[m]
            y[m] += slack
            row[m] = x
            start = m + 1
        row[start:] = x
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
        if self._profile is not None:
            self._profile.count("history", replay_flops(self.dimensions, self._profile.active))

    @_profiled("history")
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

    def _stage(self, name: str):
        """A context that charges its wall time to stage name while the solve is profiled."""
        return _UNPROFILED if self._profile is None else self._profile.stage(name)

    def _format_output(self) -> ProjectionResult:
        """Pack the ProjectionResult, and leave the profile of a profiled solve in self.profile."""
        if self._profile is not None:
            self.profile = self._profile.as_dict()
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
