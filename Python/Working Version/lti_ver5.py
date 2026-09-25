"""LTI Ver5: runs episodes whose active normals do not span the space in the
coordinates of that span, scanning the closed-form auxiliaries and slacks exactly
in blocks, settling by a KKT test on the episode limit, and jumping frozen
stalls to their crossing."""

import numpy as np
from lti_ver2 import _CF
from lti_ver4 import LTIVer4Solver


class LTIVer5Solver(LTIVer4Solver):
    """Deflated modal episodes."""

    def __init__(self, *args, eig_cond_cap: float = 1e8, block_size: int = 4096, **kwargs) -> None:
        """Solver options."""
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
        self.eig_cond_cap = eig_cond_cap
        self.block_size = int(block_size)

        # True once the result is proven to be the projection
        self.settled = False

    def _settle_at_fixed_point(self, cf: _CF, cycle: int, t: int) -> None:
        """Jump to the fixed point."""
        self.settled = True
        super()._settle_at_fixed_point(cf, cycle, t)

    def _accelerate(self, start_cycle: int) -> None:
        """Run episodes to the budget."""
        p = len(self.x)
        cycle = start_cycle
        while cycle <= self.max_iter:
            # Closed form when I - A_m is invertible, deflated episode when it is singular
            IA = np.eye(p) - self.A_m
            if np.linalg.cond(IA) < 1e12:
                switch_cycle = self._closed_form_episode(cycle, IA)
            else:
                switch_cycle = self._deflated_episode(cycle)
            if switch_cycle is None:
                return
            self._perform_switch(switch_cycle)
            cycle = switch_cycle + 1

    def _certified_settle(self, cf: _CF, start_cycle: int) -> bool:
        """KKT test of the episode limit."""
        candidate = cf.x_inf
        scale = 1.0 + max(float(np.abs(self.z).max()), float(np.abs(candidate).max()))

        # The candidate must be feasible; its tight half-spaces carry the multipliers
        slacks = self.unit_A @ candidate - self.unit_b
        if slacks.max() > 1e-9 * scale:
            return False
        tight = np.where(slacks > -1e-7 * scale)[0]

        # The step from the initial point to the candidate must be a non-negative
        # combination of the tight normals
        residual = self.z - candidate
        multipliers = np.zeros(self.n)
        if tight.size == 0:
            if np.abs(residual).max() > 1e-11 * scale:
                return False
        else:
            multipliers_tight, *_ = np.linalg.lstsq(
                self.unit_A[tight].T, residual, rcond=None
            )
            if multipliers_tight.min() < -1e-8 * (1.0 + np.abs(residual).max()):
                return False
            multipliers_tight = np.clip(multipliers_tight, 0.0, None)
            if (np.abs(residual - self.unit_A[tight].T @ multipliers_tight).max()
                    > 1e-11 * scale):
                return False
            multipliers[tight] = multipliers_tight

        active = multipliers > 0.0
        self._set_state(candidate, multipliers, active)
        self.settled = True
        self.settled_at, self.certificate = start_cycle, "kkt"
        for cycle in range(start_cycle, self.max_iter + 1):
            self._record_cycle(cycle, tuple(active))
        return True

    @staticmethod
    def _modal_state(cf: _CF, Q: np.ndarray, V: np.ndarray, lam: np.ndarray,
                     coords: np.ndarray, mu: np.ndarray, t: int) -> tuple:
        """State from the modal form."""
        # Transient and auxiliaries at cycle t from the mode powers lambda_i^t
        lam_t = lam ** t
        z_t = (Q @ (V @ (lam_t * coords))).real
        y_t = cf.G + t * cf.beta - (mu @ lam_t).real
        return cf.x_inf + z_t, y_t

    def _commit_modal_state(self, cf: _CF, Q: np.ndarray, V: np.ndarray, lam: np.ndarray,
                            coords: np.ndarray, mu: np.ndarray, t: int) -> None:
        """Commit the modal state."""
        x_t, y_t = self._modal_state(cf, Q, V, lam, coords, mu, t)
        self._set_state(x_t, y_t, cf.active)

    def _deflated_episode(self, start_cycle: int) -> int | None:
        """Scan a singular episode."""
        p = len(self.x)
        active = np.array(self.active)
        active_idx = np.where(active)[0]
        if active_idx.size == 0:
            return self._step_episode(start_cycle)

        # Orthonormal basis Q of the span of the active normals; P_1 projects onto
        # its complement, the kernel of I - A_m, where the state never moves
        U, sigma, _ = np.linalg.svd(self.unit_A[active_idx].T, full_matrices=False)
        Q = U[:, sigma > 1e-12 * sigma[0]]
        P_1 = np.eye(p) - Q @ Q.T
        IAP = np.eye(p) - self.A_m + P_1
        if np.linalg.cond(IAP) >= 1e12:
            return self._step_episode(start_cycle)

        # Deflated fixed point and the closed-form constants of the episode
        x_inf = P_1 @ self.x + np.linalg.solve(IAP, self.B_m - P_1 @ self.B_m)
        RIA = np.linalg.solve(IAP.T, self.R.T).T
        z_0 = self.x - x_inf
        cf = _CF(A_m=self.A_m, x_inf=x_inf, RIA=RIA,
                 G=self.y + RIA @ z_0, beta=self.R @ x_inf + self.s,
                 beta_floor=self._drift_floor(x_inf),
                 row_RIA=np.linalg.norm(RIA, axis=1), row_R=np.linalg.norm(self.R, axis=1),
                 active=active, inactive=~active)
        if self._certified_settle(cf, start_cycle):
            return None

        # Modes lambda_i of the contracting block T = Q^T A_m Q, and the modal
        # coefficients mu (auxiliaries) and nu (slacks) of the transient
        T = Q.T @ self.A_m @ Q
        lam, V = np.linalg.eig(T)
        if np.max(np.abs(lam)) >= 1.0 - 1e-12 or np.linalg.cond(V) >= self.eig_cond_cap:
            return self._step_episode(start_cycle)
        coords = np.linalg.solve(V, Q.T @ z_0)
        mu = (RIA @ Q) @ V * coords[None, :]
        nu = (self.R @ Q) @ V * coords[None, :]
        abs_lam = np.abs(lam)
        env_y = np.abs(mu)
        env_g = np.abs(nu)

        # Scan in blocks; a half-space cleared by its envelope leaves the watch set
        watch = np.ones(self.n, dtype=bool)
        t = 0
        while start_cycle + t <= self.max_iter:
            # A frozen state is fast-forwarded straight to its switching cycle
            x_t, y_t = self._modal_state(cf, Q, V, lam, coords, mu, t)
            if self._is_stalled(x_t):
                switch_cycle = self._stall_episode(start_cycle + t, x_t, y_t, active)
                if switch_cycle is not None:
                    return switch_cycle

            # Auxiliaries y_m and slacks g_j at every cycle of the block
            block = min(self.block_size, self.max_iter - (start_cycle + t) + 1)
            t_range = np.arange(t + 1, t + block + 1)
            lam_powers = lam[None, :] ** t_range[:, None]
            lam_powers_prev = lam[None, :] ** (t_range[:, None] - 1)
            y_vals = cf.G[None, :] + t_range[:, None] * cf.beta[None, :] - (lam_powers @ mu.T).real
            g_vals = cf.beta[None, :] + (lam_powers_prev @ nu.T).real
            vals = np.where(active[None, :], y_vals, g_vals)
            flips = ((vals > 0.0) != active[None, :]) & watch[None, :]
            flip_rows = np.where(flips.any(axis=1))[0]

            # Stop just before the first cycle whose signs differ from the active set
            if flip_rows.size:
                t_switch = t + 1 + int(flip_rows[0])
                self._commit_modal_state(cf, Q, V, lam, coords, mu, t_switch - 1)
                for cycle in range(start_cycle + t, start_cycle + t_switch - 1):
                    self._record_cycle(cycle, active)
                return start_cycle + t_switch - 1

            t += block
            self._commit_modal_state(cf, Q, V, lam, coords, mu, t)
            for cycle in range(start_cycle + t - block, start_cycle + t):
                self._record_cycle(cycle, active)

            # Clear every half-space whose envelope rules out a sign change at every
            # later cycle; once all are cleared the active set is final
            decay = abs_lam ** t
            clear_inactive = cf.inactive & (cf.beta + env_g @ decay < 0.0)
            clear_active = (cf.active & (cf.beta >= -cf.beta_floor)
                            & (cf.G + (t + 1) * cf.beta - env_y @ (decay * abs_lam) > 0.0))
            watch &= ~(clear_inactive | clear_active)
            if not watch.any():
                self._settle_at_fixed_point(cf, start_cycle + t - 1, t)
                return None
        return None
