"""LTI Ver3: within an episode, jumps whole runs of cycles at once whenever a
rigorous envelope bound shows that no half-space can change activity before the
landing cycle, and verifies the landing state before keeping it."""

import numpy as np
from lti_ver2 import LTIVer2Solver, _CF

# The contraction rate rho = ||A_m||_2 is kept strictly below one
_RHO_CAP = 1.0 - 1e-12


def rigorous_deactivation_horizon(G: float, beta: float, alpha: float, rho: float) -> float:
    """Horizon of the envelope bound."""
    def L(k: float) -> float:
        """Envelope lower bound."""
        return G + beta * k - alpha * rho ** k

    if L(1.0) <= 0.0:
        return 0.0

    # Without a negative drift the bound never descends
    if beta >= 0.0:
        return np.inf

    # Bracket the descending root, then bisect (L is concave)
    hi = 2.0
    while L(hi) > 0.0:
        hi *= 2.0
        if hi > 1e7:
            return np.inf
    lo = hi / 2.0
    for _ in range(200):
        if hi - lo <= 1e-12:
            break
        mid = 0.5 * (lo + hi)
        if mid <= lo or mid >= hi:
            break
        if L(mid) > 0.0:
            lo = mid
        else:
            hi = mid
    return float(lo)


class LTIVer3Solver(LTIVer2Solver):
    """Envelope-bounded episode jumps."""

    def _setup_closed_form(self, IA_inv: np.ndarray) -> _CF:
        """Constants and spectral norm."""
        cf = super()._setup_closed_form(IA_inv)
        self.rho = min(float(np.linalg.svd(cf.A_m, compute_uv=False)[0]), _RHO_CAP)
        return cf

    def _jump_length(self, cf: _CF, t: int, z_t: np.ndarray) -> int:
        """Switch-free jump length."""
        if self.rho <= 0.0:
            return 0
        z_norm = float(np.linalg.norm(z_t))

        # No jump while an inactive slack g_j could reach zero within its envelope
        for j in np.where(~cf.active)[0]:
            if cf.row_R[j] * z_norm >= -cf.beta[j]:
                return 0

        # Every active y_m stays positive up to the smallest envelope horizon
        k = np.inf
        for m in np.where(cf.active)[0]:
            k = min(k, rigorous_deactivation_horizon(
                cf.G[m] + t * cf.beta[m], cf.beta[m], cf.row_RIA[m] * z_norm, self.rho))
        if not np.isfinite(k):
            return 0

        # One-cycle safety margin
        return max(0, int(np.floor(k)) - 1)

    def _jump(self, cf: _CF, t: int, z_t: np.ndarray, cycle: int) -> tuple:
        """Jump and verify the landing."""
        budget = self.max_iter - cycle
        if budget < 1:
            return 0, z_t
        k = min(self._jump_length(cf, t, z_t), budget)
        if k < 1:
            return 0, z_t

        # Land at z_{t+k} = A_m^k z_t; halve the jump whenever the active set did not hold
        while k >= 1:
            z_before = np.linalg.matrix_power(cf.A_m, k - 1) @ z_t
            z_after = cf.A_m @ z_before
            y_after = self._active_auxiliaries(cf, t + k, z_after)
            g_after = cf.beta + self.R @ z_before
            if np.array_equal(np.where(cf.active, y_after, g_after) > 0.0, cf.active):
                self._set_state(cf.x_inf + z_after, y_after, cf.active)
                for later_cycle in range(cycle + 1, cycle + k + 1):
                    self._record_cycle(later_cycle, cf.active)
                return k, z_after
            k //= 2
        return 0, z_t

    def _closed_form_episode(self, start_cycle: int, IA_inv: np.ndarray) -> int | None:
        """Scan an episode with jumps."""
        cf = self._setup_closed_form(IA_inv)
        z_prev = self.x - cf.x_inf
        t = 1
        while start_cycle + t - 1 <= self.max_iter:
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

            # Jump the cycles the envelope certifies switch-free
            k, z_t = self._jump(cf, t, z_t, cycle)
            t += k + 1
            z_prev = z_t
        return None
