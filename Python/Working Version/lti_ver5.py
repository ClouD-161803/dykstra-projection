"""LTI Ver5: runs episodes whose active normals do not span the space in the
coordinates of that span, scanning the closed-form auxiliaries and slacks exactly
in blocks, settling by a KKT test on the episode limit, and jumping frozen
stalls to their crossing."""

import numpy as np
from lti_ver2 import _CF, _RESOLVENT_COND_CAP
from lti_ver4 import LTIVer4Solver

# Rows within this distance of the hint, relative to the magnitudes their slack is
# computed from, are candidates for the active set of the projection
_KKT_CANDIDATE_TOL = 1e-7

# Error certified for the projection, relative to the step from z and the multiplier
# terms; the stationarity residual of a feasible, complementary point bounds it
_KKT_ACCURACY = 1e-9


def _lawson_hanson(k: int, gradient, solve) -> np.ndarray:
    """Lawson and Hanson's active-set method over l >= 0."""
    # gradient(l) is minus the gradient of the objective, solve(P) the unconstrained
    # minimiser on support P; both iteration caps only guard against rounding cycles
    lam = np.zeros(k)
    passive = np.zeros(k, dtype=bool)
    for _ in range(3 * k + 10):
        w = gradient(lam)
        if passive.all() or w[~passive].max() <= 0.0:
            break
        passive[np.argmax(np.where(passive, -np.inf, w))] = True
        for _ in range(3 * k + 10):
            support = np.where(passive)[0]
            step = np.zeros(k)
            step[support] = solve(support)
            if np.all(step[support] > 0.0):
                lam = step
                break
            # Move towards the new solution until the first multiplier reaches zero
            blocked = support[step[support] <= 0.0]
            ratios = lam[blocked] / (lam[blocked] - step[blocked])
            first = int(np.argmin(ratios))
            lam = lam + ratios[first] * (step - lam)
            lam[blocked[first]] = 0.0
            passive &= lam > 0.0
    return lam


def _nonnegative_least_squares(C: np.ndarray, d: np.ndarray) -> np.ndarray:
    """min ||C l - d|| over l >= 0."""
    # Each support is solved on the columns of C, not the normal equations, whose
    # conditioning is squared
    eps = np.finfo(float).eps
    floor = eps * max(C.shape) * np.abs(C).max(initial=0.0) * np.abs(d).max(initial=0.0)
    return _lawson_hanson(C.shape[1], lambda lam: C.T @ (d - C @ lam) - floor,
                          lambda P: np.linalg.lstsq(C[:, P], d, rcond=None)[0])


def _nonnegative_quadratic(H: np.ndarray, c: np.ndarray) -> np.ndarray:
    """min 1/2 l^T H l - c^T l over l >= 0, H positive semidefinite."""
    eps = np.finfo(float).eps
    floor = eps * len(c) * (np.abs(H).max(initial=0.0) + np.abs(c).max(initial=0.0))
    return _lawson_hanson(len(c), lambda lam: c - H @ lam - floor,
                          lambda P: np.linalg.lstsq(H[np.ix_(P, P)], c[P], rcond=None)[0])


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

    @property
    def settled(self) -> bool:
        """Result proven to be the projection."""
        # Every Ver5 settlement passes the KKT test
        return self.settled_at is not None

    def _settle_at_fixed_point(self, cf: _CF, cycle: int, t: int) -> None:
        """Settle on a certified limit."""
        # The finality test proves the active set final, but the fixed point of the
        # rounded cycle map can still miss the projection, so the KKT test decides
        certified = self._kkt_certificate(cf.x_inf, cf.active)
        if certified is not None:
            self._settle_certified(*certified, cycle)
            return

        # No switch can follow, so the budget's iterate is one closed-form jump away
        k = self.max_iter - cycle
        if k < 1:
            return
        z_end = np.linalg.matrix_power(cf.A_m, k) @ (self.x - cf.x_inf)
        self._set_state(cf.x_inf + z_end, self._active_auxiliaries(cf, t + k, z_end), cf.active)
        for later_cycle in range(cycle + 1, self.max_iter + 1):
            self._record_cycle(later_cycle, cf.active)

    def _accelerate(self, start_cycle: int) -> None:
        """Run episodes to the budget."""
        p = len(self.x)
        cycle = start_cycle
        while cycle <= self.max_iter:
            # A point that passes the KKT test is the projection whatever the episode
            # would do next, and finding it needs no resolvent
            certified = self._kkt_certificate(self.x, np.array(self.active))
            if certified is not None:
                self._settle_certified(*certified, cycle)
                return

            # Closed form when I - A_m is invertible, deflated episode when it is singular
            IA = np.eye(p) - self.A_m
            if np.linalg.cond(IA) < _RESOLVENT_COND_CAP:
                switch_cycle = self._closed_form_episode(cycle, IA)
            else:
                switch_cycle = self._deflated_episode(cycle)
            if switch_cycle is None or switch_cycle > self.max_iter:
                return
            self._perform_switch(switch_cycle)
            cycle = switch_cycle + 1

    def _certified_settle(self, cf: _CF, start_cycle: int) -> bool:
        """KKT test of the episode limit."""
        certified = self._kkt_certificate(cf.x_inf, cf.active)
        if certified is None:
            return False
        self._settle_certified(*certified, start_cycle)
        return True

    def _settle_certified(self, x: np.ndarray, multipliers: np.ndarray, first_cycle: int) -> None:
        """Settle on the projection."""
        # The multipliers are the auxiliaries of the limit: x + sum y_m a_m = z
        active = multipliers > 0.0
        self._set_state(x, multipliers, active)
        self.settled_at, self.certificate = first_cycle, "kkt"
        self._record_limit(first_cycle, active)

    def _kkt_slack_scale(self, x: np.ndarray) -> np.ndarray:
        """Magnitudes each slack is computed from."""
        # A point reached from z rounds with |z| and the step, not with |x|, which is
        # tiny at an apex through the origin; rows ignore coordinates they do not touch
        return np.abs(self.unit_A) @ (np.abs(self.z) + np.abs(self.z - x)) + np.abs(self.unit_b)

    def _kkt_certificate(self, hint: np.ndarray, active: np.ndarray) -> tuple | None:
        """Projection and multipliers, if proven."""
        N, b, z = self.unit_A, self.unit_b, self.z

        # Candidate rows: the active set and every row near the hint
        slacks = N @ hint - b
        near = slacks >= -_KKT_CANDIDATE_TOL * self._kkt_slack_scale(hint)
        rows = np.where(np.asarray(active, dtype=bool) | near)[0]

        # Candidate supports: the rows that bind at the projection onto the polyhedron
        # of the candidate rows alone, by its dual (exact) and by the cone at a common
        # point of their boundaries (accurate when one exists), then the active set and
        # all candidate rows as they stand
        supports = []
        if rows.size:
            N_rows = N[rows]
            dual = _nonnegative_quadratic(N_rows @ N_rows.T, N_rows @ z - b[rows])
            supports.append(rows[dual > 0.0])
            apex = np.linalg.lstsq(N_rows, b[rows], rcond=None)[0]
            cone = _nonnegative_least_squares(N_rows.T, z - apex)
            supports.append(rows[cone > 0.0])
        supports.append(np.where(np.asarray(active, dtype=bool))[0])
        supports.append(rows)

        tried = set()
        for support in supports:
            if tuple(support) in tried:
                continue
            tried.add(tuple(support))
            certified = self._kkt_verify(support)
            if certified is not None:
                return certified
        return None

    def _kkt_verify(self, support: np.ndarray) -> tuple | None:
        """KKT test at the projection onto a support."""
        N, b, z = self.unit_A, self.unit_b, self.z

        # The projection of z onto the boundaries of the support, by least squares,
        # which is backward stable however nearly parallel the rows are
        x = z.copy()
        if support.size:
            x -= np.linalg.lstsq(N[support], N[support] @ z - b[support], rcond=None)[0]

        # Feasible, and multipliers nonnegative on the tight rows only, both at
        # rounding level, so complementarity holds to rounding
        slacks = N @ x - b
        floor = self._rounding_floor(self._kkt_slack_scale(x))
        if np.any(slacks > floor):
            return None
        tight = np.where(slacks >= -floor)[0]
        step = z - x
        multipliers = np.zeros(self.n)
        if tight.size:
            multipliers[tight] = _nonnegative_least_squares(N[tight].T, step)

        # x is then the projection of z - r, so the residual r bounds its error
        residual = step - N.T @ multipliers
        allowed = (_KKT_ACCURACY * (np.abs(step) + np.abs(N).T @ multipliers)
                   + self._rounding_floor(np.abs(z) + np.abs(x)))
        if np.any(np.abs(residual) > allowed):
            return None
        return x, multipliers

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
        if np.linalg.cond(IAP) >= _RESOLVENT_COND_CAP:
            return self._step_episode(start_cycle)

        # Deflated fixed point and the closed-form constants of the episode; the
        # solved part lies in the span in exact arithmetic, and keeping only that part
        # stops rounding from moving the kernel component, which no cycle can undo
        x_inf = P_1 @ self.x + Q @ (Q.T @ np.linalg.solve(IAP, self.B_m - P_1 @ self.B_m))
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
                return self._stall_episode(start_cycle + t, x_t, y_t, active)

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

            # Stop just before the first cycle whose signs differ from the active set;
            # the state at cycle t is already committed, and rebuilding it from the
            # modes would move it by about eps times the condition of V
            if flip_rows.size:
                t_switch = t + 1 + int(flip_rows[0])
                if t_switch - 1 > t:
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
