"""Numerical pieces of the LTI solvers and the oracle, free of solver state: the exact
Dykstra cycle in scalar auxiliaries, the cycle map of an episode and its closed-form
constants, the envelope horizon, the frozen-stall crossing, and the KKT certificate."""

from collections import namedtuple
import numpy as np

# Rounding allowed, in units of eps times the magnitudes that cancel, before a
# quantity that is zero in exact arithmetic counts as nonzero
_ROUNDING_UNITS = 64

# The closed form works through (I - A_m)^-1, whose rounding grows with its condition
# number; a more ill-conditioned episode is stepped or deflated instead
RESOLVENT_COND_CAP = 1e8

# The contraction rate rho = ||A_m||_2 is kept strictly below one
RHO_CAP = 1.0 - 1e-12

# Rows within this distance of the hint, relative to the magnitudes their slack is
# computed from, are candidates for the active set of the projection
_KKT_CANDIDATE_TOL = 1e-7

# Error certified for the projection, relative to the step from z and the multiplier
# terms; the stationarity residual of a feasible, complementary point bounds it
_KKT_ACCURACY = 1e-9

# The cycle map of an episode: A_m and B_m carry a cycle's start to its end, row m of
# R and s gives the increment R_m x + s_m of auxiliary m over a cycle from x, R_scale
# and s_scale hold the magnitudes R and s are computed from, and span is an
# orthonormal basis of the active normals, the only directions a cycle moves
CycleMap = namedtuple("CycleMap", "active A_m B_m R R_scale s s_scale span")

# Constants of one episode's closed form: its fixed point x_inf, RIA = R (I - A_m)^-1,
# the levels G and drifts beta of the auxiliaries (floors Gamma on inactive rows), the
# rounding level of beta, the row norms of RIA and R, and the activity masks
ClosedForm = namedtuple("ClosedForm", "A_m x_inf RIA G beta beta_floor row_RIA row_R active inactive")


def rounding_floor(magnitude: np.ndarray) -> np.ndarray:
    """Rounding level of a sum."""
    return _ROUNDING_UNITS * np.finfo(float).eps * magnitude


def unit_constraints(A: np.ndarray, b: np.ndarray) -> tuple:
    """Unit normals and their offsets."""
    norms = np.array([np.linalg.norm(row) for row in A]).reshape(-1, 1)
    return A / norms, b / norms[:, 0]


def exact_cycle(x: np.ndarray, y: np.ndarray, unit_A: np.ndarray, unit_b: np.ndarray) -> tuple:
    """One exact Dykstra cycle."""
    # Dykstra's correction for a half-space is always a multiple of its unit normal,
    # e_m = y_m a_m, so one scalar per half-space carries the whole state
    x, y = x.copy(), y.copy()
    points = np.empty((len(y), len(x)))
    for m, (normal, offset) in enumerate(zip(unit_A, unit_b)):
        shifted = x + y[m] * normal
        slack = float(normal @ shifted) - offset
        if slack > 0.0:
            x, y[m] = shifted - slack * normal, slack
        else:
            x, y[m] = shifted, 0.0
        points[m] = x
    return x, y, points


def cycle_map(unit_A: np.ndarray, unit_b: np.ndarray, active: tuple) -> CycleMap:
    """Cycle map of an active set."""
    n, p = unit_A.shape

    # Prefix map carrying the cycle start x_0 to the point entering half-space m
    P = np.eye(p)
    q = np.zeros(p)
    R = np.zeros((n, p))
    R_scale = np.zeros((n, p))
    s = np.zeros(n)
    s_scale = np.zeros(n)
    for m in range(n):
        normal, offset = unit_A[m], unit_b[m]

        # Row m of the auxiliary update y_m += a_m . x_{m-1} - b_m; for a half-space
        # that repeats the last active one before it R_m cancels to rounding, so floors
        # use R_scale
        R[m] = P.T @ normal
        R_scale[m] = np.abs(P).T @ np.abs(normal)
        s[m] = float(normal @ q) - offset
        s_scale[m] = float(np.abs(normal) @ np.abs(q)) + abs(offset)

        # Advance the prefix map through the projector M_m = I - a_m a_m^T
        if active[m]:
            M = np.eye(p) - np.outer(normal, normal)
            P = M @ P
            q = M @ q + offset * normal

    span = np.zeros((p, 0))
    rows = np.where(np.array(active, dtype=bool))[0]
    if rows.size:
        U, sigma, _ = np.linalg.svd(unit_A[rows].T, full_matrices=False)
        span = U[:, sigma > 1e-12 * sigma[0]]
    return CycleMap(active=tuple(active), A_m=P, B_m=q, R=R, R_scale=R_scale, s=s, s_scale=s_scale,
                    span=span)


def advance(cmap: CycleMap, x: np.ndarray) -> np.ndarray:
    """State after one cycle."""
    # Confining the rounded step to the span stops the kernel component, which no
    # cycle moves, from drifting by a rounding error per cycle
    step = cmap.A_m @ x + cmap.B_m - x
    return x + cmap.span @ (cmap.span.T @ step)


def predicted_activity(cmap: CycleMap, x: np.ndarray, y: np.ndarray) -> tuple:
    """Next cycle's activity and auxiliaries."""
    # Increment of every auxiliary over one cycle from x. Activity changes only on a
    # value beyond rounding, since the idle member of an equality written as two
    # half-spaces never shows more than rounding; a row kept active keeps a nonnegative
    # auxiliary
    delta = cmap.R @ x + cmap.s
    y_next = y + delta
    floor = rounding_floor(np.abs(y) + cmap.R_scale @ np.abs(x) + cmap.s_scale)
    active = np.array(cmap.active)
    active_next = np.where(active, y_next > -floor, delta > floor)
    return tuple(active_next), np.where(active_next, np.maximum(y_next, 0.0), 0.0)


def is_frozen(cmap: CycleMap, x: np.ndarray) -> bool:
    """Cycle leaves the state unchanged."""
    # Fixed to rounding, coordinate by coordinate: holding a state that still moves,
    # however slowly, would leave Dykstra's path
    motion = np.abs(cmap.A_m @ x + cmap.B_m - x)
    return bool(np.all(motion <= rounding_floor(np.abs(cmap.A_m) @ np.abs(x) + np.abs(cmap.B_m) + np.abs(x))))


def _closed_form(cmap: CycleMap, x: np.ndarray, y: np.ndarray, x_inf: np.ndarray,
                 RIA: np.ndarray) -> ClosedForm:
    """Constants around a fixed point."""
    # Level G_m of each auxiliary without its transient z_0 = x - x_inf, drift beta_m
    # per cycle, and the rounding level of that drift
    active = np.array(cmap.active)
    beta = cmap.R @ x_inf + cmap.s
    return ClosedForm(A_m=cmap.A_m, x_inf=x_inf, RIA=RIA, G=y + RIA @ (x - x_inf), beta=beta,
                      beta_floor=rounding_floor(cmap.R_scale @ np.abs(x_inf) + cmap.s_scale),
                      row_RIA=np.linalg.norm(RIA, axis=1), row_R=np.linalg.norm(cmap.R, axis=1),
                      active=active, inactive=~active)


def closed_form(cmap: CycleMap, x: np.ndarray, y: np.ndarray, IA: np.ndarray) -> ClosedForm:
    """Closed form of a regular episode."""
    # Fixed point x_inf = (I - A_m)^-1 B_m, by solves rather than an explicit inverse,
    # which loses cond * eps
    return _closed_form(cmap, x, y, np.linalg.solve(IA, cmap.B_m), np.linalg.solve(IA.T, cmap.R.T).T)


def deflated_closed_form(cmap: CycleMap, x: np.ndarray, y: np.ndarray) -> ClosedForm | None:
    """Closed form of a singular episode."""
    # P_1 projects onto the kernel of I - A_m, the complement of the active normals'
    # span, where the state never moves; I - A_m + P_1 is invertible when the span is
    # contracted, and the solved part lies in the span in exact arithmetic, so keeping
    # only that part stops rounding from moving the kernel component
    p = len(x)
    Q = cmap.span
    P_1 = np.eye(p) - Q @ Q.T
    IAP = np.eye(p) - cmap.A_m + P_1
    if np.linalg.cond(IAP) >= RESOLVENT_COND_CAP:
        return None
    x_inf = P_1 @ x + Q @ (Q.T @ np.linalg.solve(IAP, cmap.B_m - P_1 @ cmap.B_m))
    return _closed_form(cmap, x, y, x_inf, np.linalg.solve(IAP.T, cmap.R.T).T)


def active_auxiliaries(cf: ClosedForm, t: int, z_t: np.ndarray) -> np.ndarray:
    """Closed-form auxiliaries at cycle t."""
    # Level plus drift, minus the decaying transient
    return cf.G + t * cf.beta - cf.RIA @ z_t


def closed_form_activity(cf: ClosedForm, cmap: CycleMap, y_t: np.ndarray, z_prev: np.ndarray) -> np.ndarray:
    """Signs of a cycle's auxiliaries and slacks."""
    # Slack g_j = beta_j + R_j z_{t-1}; an inactive half-space reactivates only on a
    # slack above rounding
    g_t = cf.beta + cmap.R @ z_prev
    floor = cf.beta_floor + rounding_floor(cmap.R_scale @ np.abs(z_prev))
    return np.where(cf.active, y_t > 0.0, g_t > floor)


def active_set_is_final(cf: ClosedForm, t: int, z_t: np.ndarray) -> bool:
    """No further activity change."""
    # Every later transient is bounded by the current ||z_t||; a drift is zero only up
    # to rounding, since any real drain reaches zero eventually, and an inactive slack
    # must stay within the rounding its prediction ignores
    z_norm = float(np.linalg.norm(z_t))
    active_ok = not cf.active.any() or (
        np.all(cf.beta[cf.active] >= -cf.beta_floor[cf.active])
        and np.all(cf.G[cf.active] + t * cf.beta[cf.active] - cf.row_RIA[cf.active] * z_norm > 0.0))
    inactive_ok = not cf.inactive.any() or np.all(
        cf.beta[cf.inactive] + cf.row_R[cf.inactive] * z_norm <= cf.beta_floor[cf.inactive])
    return active_ok and inactive_ok


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

    # Bracket the descending root, then bisect (L is concave); past the cap the last
    # bracket point still has L > 0, so the whole run up to it is certified
    hi = 2.0
    while L(hi) > 0.0:
        hi *= 2.0
        if hi > 1e7:
            return hi / 2.0
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


def jump_length(cf: ClosedForm, t: int, z_t: np.ndarray, rho: float) -> int:
    """Switch-free jump length."""
    z_norm = float(np.linalg.norm(z_t))

    # No jump while an inactive slack g_j could reach zero within its envelope
    for j in np.where(cf.inactive)[0]:
        if cf.row_R[j] * z_norm >= -cf.beta[j]:
            return 0

    # Every active y_m stays positive up to the smallest envelope horizon
    k = np.inf
    for m in np.where(cf.active)[0]:
        k = min(k, rigorous_deactivation_horizon(
            cf.G[m] + t * cf.beta[m], cf.beta[m], cf.row_RIA[m] * z_norm, rho))
    if not np.isfinite(k):
        return 0

    # One-cycle safety margin
    return max(0, int(np.floor(k)) - 1)


def stall_crossing(y: np.ndarray, delta: np.ndarray, delta_floor: np.ndarray,
                   active: np.ndarray, limit: int) -> float:
    """Cycles until the first crossing."""
    crossing = np.inf

    # A draining active auxiliary reaches zero after ceil(y_m / -delta_m) cycles
    for m in np.where(active)[0]:
        if delta[m] < -delta_floor[m] and y[m] > 0.0:
            # A crossing past the limit is never taken, and above 2**53 refining it one
            # cycle at a time takes about ratio / 2**53 steps, which never finishes at
            # ratios like 1e30
            ratio = y[m] / (-delta[m])
            if not ratio <= limit:
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
            crossing = min(crossing, 1)
    return crossing


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


def nonnegative_least_squares(C: np.ndarray, d: np.ndarray) -> np.ndarray:
    """min ||C l - d|| over l >= 0."""
    # Each support is solved on the columns of C, not the normal equations, whose
    # conditioning is squared
    eps = np.finfo(float).eps
    floor = eps * max(C.shape) * np.abs(C).max(initial=0.0) * np.abs(d).max(initial=0.0)
    return _lawson_hanson(C.shape[1], lambda lam: C.T @ (d - C @ lam) - floor,
                          lambda P: np.linalg.lstsq(C[:, P], d, rcond=None)[0])


def nonnegative_quadratic(H: np.ndarray, c: np.ndarray) -> np.ndarray:
    """min 1/2 l^T H l - c^T l over l >= 0, H positive semidefinite."""
    eps = np.finfo(float).eps
    floor = eps * len(c) * (np.abs(H).max(initial=0.0) + np.abs(c).max(initial=0.0))
    return _lawson_hanson(len(c), lambda lam: c - H @ lam - floor,
                          lambda P: np.linalg.lstsq(H[np.ix_(P, P)], c[P], rcond=None)[0])


def _kkt_slack_scale(z: np.ndarray, unit_A: np.ndarray, unit_b: np.ndarray, x: np.ndarray) -> np.ndarray:
    """Magnitudes each slack is computed from."""
    # A point reached from z rounds with |z| and the step, not with |x|, which is tiny
    # at an apex through the origin; rows ignore coordinates they do not touch
    return np.abs(unit_A) @ (np.abs(z) + np.abs(z - x)) + np.abs(unit_b)


def kkt_certificate(z: np.ndarray, unit_A: np.ndarray, unit_b: np.ndarray,
                    hint: np.ndarray, active: np.ndarray) -> tuple | None:
    """Projection and multipliers, if proven."""
    # Candidate rows: the active set and every row near the hint
    active = np.asarray(active, dtype=bool)
    slacks = unit_A @ hint - unit_b
    near = slacks >= -_KKT_CANDIDATE_TOL * _kkt_slack_scale(z, unit_A, unit_b, hint)
    rows = np.where(active | near)[0]

    # Candidate supports: the rows that bind at the projection onto the polyhedron of
    # the candidate rows alone, by its dual (exact) and by the cone at a common point
    # of their boundaries (accurate when one exists), then the active set and all
    # candidate rows as they stand
    supports = []
    if rows.size:
        N = unit_A[rows]
        dual = nonnegative_quadratic(N @ N.T, N @ z - unit_b[rows])
        supports.append(rows[dual > 0.0])
        apex = np.linalg.lstsq(N, unit_b[rows], rcond=None)[0]
        cone = nonnegative_least_squares(N.T, z - apex)
        supports.append(rows[cone > 0.0])
    supports.append(np.where(active)[0])
    supports.append(rows)

    tried = set()
    for support in supports:
        if tuple(support) in tried:
            continue
        tried.add(tuple(support))
        certified = _kkt_verify(z, unit_A, unit_b, support)
        if certified is not None:
            return certified
    return None


def _kkt_verify(z: np.ndarray, unit_A: np.ndarray, unit_b: np.ndarray, support: np.ndarray) -> tuple | None:
    """KKT test at the projection onto a support."""
    # The projection of z onto the boundaries of the support, by least squares, which
    # is backward stable however nearly parallel the rows are
    x = z.copy()
    if support.size:
        x -= np.linalg.lstsq(unit_A[support], unit_A[support] @ z - unit_b[support], rcond=None)[0]

    # Feasible, and multipliers nonnegative on the tight rows only, both at rounding
    # level, so complementarity holds to rounding
    slacks = unit_A @ x - unit_b
    floor = rounding_floor(_kkt_slack_scale(z, unit_A, unit_b, x))
    if np.any(slacks > floor):
        return None
    tight = np.where(slacks >= -floor)[0]
    step = z - x
    multipliers = np.zeros(len(unit_b))
    if tight.size:
        multipliers[tight] = nonnegative_least_squares(unit_A[tight].T, step)

    # x is then the projection of z - r, so the residual r bounds its error
    residual = step - unit_A.T @ multipliers
    allowed = (_KKT_ACCURACY * (np.abs(step) + np.abs(unit_A).T @ multipliers)
               + rounding_floor(np.abs(z) + np.abs(x)))
    if np.any(np.abs(residual) > allowed):
        return None
    return x, multipliers
