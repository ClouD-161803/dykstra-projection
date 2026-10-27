"""Numerical pieces of the LTI solvers and the oracle, free of solver state: the exact
Dykstra cycle in scalar auxiliaries, the cycle map of an episode and its closed-form
constants, the envelope horizon, the frozen-stall crossing, the KKT certificate, and the
work of each piece in flops."""

from collections import namedtuple
from functools import wraps
from itertools import combinations
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

# Candidate rows up to this many also have every subset tried as a support, since
# nearly dependent rows leave the choice among them to rounding
_KKT_EXHAUSTIVE_ROWS = 6

# Error certified for the projection, relative to the step from z; the stationarity
# residual of a feasible, complementary point bounds it
_KKT_ACCURACY = 1e-9

# The cycle map of an episode: A_m and B_m carry a cycle's start to its end, row m of
# R and s gives the increment R_m x + s_m of auxiliary m over a cycle from x, R_scale
# and s_scale hold the magnitudes R and s are computed from, span is an orthonormal
# basis of the active normals, the only directions a cycle moves, touched marks the
# coordinates some constraint touches, and coupled those an active normal touches,
# which a cycle mixes into one another
CycleMap = namedtuple("CycleMap", "active A_m B_m R R_scale s s_scale span touched coupled")

# Constants of one episode's closed form: its fixed point x_inf, RIA = R (I - A_m)^-1
# and the magnitudes RIA_scale its rounding comes from, the levels G and drifts beta
# of the auxiliaries (floors Gamma on inactive rows), the rounding levels of G and
# beta, the row norms of RIA and of R beyond its rounding, and the activity masks
ClosedForm = namedtuple("ClosedForm", "A_m x_inf RIA RIA_scale G G_floor beta beta_floor row_RIA "
                                      "row_R active inactive")


def rounding_floor(magnitude: np.ndarray) -> np.ndarray:
    """Rounding level of a sum."""
    return _ROUNDING_UNITS * np.finfo(float).eps * magnitude


def ignore_blas_flags(function):
    """Runs function with numpy ignoring the floating-point flags its products raise."""
    # numpy 2.1 linked to Apple's Accelerate reports the flags that Accelerate leaves set
    # after products of finite matrices, from about size 15 up, as divide-by-zero,
    # overflow and invalid-value warnings in matmul. The flags therefore say nothing
    # about the values, and an infinity or a NaN would still show in the results. The
    # solvers' and the oracle's entry points carry this, as do the numerics called alone
    @wraps(function)
    def quiet(*args, **kwargs):
        with np.errstate(divide="ignore", over="ignore", invalid="ignore"):
            return function(*args, **kwargs)
    return quiet


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
    # The magnitudes of P change only at an active half-space
    abs_P = np.abs(P)
    for m in range(n):
        normal, offset = unit_A[m], unit_b[m]

        # Row m of the auxiliary update y_m += a_m . x_{m-1} - b_m; for a half-space
        # that repeats the last active one before it R_m cancels to rounding, so floors
        # use R_scale
        R[m] = P.T @ normal
        R_scale[m] = abs_P.T @ np.abs(normal)
        s[m] = float(normal @ q) - offset
        s_scale[m] = float(np.abs(normal) @ np.abs(q)) + abs(offset)

        # Advance the prefix map through the projector M_m = I - a_m a_m^T
        if active[m]:
            M = np.eye(p) - np.outer(normal, normal)
            P = M @ P
            q = M @ q + offset * normal
            abs_P = np.abs(P)

    # A coordinate no constraint touches is not part of the problem: the SVD can leave
    # rounding in its row of the span, which advance would add to it every cycle
    touched = np.any(unit_A != 0.0, axis=0)
    span = np.zeros((p, 0))
    rows = np.where(np.array(active, dtype=bool))[0]
    if rows.size:
        U, sigma, _ = np.linalg.svd(unit_A[rows].T, full_matrices=False)
        span = U[:, sigma > 1e-12 * sigma[0]]
        span[~touched] = 0.0
    return CycleMap(active=tuple(active), A_m=P, B_m=q, R=R, R_scale=R_scale, s=s, s_scale=s_scale,
                    span=span, touched=touched, coupled=np.any(unit_A[rows] != 0.0, axis=0))


def touched_resolvent(cmap: CycleMap) -> np.ndarray:
    """I - A_m on the touched coordinates."""
    # A_m is the identity on a coordinate no constraint touches, so the full I - A_m is
    # singular whenever one exists, and that coordinate would decide whether an episode
    # counts as regular
    touched = cmap.touched
    return np.eye(int(touched.sum())) - cmap.A_m[np.ix_(touched, touched)]


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
    # however slowly, would leave Dykstra's path. A cycle mixes the coordinates its active
    # normals touch, so its rounding there is on the scale of the largest of them, and a small
    # coordinate beside larger ones turned into it is charged with theirs. A coordinate no
    # active normal touches does not move at all
    abs_x = np.abs(x)
    motion = np.abs(cmap.A_m @ x + cmap.B_m - x)
    magnitude = np.abs(cmap.A_m) @ abs_x + np.abs(cmap.B_m) + abs_x
    if cmap.coupled.any():
        magnitude[cmap.coupled] = magnitude[cmap.coupled].max()
    return bool(np.all(motion <= rounding_floor(magnitude)))


def _closed_form(cmap: CycleMap, x: np.ndarray, y: np.ndarray, x_inf: np.ndarray,
                 RIA: np.ndarray, resolvent: np.ndarray) -> ClosedForm:
    """Constants around a fixed point."""
    # Level G_m of each auxiliary without its transient z_0 = x - x_inf, drift beta_m
    # per cycle, and the rounding level of each. A row of R that cancelled to the
    # rounding of the magnitudes it was formed from is noise, and so is its row of RIA:
    # only such a row takes its rounding from those magnitudes through the resolvent,
    # since charging every row with the resolvent's condition would hide real drains,
    # and only what exceeds that rounding counts as a slack's envelope
    active = np.array(cmap.active)
    z_0 = x - x_inf
    R_norm = np.linalg.norm(cmap.R, axis=1)
    R_rounding = rounding_floor(np.linalg.norm(cmap.R_scale, axis=1))
    cancelled = R_norm <= R_rounding
    RIA_scale = np.abs(RIA)
    if cancelled.any():
        # The resolvent acts on the touched coordinates only
        rows = np.ix_(cancelled, cmap.touched)
        RIA_scale[rows] = cmap.R_scale[rows] @ np.abs(np.linalg.inv(resolvent))
    return ClosedForm(A_m=cmap.A_m, x_inf=x_inf, RIA=RIA, RIA_scale=RIA_scale, G=y + RIA @ z_0,
                      G_floor=rounding_floor(np.abs(y) + RIA_scale @ np.abs(z_0)),
                      beta=cmap.R @ x_inf + cmap.s,
                      beta_floor=rounding_floor(cmap.R_scale @ np.abs(x_inf) + cmap.s_scale),
                      row_RIA=np.linalg.norm(RIA, axis=1), row_R=np.maximum(R_norm - R_rounding, 0.0),
                      active=active, inactive=~active)


def closed_form(cmap: CycleMap, x: np.ndarray, y: np.ndarray, IA: np.ndarray) -> ClosedForm:
    """Closed form of a regular episode, with IA from touched_resolvent."""
    # Fixed point x_inf = (I - A_m)^-1 B_m on the touched coordinates, by solves rather
    # than an explicit inverse, which loses cond * eps. An untouched coordinate never
    # moves, so x_inf keeps its value, and no auxiliary reads it
    touched = cmap.touched
    x_inf = x.copy()
    x_inf[touched] = np.linalg.solve(IA, cmap.B_m[touched])
    RIA = np.zeros_like(cmap.R)
    RIA[:, touched] = np.linalg.solve(IA.T, cmap.R[:, touched].T).T
    return _closed_form(cmap, x, y, x_inf, RIA, IA)


def deflated_closed_form(cmap: CycleMap, x: np.ndarray, y: np.ndarray) -> ClosedForm | None:
    """Closed form of a singular episode."""
    # P_1 projects onto the kernel of I - A_m, the complement of the active normals'
    # span, where the state never moves; I - A_m + P_1 is invertible when the span is
    # contracted, and the solved part lies in the span in exact arithmetic, so keeping
    # only that part stops rounding from moving the kernel component. All of it is
    # taken on the touched coordinates, as in touched_resolvent, so that an untouched
    # one cannot change the condition number tested against the cutoff
    touched = cmap.touched
    Q = cmap.span[touched]
    P_1 = np.eye(len(Q)) - Q @ Q.T
    IAP = touched_resolvent(cmap) + P_1
    if np.linalg.cond(IAP) >= RESOLVENT_COND_CAP:
        return None
    B_m = cmap.B_m[touched]
    x_inf = x.copy()
    x_inf[touched] = P_1 @ x[touched] + Q @ (Q.T @ np.linalg.solve(IAP, B_m - P_1 @ B_m))
    RIA = np.zeros_like(cmap.R)
    RIA[:, touched] = np.linalg.solve(IAP.T, cmap.R[:, touched].T).T
    return _closed_form(cmap, x, y, x_inf, RIA, IAP)


def active_auxiliaries(cf: ClosedForm, t: int, z_t: np.ndarray) -> np.ndarray:
    """Closed-form auxiliaries at cycle t."""
    # Level plus drift, minus the decaying transient
    return cf.G + t * cf.beta - cf.RIA @ z_t


def auxiliary_floor(cf: ClosedForm, t) -> np.ndarray:
    """Rounding of the auxiliaries extrapolated to cycle t."""
    # A drift within rounding may be rounding of either sign, which extrapolation
    # multiplies by t; a drift beyond it is real and decides on its own
    return cf.G_floor + t * np.where(np.abs(cf.beta) <= cf.beta_floor, cf.beta_floor, 0.0)


def closed_form_activity(cf: ClosedForm, cmap: CycleMap, t: int, y_t: np.ndarray,
                         z_t: np.ndarray, z_prev: np.ndarray) -> np.ndarray:
    """Activity of cycle t."""
    # Activity changes only on a value beyond rounding: an inactive half-space
    # reactivates on a slack g_j = beta_j + R_j z_{t-1} above it, and an active one
    # deactivates on an auxiliary below minus the rounding its extrapolation carries,
    # which is all the idle member of an equality written as two half-spaces shows
    g_t = cf.beta + cmap.R @ z_prev
    g_floor = cf.beta_floor + rounding_floor(cmap.R_scale @ np.abs(z_prev))
    y_floor = auxiliary_floor(cf, t) + rounding_floor(cf.RIA_scale @ np.abs(z_t))
    return np.where(cf.active, y_t > -y_floor, g_t > g_floor)


def active_set_is_final(cf: ClosedForm, t: int, z_t: np.ndarray) -> bool:
    """No further activity change."""
    # Every later transient is bounded by the current ||z_t||; a drift is zero only up
    # to rounding, since any real drain reaches zero eventually, an inactive slack must
    # stay within the rounding its prediction ignores, and an active auxiliary need only
    # stay above minus the rounding of its extrapolation, as in closed_form_activity
    z_norm = float(np.linalg.norm(z_t))
    active_ok = not cf.active.any() or (
        np.all(cf.beta[cf.active] >= -cf.beta_floor[cf.active])
        and np.all(cf.G[cf.active] + t * cf.beta[cf.active] - cf.row_RIA[cf.active] * z_norm
                   > -auxiliary_floor(cf, t)[cf.active]))
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

    # No jump while an inactive slack g_j could rise above the rounding its
    # prediction ignores within its envelope
    for j in np.where(cf.inactive)[0]:
        if cf.beta[j] + cf.row_R[j] * z_norm > cf.beta_floor[j]:
            return 0

    # Every active y_m stays above minus the rounding its extrapolation carries, as in
    # closed_form_activity, up to the smallest envelope horizon; a drift within rounding
    # is no drain, as in the finality test, whatever its sign, and the level keeps the
    # drift itself so that the floor, which grows at least as fast, covers it
    floor = auxiliary_floor(cf, t)
    k = np.inf
    for m in np.where(cf.active)[0]:
        beta = 0.0 if abs(cf.beta[m]) <= cf.beta_floor[m] else cf.beta[m]
        k = min(k, rigorous_deactivation_horizon(cf.G[m] + floor[m] + t * cf.beta[m], beta,
                                                 cf.row_RIA[m] * z_norm, rho))
    if not np.isfinite(k):
        return 0

    # One-cycle safety margin
    return max(0, int(np.floor(k)) - 1)


def stall_crossing(y: np.ndarray, delta: np.ndarray, delta_floor: np.ndarray,
                   active: np.ndarray, limit: int) -> float:
    """Cycles until the first crossing."""
    crossing = np.inf

    # A draining active auxiliary reaches zero after ceil(y_m / -delta_m) cycles, and
    # one already at zero on the next
    for m in np.where(active)[0]:
        if delta[m] < -delta_floor[m] and y[m] <= 0.0:
            crossing = min(crossing, 1)
        elif delta[m] < -delta_floor[m]:
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
    # gradient(l) is minus the gradient of the objective. solve(P) returns the
    # unconstrained minimiser on support P and, when the objective has none there but
    # falls without bound, the direction it falls along. Both iteration caps only guard
    # against rounding cycles
    lam = np.zeros(k)
    passive = np.zeros(k, dtype=bool)
    for _ in range(3 * k + 10):
        w = gradient(lam)
        if passive.all() or w[~passive].max() <= 0.0:
            break
        entering = int(np.argmax(np.where(passive, -np.inf, w)))
        passive[entering] = True
        for _ in range(3 * k + 10):
            support = np.where(passive)[0]
            minimiser, ray = solve(support)
            if ray is not None:
                # The multipliers move along the ray until the first one it lowers
                # reaches zero, and that row leaves the support. A ray that does not
                # raise the entering row would leave it at zero, so the minimiser is
                # stepped towards instead
                direction = np.zeros(k)
                direction[support] = ray
                lowered = support[direction[support] < 0.0]
                if direction[entering] > 0.0 and lowered.size:
                    ratios = lam[lowered] / -direction[lowered]
                    first = int(np.argmin(ratios))
                    lam = lam + ratios[first] * direction
                    lam[lowered[first]] = 0.0
                    passive &= lam > 0.0
                    continue
            step = np.zeros(k)
            step[support] = minimiser
            if np.all(step[support] > 0.0):
                lam = step
                break
            # Move towards the new solution until the first multiplier reaches zero. A
            # row entering at zero whose solve also gives it zero does not move, and
            # its ratio 0 / 0 would turn every multiplier into NaN
            blocked = support[step[support] <= 0.0]
            gaps = lam[blocked] - step[blocked]
            ratios = np.divide(lam[blocked], gaps, out=np.zeros(len(blocked)), where=gaps != 0.0)
            first = int(np.argmin(ratios))
            lam = lam + ratios[first] * (step - lam)
            lam[blocked[first]] = 0.0
            passive &= lam > 0.0
    return lam


@ignore_blas_flags
def nonnegative_least_squares(C: np.ndarray, d: np.ndarray, tally=None) -> np.ndarray:
    """min ||C l - d|| over l >= 0, adding its flops to tally if a KKTTally is given."""
    # Each support is solved on the columns of C, not the normal equations, whose
    # conditioning is squared. A gradient entry counts as positive only above the
    # rounding of the terms summed into it, so a row of C that is zero, a coordinate no
    # normal touches, raises no entry's threshold
    abs_C = np.abs(C)
    gradient = lambda lam: C.T @ (d - C @ lam) - rounding_floor(abs_C.T @ (np.abs(d) + abs_C @ lam))
    solve = lambda P: (np.linalg.lstsq(C[:, P], d, rcond=None)[0], None)
    if tally is not None:
        gradient, solve = tally.counting(gradient, 8 * C.size, solve, lambda P: (C.shape[0], len(P)))
    return _lawson_hanson(C.shape[1], gradient, solve)


@ignore_blas_flags
def nonnegative_quadratic(H: np.ndarray, c: np.ndarray, tally=None) -> np.ndarray:
    """min 1/2 l^T H l - c^T l over l >= 0, H positive semidefinite, adding its flops to tally
    if a KKTTally is given."""
    abs_H = np.abs(H)

    def solve(P: np.ndarray) -> tuple:
        """Minimiser on support P, and the direction of unbounded descent if any."""
        # Rows of a Gram matrix H = N N^T are dependent whenever there are more of them
        # than dimensions. The minimum-norm solution of a singular H_P leaves the part of
        # c_P outside its range as a residual beyond rounding, and the objective falls
        # without bound along that residual, as H_P maps it to zero. Only the rank least
        # squares reports tells a singular H_P apart: an ill-conditioned one of full rank
        # can leave a residual above a componentwise floor that is still rounding
        H_P, c_P = H[np.ix_(P, P)], c[P]
        minimiser, _, rank, _ = np.linalg.lstsq(H_P, c_P, rcond=None)
        if rank < len(P):
            residual = c_P - H_P @ minimiser
            if np.any(np.abs(residual) > rounding_floor(np.abs(c_P) + np.abs(H_P) @ np.abs(minimiser))):
                return minimiser, residual
        return minimiser, None

    gradient = lambda lam: c - H @ lam - rounding_floor(np.abs(c) + abs_H @ lam)
    if tally is not None:
        gradient, solve = tally.counting(gradient, 4 * H.size, solve, lambda P: (len(P), len(P)))
    return _lawson_hanson(len(c), gradient, solve)


def _kkt_slack_scale(z: np.ndarray, unit_A: np.ndarray, unit_b: np.ndarray, x: np.ndarray) -> np.ndarray:
    """Magnitudes each slack is computed from."""
    # A point reached from z rounds with |z| and the step, not with |x|, which is tiny
    # at an apex through the origin; rows ignore coordinates they do not touch
    return np.abs(unit_A) @ (np.abs(z) + np.abs(z - x)) + np.abs(unit_b)


class KKTMemo:
    """Work of the KKT tests already run in one solve."""

    # A candidate support follows from the candidate rows alone, and a verdict from the
    # support alone, once z and the constraints are fixed, as they are through a solve. A
    # support that failed fails again, so reusing either changes neither the outcome nor
    # the point. costs holds the flops each entry of derived and rejected took, when a
    # KKTTally counted them, so that a later test can count the work it reuses
    def __init__(self) -> None:
        self.derived = {}
        self.rejected = set()
        self.costs = {}


@ignore_blas_flags
def kkt_certificate(z: np.ndarray, unit_A: np.ndarray, unit_b: np.ndarray,
                    hint: np.ndarray, active: np.ndarray, memo: KKTMemo | None = None,
                    tally=None) -> tuple | None:
    """Projection and multipliers, if proven, adding the test's flops to tally if a KKTTally is
    given."""
    # Candidate rows: the active set and every row near the hint
    active = np.asarray(active, dtype=bool)
    slacks = unit_A @ hint - unit_b
    near = slacks >= -_KKT_CANDIDATE_TOL * _kkt_slack_scale(z, unit_A, unit_b, hint)
    rows = np.where(active | near)[0]
    memo = KKTMemo() if memo is None else memo
    if tally is not None:
        tally.candidates = rows.size
        tally.flops += 4 * unit_A.size

    def derived(dual: bool) -> np.ndarray:
        """Rows binding at the projection onto the candidate rows' polyhedron."""
        key = (tuple(rows), dual)
        if key not in memo.derived:
            spent = 0 if tally is None else tally.flops
            N = unit_A[rows]
            if dual:
                weights = nonnegative_quadratic(N @ N.T, N @ z - unit_b[rows], tally=tally)
            else:
                apex = np.linalg.lstsq(N, unit_b[rows], rcond=None)[0]
                weights = nonnegative_least_squares(N.T, z - apex, tally=tally)
            memo.derived[key] = rows[weights > 0.0]
            if tally is not None:
                # the Gram matrix and N z for the dual, the apex for the cone
                if dual:
                    tally.flops += 2 * N.size * (len(N) + 1)
                else:
                    tally.solve(*N.shape)
                memo.costs[key] = tally.flops - spent
        elif tally is not None:
            tally.reused += memo.costs.get(key, 0)
        return memo.derived[key]

    def supports():
        """Candidate supports in the order they are tried."""
        # The rows that bind at the projection onto the polyhedron of the candidate rows
        # alone, by its dual (exact) and by the cone at a common point of their boundaries
        # (accurate when one exists), then the active set, all candidate rows as they
        # stand, and, when there are at most _KKT_EXHAUSTIVE_ROWS candidate rows, every
        # subset of them. Each is built only once the ones before it have failed
        if rows.size:
            yield derived(True)
            yield derived(False)
        yield np.where(active)[0]
        yield rows
        if rows.size <= _KKT_EXHAUSTIVE_ROWS:
            for size in range(1, rows.size + 1):
                for subset in combinations(rows, size):
                    yield np.array(subset)

    # A support repeated within one test is verified once, as the C test does, so a repeat is
    # neither verified nor counted as work the memo saved
    tried = set()
    for support in supports():
        key = tuple(support)
        if key in tried:
            continue
        tried.add(key)
        if key in memo.rejected:
            if tally is not None:
                tally.skipped += 1
                tally.reused += memo.costs.get(key, 0)
            continue
        spent = 0 if tally is None else tally.flops
        certified = _kkt_verify(z, unit_A, unit_b, support, tally=tally)
        if certified is not None:
            return certified
        memo.rejected.add(key)
        if tally is not None:
            memo.costs[key] = tally.flops - spent
    return None


def _kkt_verify(z: np.ndarray, unit_A: np.ndarray, unit_b: np.ndarray, support: np.ndarray,
                tally=None) -> tuple | None:
    """KKT test at the projection onto a support, adding its flops to tally if a KKTTally is
    given."""
    if tally is not None:
        # the slacks, their floors, the residual and its allowance, and rows @ z
        tally.verified += 1
        tally.flops += 8 * unit_A.size + 2 * support.size * len(z)
        if support.size:
            tally.solve(support.size, len(z))
    # The projection of z onto the boundaries of the support, by least squares, which
    # is backward stable however nearly parallel the rows are. The step lies in the
    # span of the support's normals, so a coordinate none of them touches keeps z's
    # value. Least squares leaves rounding there on the scale of the whole correction,
    # while the allowance there is built from that coordinate alone and can be far
    # smaller, so the true support would fail. A coordinate the support touches carries
    # that rounding however little the step moves it, so its slack floors take the
    # size of the whole correction rather than its own step
    x = z.copy()
    moved = np.zeros(len(z))
    if support.size:
        rows = unit_A[support]
        correction = np.linalg.lstsq(rows, rows @ z - unit_b[support], rcond=None)[0]
        touched = np.any(rows != 0.0, axis=0)
        correction[~touched] = 0.0
        x -= correction
        moved[touched] = np.linalg.norm(correction)

    # Feasible, and multipliers nonnegative on the tight rows only, both at rounding
    # level, so complementarity holds to rounding
    slacks = unit_A @ x - unit_b
    floor = rounding_floor(np.abs(unit_A) @ (np.abs(z) + moved) + np.abs(unit_b))
    if np.any(slacks > floor):
        return None
    tight = np.where(slacks >= -floor)[0]
    step = z - x
    multipliers = np.zeros(len(unit_b))
    if tight.size:
        multipliers[tight] = nonnegative_least_squares(unit_A[tight].T, step, tally=tally)

    # x is then the projection of z - r, so the residual r bounds its error. The
    # multiplier terms enter only at rounding: they grow as the tight normals become
    # nearly dependent, and letting them loosen the accuracy certified wrong supports.
    # Both least-squares solves spread their rounding over every coordinate the tight
    # rows touch, on the scale of the whole step, which a coordinate the step does not
    # move would otherwise have no allowance for
    residual = step - unit_A.T @ multipliers
    spread = np.zeros(len(z))
    if tight.size:
        spread[np.any(unit_A[tight] != 0.0, axis=0)] = np.linalg.norm(step)
    allowed = (_KKT_ACCURACY * np.abs(step)
               + rounding_floor(np.abs(z) + np.abs(x) + spread + np.abs(unit_A).T @ multipliers))
    if np.any(np.abs(residual) > allowed):
        return None
    return x, multipliers


# ---- work, in flops

# One flop is one addition or one multiplication, counted to leading order in the sizes a piece
# of the solver is given: n half-spaces, p coordinates, m active half-spaces, k the rank of the
# active normals. A piece that works on the touched coordinates alone is given their number as
# p. A power of a float or of a complex number counts as _POWER_FLOPS, an estimate. Comparisons,
# absolute values, copies and interpreter time are not counted. cost_model.py derives every
# count from the code above, sets them against the C solver's, and fits them to wall times
_POWER_FLOPS = 20


def lstsq_flops(rows: int, cols: int) -> int:
    """Flops of an SVD of a rows x cols matrix, or of a least-squares solve through one, at the
    cost of its bidiagonalisation."""
    short, long = min(rows, cols), max(rows, cols)
    return 4 * long * short * short


def lu_solve_flops(p: int, right_hand_sides: int) -> int:
    """Flops of numpy.linalg.solve on a p x p matrix: its LU factorisation and the
    substitutions for every right-hand side."""
    return 2 * p ** 3 // 3 + 2 * right_hand_sides * p * p


def matrix_power_flops(p: int, exponent: int) -> int:
    """Flops of numpy.linalg.matrix_power on a p x p matrix, which squares once for every bit of
    the exponent after the first and multiplies once for every set bit after the first."""
    if exponent < 2:
        return 0
    return 2 * p ** 3 * (exponent.bit_length() + bin(exponent).count("1") - 2)


def exact_cycle_flops(n: int, p: int, m: int) -> int:
    """exact_cycle: the shifted point and its slack at every half-space, and the projection at
    the m that end the cycle active."""
    return 4 * n * p + 2 * m * p


def cycle_map_flops(n: int, p: int, m: int) -> int:
    """cycle_map: a row of R and of R_scale at every half-space, the product of the prefix map
    by a dense projector at every active one, and the SVD of the active normals."""
    return 4 * n * p * (p + 1) + m * (2 * p ** 3 + 4 * p * p) + lstsq_flops(p, m)


def resolvent_flops(p: int) -> int:
    """touched_resolvent and the condition number of I - A_m, by SVD."""
    return p * p + lstsq_flops(p, p)


def closed_form_flops(n: int, p: int, envelope: bool) -> int:
    """closed_form: two LU solves, for x_inf and for RIA, the levels and drifts of every
    auxiliary, and, when the envelope is used, the contraction rate by SVD."""
    return (lu_solve_flops(p, 1) + lu_solve_flops(p, n) + 14 * n * p
            + (lstsq_flops(p, p) if envelope else 0))


def deflated_closed_form_flops(n: int, p: int, k: int) -> int:
    """deflated_closed_form: the projector P_1 onto the kernel, the condition number of
    I - A_m + P_1 by SVD, two LU solves, and the levels and drifts of every auxiliary."""
    return (2 * k * p * p + 4 * p * p + lstsq_flops(p, p) + lu_solve_flops(p, 1)
            + lu_solve_flops(p, n) + 14 * n * p)


def modal_form_flops(n: int, p: int, k: int) -> int:
    """The modal form of a singular episode: the block Q^T A_m Q, its eigendecomposition, the
    condition number of its complex eigenvectors, the coordinates of the transient, and the
    modal coefficients and envelopes of every half-space."""
    return (2 * k * p * (p + k) + 25 * k ** 3 + 4 * lstsq_flops(k, k) + 4 * lu_solve_flops(k, 1)
            + 8 * n * p * k + 10 * n * k * k)


def closed_form_cycle_flops(n: int, p: int) -> int:
    """One cycle of a regular episode in closed form: A_m z_t, the auxiliaries, the slacks, the
    rounding level of each, and the finality test."""
    return 2 * p * p + 8 * n * p


def stepped_cycle_flops(n: int, p: int, k: int) -> int:
    """One cycle stepped through the cycle map: the increment of every auxiliary and its
    rounding level, and the step confined to the span of the active normals."""
    return 4 * n * p + 2 * p * p + 4 * p * k


def frozen_probe_flops(p: int) -> int:
    """is_frozen: one application of the cycle map and of its magnitudes."""
    return 4 * p * p


def modal_block_flops(n: int, p: int, k: int, cycles: int) -> int:
    """One block of the modal scan: the powers of the k modes, and the auxiliaries, slacks and
    envelopes of every half-space at every cycle of the block, then the state, the
    stationarity probe and the clearance at the block's ends."""
    return (cycles * (18 * n * k + 2 * k * _POWER_FLOPS) + 20 * n * k + 16 * k * k + 8 * p * k
            + 4 * p * p)


def jump_length_flops(n: int, m: int) -> int:
    """jump_length: the slack of every inactive half-space, and the envelope horizon of every
    active one by bisection, about 60 evaluations of the envelope with a power each."""
    return 6 * n + 60 * (4 + _POWER_FLOPS) * m


def jump_landing_flops(n: int, p: int, exponent: int) -> int:
    """One landing an envelope jump tries: A_m to the exponent applied to the transient, one
    more cycle, and the activity there."""
    return matrix_power_flops(p, exponent) + 4 * p * p + 8 * n * p


def budget_jump_flops(n: int, p: int, exponent: int) -> int:
    """The jump to the budget once the active set is final: A_m to the exponent applied to the
    transient, and the auxiliaries there."""
    return matrix_power_flops(p, exponent) + 2 * p * p + 2 * n * p


def stall_flops(n: int, p: int) -> int:
    """A stall fast-forward: the increment of every auxiliary and its rounding level, and the
    first crossing."""
    return 4 * n * p + 12 * n


def replay_flops(p: int, m: int) -> int:
    """One cycle replayed into the history from the previous one: a projection at every active
    half-space."""
    return 4 * m * p


class KKTTally:
    """Work of one KKT test in flops, from the sizes of its products and least-squares solves,
    which kkt_certificate adds to as it runs."""

    def __init__(self) -> None:
        self.flops = 0
        # Flops that an earlier test of the same solve spent on a derived support or on a support
        # it rejected, which KKTMemo lets this test reuse. Without the memo, which the C solver
        # does not keep, this test would cost flops + reused
        self.reused = 0
        self.solves = 0
        self.candidates = 0
        self.verified = 0
        self.skipped = 0

    def solve(self, rows: int, cols: int) -> None:
        """Counts a least-squares solve on a rows x cols matrix."""
        self.solves += 1
        self.flops += lstsq_flops(rows, cols)

    def counting(self, gradient, gradient_flops: int, solve, shape) -> tuple:
        """The gradient and the support solve of _lawson_hanson, wrapped to count gradient_flops
        for every gradient and a solve on a matrix of shape(support) for every solve."""
        def counted_gradient(lam: np.ndarray) -> np.ndarray:
            self.flops += gradient_flops
            return gradient(lam)

        def counted_solve(support: np.ndarray) -> tuple:
            self.solve(*shape(support))
            return solve(support)
        return counted_gradient, counted_solve
