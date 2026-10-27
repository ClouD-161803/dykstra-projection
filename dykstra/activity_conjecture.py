"""The activity-change study of the article's thm:threetwo, three half-planes in the plane.

Run from the repository root with `python -m dykstra.activity_conjecture` for the full study, about
three minutes on ten worker processes of an Apple M5 Pro, or with `--quick` for a reduced run into
a temporary directory. The table tab-conjecture.tex and the
file conjecture.csv go to results/paper at the repository root, and every number in them is
printed, together with the activity sequences of the two counterexamples.

thm:threetwo states that Dykstra's algorithm on n = 3 half-planes in general position, visited in
the order 1, 2, 3, changes activity at most six times in total and at most three times for any one
half-space, and at most three and two times after the opening cycle. General position means that
no two boundary lines are parallel, that the three lines do not meet in one point, and that the
polyhedron has nonempty interior.

Activity follows eq:activeset: half-space i is active in cycle t when the point it is visited
with, shifted by its auxiliary y_i, violates it, which leaves y_i > 0 after the visit. A change of
half-space i in cycle t >= 2 is a difference between its activity in cycles t - 1 and t, as in
activity_study.py, and these changes make the counts after the opening cycle. The article defines
an activation as a successful test for a previously inactive half-space, and every auxiliary is
zero before cycle 1, hence the opening cycle adds one change for each half-space active in cycle
1. The counts with the opening cycle add these to the counts after it. activity_study.py compares
cycle 1 with the rows z violates instead, and the counts under that convention are printed and
written beside. The four counts of an instance are the total and the largest count of one
half-space, each with and after the opening cycle, and COUNTS names them.

An exact run uses fractions.Fraction on rational unit normals, hence every visit is a few rational
operations. It ends at a certificate that the activity never changes again: an exact fixed point
of the state (x, y), or a wedge phase {i, j} that has lasted three cycles and whose limit is the
projection. The cycle map of a wedge phase has rank one, with eigenvalue (a_i . a_j)^2, so from the
second cycle of the phase on, y_i, y_j, and the test value of the third half-space k move
monotonically to their limits. These limits are the multipliers of the vertex v_ij, given by
z - v_ij = y_i a_i + y_j a_j, and the floor Gamma_k = a_k . v_ij - b_k. All three keep their
current signs when the multipliers are positive and Gamma_k is negative.

A floating-point run is vectorised over a batch of instances, and every visit takes Dykstra's
branch by the sign of its slack. A visit whose slack lies within lti_numerics.rounding_floor of
the magnitudes summed in it marks the instance as undecided, since rounding may have chosen the
branch. Undecided instances are counted and left out of the maxima. A run stops once its state is
fixed to the same rounding floor, once its wedge phase is certified as above with the limits
beyond CERTIFICATE_MARGIN, or after HORIZON cycles, in which case its counts are lower bounds and
it is not used as a maximum either. Every maximum found in floating point is checked in exact
arithmetic, after rounding its instance to nearby rationals, by check_maximum.
"""

from __future__ import annotations

import argparse
import csv
from concurrent.futures import ProcessPoolExecutor
from fractions import Fraction
import math
import os
import tempfile
import time

import numpy as np

from dykstra import paper_figures
from dykstra.lti_numerics import rounding_floor


# The four counts of thm:threetwo, with the bounds it claims
COUNTS = ("total", "per_half_space", "total_after_opening", "per_half_space_after_opening")
CLAIMED_BOUNDS = {"total": 6, "per_half_space": 3, "total_after_opening": 3,
                  "per_half_space_after_opening": 2, "total_violated_start": 6,
                  "per_half_space_violated_start": 3}
COUNT_LABELS = {"total": "changes in total",
                "per_half_space": "changes of one half-space",
                "total_after_opening": "changes after the opening cycle",
                "per_half_space_after_opening": "changes of one half-space after the opening cycle"}
# The two counts with the opening cycle under activity_study.py's convention, held to the same
# bounds
VIOLATED_COUNTS = ("total_violated_start", "per_half_space_violated_start")

# The reviewer's two counterexamples, as (normals, offsets, start)
COUNTEREXAMPLES = {
    "i": (((Fraction(-24, 25), Fraction(-7, 25)), (Fraction(-5, 13), Fraction(-12, 13)),
           (Fraction(21, 29), Fraction(-20, 29))),
          (Fraction(1), Fraction(1), Fraction(0)), (Fraction(0), Fraction(-7))),
    "ii": (((Fraction(-5, 13), Fraction(-12, 13)), (Fraction(-5, 13), Fraction(12, 13)),
            (Fraction(3, 5), Fraction(4, 5))),
           (Fraction(-1, 2), Fraction(-2), Fraction(3, 2)), (Fraction(10), Fraction(8))),
}

# Every constant below was chosen here.
# Random instances: normals at angles uniform on the circle, offsets standard normal, and starts
# normal with standard deviation START_SCALE, redrawn while feasible
SAMPLES, SAMPLE_BATCH = 10_000_000, 50_000
START_SCALE = 4.0
# General position, relative to the largest of |b| and the coordinates of z: 1 - |a_i . a_j| of
# at least PARALLEL_MARGIN, and every floor |Gamma_k| at least CONCURRENCY_MARGIN
PARALLEL_MARGIN, CONCURRENCY_MARGIN = 1e-6, 1e-6
# Cycle budget of a floating-point run
HORIZON = 20_000
# Margin, relative to the same scale, beyond which a floating-point wedge certificate is trusted
CERTIFICATE_MARGIN = 1e-9
# Hill climbing: chains per count, each started from one of the best sampled instances, steps
# per chain, chains per worker task, and the range of the log-uniform step size, in radians for
# the angles, in the units of b for the offsets, and in START_SCALE units for the start
CLIMB_CHAINS, CLIMB_STEPS, CLIMB_GROUP = 1000, 1000, 50
CLIMB_STEP_RANGE = (1e-4, 1e-1)
# The climbing objectives: the four counts and the two of activity_study.py's convention
OBJECTIVES = COUNTS + VIOLATED_COUNTS
# Instances kept per objective and task, as candidates for chains and for the exact check
KEEP = 200
# Denominators tried in turn when an instance is rounded to rationals, and the cycle budget of an
# exact run
RATIONAL_DENOMINATORS = (100, 1000, 10_000, 100_000)
EXACT_CYCLES = 5000
# Candidates tried per maximum in the exact check
EXACT_CANDIDATES = 8
SEED = 31_000_000
WORKERS = 10

QUICK = {"SAMPLES": 4000, "SAMPLE_BATCH": 2000, "HORIZON": 2000, "CLIMB_CHAINS": 4,
         "CLIMB_STEPS": 6, "CLIMB_GROUP": 4, "KEEP": 8, "EXACT_CANDIDATES": 2}

# Run outcomes of a floating-point run
FIXED, CERTIFIED, BUDGET = 1, 2, 3


def settings(quick: bool) -> dict:
    """The study's parameters, reduced by QUICK for a quick run."""
    chosen = {name: globals()[name] for name in (
        "SAMPLES", "SAMPLE_BATCH", "HORIZON", "CLIMB_CHAINS", "CLIMB_STEPS", "CLIMB_GROUP", "KEEP",
        "EXACT_CANDIDATES", "WORKERS")}
    if quick:
        chosen.update(QUICK)
    return chosen


def four_counts(after: np.ndarray, opening: np.ndarray) -> dict:
    """The four counts from the changes after the opening cycle and those of the opening cycle.

    Both arrays hold one count per half-space along their last axis, and the counts keep the
    leading axes.
    """
    with_opening = after + opening
    return {"total": with_opening.sum(axis=-1), "per_half_space": with_opening.max(axis=-1),
            "total_after_opening": after.sum(axis=-1),
            "per_half_space_after_opening": after.max(axis=-1)}


def all_counts(after: np.ndarray, cycle1: np.ndarray, violated: np.ndarray) -> dict:
    """The four counts and the two of the activity_study.py convention, from a run's activity.

    cycle1 is the activity of cycle 1, and violated marks the rows the start violates.
    """
    counts = four_counts(after, cycle1.astype(int))
    alternative = four_counts(after, (cycle1 != violated).astype(int))
    counts["total_violated_start"] = alternative["total"]
    counts["per_half_space_violated_start"] = alternative["per_half_space"]
    return counts


# Exact arithmetic

def exact_cycle(x: tuple, y: tuple, normals: tuple, offsets: tuple) -> tuple:
    """One Dykstra cycle in rationals, as lti_numerics.exact_cycle does in floating point.

    Returns the point, the auxiliaries, and the activity of each visit.
    """
    x0, x1 = x
    auxiliaries = list(y)
    active = []
    for i, ((u, v), offset) in enumerate(zip(normals, offsets)):
        s0, s1 = x0 + auxiliaries[i] * u, x1 + auxiliaries[i] * v
        slack = u * s0 + v * s1 - offset
        if slack > 0:
            x0, x1, auxiliaries[i] = s0 - slack * u, s1 - slack * v, slack
        else:
            x0, x1, auxiliaries[i] = s0, s1, Fraction(0)
        active.append(slack > 0)
    return (x0, x1), tuple(auxiliaries), tuple(active)


def wedge_limits(normals, offsets, start, k: int) -> tuple:
    """Multipliers of the vertex of the two half-spaces other than k, and the floor of k.

    Works on Fractions and on floats alike, and returns (y_i, y_j, Gamma_k) with i < j.
    """
    i, j = (m for m in range(3) if m != k)
    (ui, vi), (uj, vj), (uk, vk) = normals[i], normals[j], normals[k]
    det = ui * vj - vi * uj
    v0 = (offsets[i] * vj - offsets[j] * vi) / det
    v1 = (ui * offsets[j] - uj * offsets[i]) / det
    d0, d1 = start[0] - v0, start[1] - v1
    return ((d0 * vj - d1 * uj) / det, (ui * d1 - vi * d0) / det,
            uk * v0 + vk * v1 - offsets[k])


def exact_run(normals: tuple, offsets: tuple, start: tuple, cycles: int = EXACT_CYCLES) -> dict:
    """Run Dykstra exactly until its activity is certified final or for cycles cycles.

    normals must be rational unit vectors, and ValueError is raised otherwise. Returns the
    activity of every cycle, cycle 0 holding the rows the start violates, the changes after the
    opening cycle, all six counts, the certificate ("fixed", "wedge", or None at the budget), and
    the cycle it fired in.
    """
    if any(u * u + v * v != 1 for u, v in normals):
        raise ValueError("the normals of an exact run must be unit vectors.")
    x, y = tuple(start), (Fraction(0),) * 3
    violated = tuple(u * x[0] + v * x[1] - b > 0 for (u, v), b in zip(normals, offsets))
    activity, phase_start, certificate, cycle = [violated], 1, None, cycles
    for t in range(1, cycles + 1):
        x_new, y_new, active = exact_cycle(x, y, normals, offsets)
        if t >= 2 and active != activity[-1]:
            phase_start = t
        activity.append(active)
        if (x_new, y_new) == (x, y):
            certificate, cycle = "fixed", t
            break
        x, y = x_new, y_new
        if sum(active) == 2 and t - phase_start >= 2:
            y_i, y_j, gamma = wedge_limits(normals, offsets, start, active.index(False))
            if y_i > 0 and y_j > 0 and gamma < 0:
                certificate, cycle = "wedge", t
                break
    record = np.array(activity)
    after = (record[2:] != record[1:-1]).sum(axis=0)
    counts = {name: int(value) for name, value in
              all_counts(after, record[1], record[0]).items()}
    return {"activity": activity, "after": after, "counts": counts, "certificate": certificate,
            "cycle": cycle, "point": x, "auxiliaries": y}


def activity_sequence(activity: list, half_space: int) -> str:
    """Runs of constant activity of one half-space, e.g. "active 0-1, inactive 2-9"."""
    values = [active[half_space] for active in activity]
    runs, start = [], 0
    for t in range(1, len(values) + 1):
        if t == len(values) or values[t] != values[start]:
            state = "active" if values[start] else "inactive"
            runs.append(f"{state} {start}" if t - 1 == start else f"{state} {start}-{t - 1}")
            start = t
    return ", ".join(runs)


def set_sequence(activity: list) -> str:
    """Runs of a constant active set, e.g. "123 0-1, 23 2-9", with cycle 0 the rows z violates."""
    names = ["".join(str(m + 1) for m in range(3) if active[m]) or "none" for active in activity]
    runs, start = [], 0
    for t in range(1, len(names) + 1):
        if t == len(names) or names[t] != names[start]:
            runs.append(f"{names[start]} {start}" if t - 1 == start
                        else f"{names[start]} {start}-{t - 1}")
            start = t
    return ", ".join(runs)


def exact_general_position(normals: tuple, offsets: tuple, start: tuple) -> bool:
    """Whether an exact instance is in general position and its start lies outside."""
    (u1, v1), (u2, v2), (u3, v3) = normals
    w = (u2 * v3 - v2 * u3, u3 * v1 - v3 * u1, u1 * v2 - v1 * u2)
    if any(weight == 0 for weight in w):
        return False
    D = sum(weight * b for weight, b in zip(w, offsets))
    if D == 0:
        return False
    if all(weight > 0 for weight in w) and D <= 0 or all(weight < 0 for weight in w) and D >= 0:
        return False
    return any(u * start[0] + v * start[1] - b > 0 for (u, v), b in zip(normals, offsets))


def rational_unit_vector(angle: float, denominator: int) -> tuple:
    """A rational unit vector near (cos angle, sin angle), from a rational tangent of half the angle."""
    # (1 - s^2, 2 s) / (1 + s^2) is a unit vector for every rational s, and keeping |s| <= 1 by
    # turning a vector beyond a quarter turn through pi keeps the denominator small
    angle = math.remainder(angle, 2.0 * math.pi)
    flip = abs(angle) > math.pi / 2
    s = Fraction(math.tan((angle - math.copysign(math.pi, angle) if flip else angle) / 2.0))
    s = s.limit_denominator(denominator)
    u, v = (1 - s * s) / (1 + s * s), 2 * s / (1 + s * s)
    return (-u, -v) if flip else (u, v)


def rationalise(parameters: np.ndarray, denominator: int) -> tuple:
    """An exact instance (normals, offsets, start) near the eight floating-point parameters."""
    normals = tuple(rational_unit_vector(float(angle), denominator) for angle in parameters[:3])
    offsets = tuple(Fraction(float(b)).limit_denominator(denominator) for b in parameters[3:6])
    start = tuple(Fraction(float(c)).limit_denominator(denominator) for c in parameters[6:])
    return normals, offsets, start


def instance_text(normals: tuple, offsets: tuple, start: tuple) -> str:
    """An exact instance written as fractions, in the form the counterexamples are given."""
    def vector(values):
        return "(" + ", ".join(str(value) for value in values) + ")"
    return (f"a_1 = {vector(normals[0])}, a_2 = {vector(normals[1])}, a_3 = {vector(normals[2])}, "
            f"b = {vector(offsets)}, x0 = {vector(start)}")


# Floating-point runs

def arrays(parameters: np.ndarray) -> tuple:
    """Unit normals (B, 3, 2), offsets (B, 3), and starts (B, 2) of a batch of parameters.

    Each row of parameters holds the three normal angles, the three offsets, and the start.
    """
    angles = parameters[:, :3]
    normals = np.stack([np.cos(angles), np.sin(angles)], axis=-1)
    return normals, parameters[:, 3:6], parameters[:, 6:]


def valid(parameters: np.ndarray) -> np.ndarray:
    """Which instances are in general position, with margins, and start outside the polyhedron."""
    normals, offsets, starts = arrays(parameters)
    u, v = normals[..., 0], normals[..., 1]
    w = np.stack([u[:, 1] * v[:, 2] - v[:, 1] * u[:, 2], u[:, 2] * v[:, 0] - v[:, 2] * u[:, 0],
                  u[:, 0] * v[:, 1] - v[:, 0] * u[:, 1]], axis=1)
    D = (w * offsets).sum(axis=1)
    scale = np.maximum(np.abs(offsets).max(axis=1), np.abs(starts).max(axis=1))
    dots = np.stack([(normals[:, i] * normals[:, j]).sum(axis=1)
                     for i, j in ((0, 1), (0, 2), (1, 2))], axis=1)
    with np.errstate(divide="ignore", invalid="ignore"):
        floors = np.abs(D[:, None] / w)
    apart = (1.0 - np.abs(dots) >= PARALLEL_MARGIN).all(axis=1)
    not_concurrent = (floors >= CONCURRENCY_MARGIN * scale[:, None]).all(axis=1)
    # sum_i w_i (a_i . x - b_i) = -D at every x, so when the normals positively span the plane,
    # i.e. every w_i has one sign, an interior point needs D of that sign
    positive, negative = (w > 0).all(axis=1), (w < 0).all(axis=1)
    interior = ~(positive & (D <= 0)) & ~(negative & (D >= 0))
    outside = ((normals * starts[:, None, :]).sum(axis=2) - offsets > 0).any(axis=1)
    return apart & not_concurrent & interior & outside


def draw_parameters(rng: np.random.Generator, count: int) -> np.ndarray:
    """count random valid instances as rows of parameters."""
    drawn = []
    while sum(len(d) for d in drawn) < count:
        batch = np.concatenate([rng.uniform(0.0, 2.0 * np.pi, (count, 3)),
                                rng.standard_normal((count, 3)),
                                START_SCALE * rng.standard_normal((count, 2))], axis=1)
        drawn.append(batch[valid(batch)])
    return np.concatenate(drawn)[:count]


def wedge_certificates(normals: np.ndarray, offsets: np.ndarray, starts: np.ndarray) -> np.ndarray:
    """Whether the wedge phase without half-space k ends at the projection, as (3, B) flags."""
    scale = np.maximum(np.abs(offsets).max(axis=1), np.abs(starts).max(axis=1))
    margin = CERTIFICATE_MARGIN * scale
    flags = np.zeros((3, len(starts)), dtype=bool)
    for k in range(3):
        y_i, y_j, gamma = wedge_limits([(normals[:, m, 0], normals[:, m, 1]) for m in range(3)],
                                       [offsets[:, m] for m in range(3)],
                                       (starts[:, 0], starts[:, 1]), k)
        flags[k] = (y_i > margin) & (y_j > margin) & (gamma < -margin)
    return flags


def float_runs(parameters: np.ndarray, horizon: int) -> dict:
    """float_runs_on of the instances given as rows of parameters."""
    return float_runs_on(*arrays(parameters), horizon)


def float_runs_on(normals: np.ndarray, offsets: np.ndarray, starts: np.ndarray,
                  horizon: int) -> dict:
    """Run plain Dykstra on a batch of instances and count their activity changes.

    normals (B, 3, 2) are unit vectors, offsets are (B, 3), and starts are (B, 2). Returns per
    instance the changes after the opening cycle of each half-space, the activity of
    cycle 1, the rows the start violates, the cycle the run stopped in, its outcome (FIXED,
    CERTIFIED, or BUDGET), whether a visit was undecided, the last cycle with a change, and all six
    counts.
    """
    count = len(starts)
    u, v = np.ascontiguousarray(normals[..., 0].T), np.ascontiguousarray(normals[..., 1].T)
    b = np.ascontiguousarray(offsets.T)
    certificates = wedge_certificates(normals, offsets, starts)
    out = {"after": np.zeros((count, 3), dtype=int), "cycle1": np.zeros((count, 3), dtype=bool),
           "violated": ((normals * starts[:, None, :]).sum(axis=2) - offsets > 0),
           "stopped_at": np.full(count, horizon), "outcome": np.full(count, BUDGET),
           "undecided": np.zeros(count, dtype=bool), "last_change": np.zeros(count, dtype=int)}

    alive = np.arange(count)
    x0, x1 = starts[:, 0].copy(), starts[:, 1].copy()
    y = np.zeros((3, count))
    label = np.zeros((3, count), dtype=bool)
    previous_label = np.zeros((3, count), dtype=bool)
    cycle1 = np.zeros((3, count), dtype=bool)
    after = np.zeros((3, count), dtype=int)
    phase_start = np.ones(count, dtype=int)
    last_change = np.zeros(count, dtype=int)
    undecided = np.zeros(count, dtype=bool)
    columns = np.arange(count)
    for t in range(1, horizon + 1):
        previous_x0, previous_x1, previous_y = x0, x1, y.copy()
        for i in range(3):
            s0, s1 = x0 + y[i] * u[i], x1 + y[i] * v[i]
            p0, p1 = u[i] * s0, v[i] * s1
            slack = p0 + p1 - b[i]
            undecided |= np.abs(slack) <= rounding_floor(np.abs(p0) + np.abs(p1) + np.abs(b[i]))
            active = slack > 0.0
            step = np.where(active, slack, 0.0)
            x0, x1 = s0 - step * u[i], s1 - step * v[i]
            y[i] = step
            label[i] = active
        if t == 1:
            cycle1 = label.copy()
            previous_label = label.copy()
        else:
            changed = label != previous_label
            after += changed
            moved = changed.any(axis=0)
            last_change[moved] = t
            phase_start[moved] = t
            previous_label = label.copy()

        floor = rounding_floor(np.maximum(np.abs(previous_x0), np.abs(previous_x1))
                               + np.abs(previous_y).sum(axis=0) + np.abs(b).sum(axis=0))
        fixed = ((np.abs(x0 - previous_x0) <= floor) & (np.abs(x1 - previous_x1) <= floor)
                 & (np.abs(y - previous_y) <= floor).all(axis=0))
        wedge = (label.sum(axis=0) == 2) & (t - phase_start >= 2)
        certified = wedge & certificates[np.argmin(label, axis=0), columns] & ~fixed
        done = fixed | certified
        if t == horizon:
            done = np.ones_like(done)
        if done.any():
            finished = alive[done]
            out["after"][finished] = after[:, done].T
            out["cycle1"][finished] = cycle1[:, done].T
            out["stopped_at"][finished] = t
            out["outcome"][finished] = np.where(fixed[done], FIXED,
                                                np.where(certified[done], CERTIFIED, BUDGET))
            out["undecided"][finished] = undecided[done]
            out["last_change"][finished] = last_change[done]
            keep = ~done
            if not keep.any():
                break
            alive, x0, x1, y = alive[keep], x0[keep], x1[keep], y[:, keep]
            u, v, b = u[:, keep], v[:, keep], b[:, keep]
            label, previous_label, cycle1 = label[:, keep], previous_label[:, keep], cycle1[:, keep]
            after, phase_start = after[:, keep], phase_start[keep]
            last_change, undecided = last_change[keep], undecided[keep]
            certificates = certificates[:, keep]
            columns = np.arange(alive.size)
    out.update(all_counts(out["after"], out["cycle1"], out["violated"]))
    return out


def usable(runs: dict) -> np.ndarray:
    """Runs whose counts are final and decided beyond rounding."""
    return (runs["outcome"] != BUDGET) & ~runs["undecided"]


def best_rows(parameters: np.ndarray, runs: dict, keep: int) -> dict:
    """The keep usable instances with the highest value of each objective.

    Ties go to the run that stopped first. Returns, per objective, the parameters, values, and
    stopping cycles.
    """
    rows = np.flatnonzero(usable(runs))
    best = {}
    for name in OBJECTIVES:
        order = np.lexsort((runs["stopped_at"][rows], -runs[name][rows]))[:keep]
        chosen = rows[order]
        best[name] = {"parameters": parameters[chosen], "values": runs[name][chosen],
                      "stopped_at": runs["stopped_at"][chosen]}
    return best


def merge_best(parts: list, keep: int) -> dict:
    """The best keep instances per objective over several results of best_rows."""
    merged = {}
    for name in OBJECTIVES:
        parameters = np.concatenate([part[name]["parameters"] for part in parts])
        values = np.concatenate([part[name]["values"] for part in parts])
        stopped = np.concatenate([part[name]["stopped_at"] for part in parts])
        order = np.lexsort((stopped, -values))[:keep]
        merged[name] = {"parameters": parameters[order], "values": values[order],
                        "stopped_at": stopped[order]}
    return merged


def histogram(runs: dict, mask: np.ndarray) -> dict:
    """The number of instances at each value of each objective, over the runs in mask."""
    return {name: dict(zip(*(values.tolist() for values in np.unique(runs[name][mask],
                                                                      return_counts=True))))
            for name in OBJECTIVES}


def sampling_task(task: tuple) -> dict:
    """Draw and run one batch of random instances, and keep the best of each objective."""
    seed, count, horizon, keep = task
    started = time.perf_counter()
    parameters = draw_parameters(np.random.default_rng(seed), count)
    runs = float_runs(parameters, horizon)
    mask = usable(runs)
    return {"best": best_rows(parameters, runs, keep), "histogram": histogram(runs, mask),
            "sampled": count, "undecided": int(runs["undecided"].sum()),
            "budget": int((runs["outcome"] == BUDGET).sum()),
            "fixed": int((runs["outcome"] == FIXED).sum()),
            "certified": int((runs["outcome"] == CERTIFIED).sum()),
            "longest": int(runs["stopped_at"][mask].max(initial=0)),
            "seconds": time.perf_counter() - started}


def climb_task(task: tuple) -> dict:
    """Hill-climb one group of chains on one objective.

    Each step perturbs every chain's angles, offsets, and start by a log-uniform step and keeps
    the proposal if it is valid, usable, and does not lower the objective, so that a chain also
    drifts along a level of the objective. Returns the best instances of every objective among
    all usable proposals, the number of usable proposals at each value of each objective, and the
    numbers of proposals, valid proposals run, usable proposals, and accepted steps.
    """
    objective, chains, steps, horizon, keep, seed = task
    rng = np.random.default_rng(seed)
    current = np.array(chains, dtype=float)
    value = float_runs(current, horizon)[objective]
    parts, proposals, evaluated, kept, accepted = [], 0, 0, 0, 0
    values = {name: {} for name in OBJECTIVES}
    scales = np.array([1.0] * 6 + [START_SCALE] * 2)
    low, high = np.log(CLIMB_STEP_RANGE[0]), np.log(CLIMB_STEP_RANGE[1])
    for _ in range(steps):
        size = np.exp(rng.uniform(low, high, (len(current), 1)))
        proposal = current + size * scales * rng.standard_normal(current.shape)
        ok = valid(proposal)
        proposals += len(proposal)
        if not ok.any():
            continue
        rows = np.flatnonzero(ok)
        runs = float_runs(proposal[rows], horizon)
        mask = usable(runs)
        evaluated += len(rows)
        kept += int(mask.sum())
        parts.append(best_rows(proposal[rows], runs, keep))
        for name, counted in histogram(runs, mask).items():
            for level, number in counted.items():
                values[name][level] = values[name].get(level, 0) + number
        gain = mask & (runs[objective] >= value[rows])
        accepted += int(gain.sum())
        value[rows[gain]] = runs[objective][gain]
        current[rows[gain]] = proposal[rows[gain]]
    best = merge_best(parts, keep) if parts else None
    return {"best": best, "proposals": proposals, "evaluated": evaluated, "usable": kept,
            "accepted": accepted, "histogram": values}


def map_tasks(function, tasks: list, workers: int) -> list:
    """function over tasks, on a pool of workers processes, or in this process for one worker."""
    if workers <= 1:
        return [function(task) for task in tasks]
    with ProcessPoolExecutor(workers) as pool:
        return list(pool.map(function, tasks))
