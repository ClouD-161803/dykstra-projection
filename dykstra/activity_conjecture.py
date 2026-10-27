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
