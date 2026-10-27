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
