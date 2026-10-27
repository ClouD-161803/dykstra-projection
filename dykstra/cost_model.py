"""The cost of an episode of the accelerated solver against the cost of a cycle of Dykstra's
algorithm, measured for the cost model of the SIAM paper's section on the accelerated algorithm,
which calls an episode, a run of cycles with a constant active set, a phase.

Run from the repository root with `python -m dykstra.cost_model`, with `--quick` for a short run
written to a temporary directory, or with `--from-csv` to rebuild the outputs from the records
of the last run without timing again. The problems are the six size tiers of tab:dykvsacc,
problems 0 to SEEDS - 1 of each as performance_timing.py draws them, and the wedge family of
problem_generators.py in every cell of WEDGE_DIMENSIONS and WEDGE_ANGLES, WEDGE_PROBLEMS each.
Every accelerated run goes on until it settles or reaches the CAP of performance_timing.py.

For every problem the script measures, each time the median of at least REPEATS runs:
- the time of a Dykstra cycle, in C as the difference between a run of C_DYKSTRA_CYCLES cycles
  and a run of one, and in Python over PYTHON_DYKSTRA_CYCLES cycles in solve() alone;
- the time and the flops of every stage of LTIVer5Solver built with profile=True, its counts,
  and one record per KKT test, beside the time of the same solver unprofiled;
- the time of a call of the C accelerated solver, its counters, its cycle map at cycle 1 as the
  difference between its one-cycle run and plain Dykstra's, and its first episode entry as the
  difference between its runs of two cycles and one;
- Dykstra's cycles to the squared errors TOLERANCE and TIGHT_TOLERANCE, by bisection on the C
  iterate, and the time of C Dykstra runs of that many cycles.

An episode's cost is everything a run costs beyond its exact cycles, each counted at the time of
a Dykstra cycle, divided by its episodes. In C this cost is inferred from the time per call and
the counters, since the C solver does not time its stages, and it includes the cycles evaluated
within the episode. Its break-even is the same cost in Dykstra cycles. An episode pays for
itself when the cycles it covers beyond its exact cycle, evaluated or skipped, outnumber its
break-even, and a run pays when it settles in less time than Dykstra needs to a tolerance, which
is timed in C and estimated in Python as the time of a cycle times the cycles.

The outputs go to results/paper: tab-cost-model.tex, one row per tier and per wedge cell, in the
column order its header comment gives; cost_model.csv, the same rows at full precision;
cost_model_problems.csv, one row per problem; cost_model_kkt.csv, one row per KKT test; and
cost_model_fits.csv, the fit of every profiled stage's time to its flops.
"""

# The flop model. n is the number of half-spaces, p the dimension, m the number of half-spaces
# active in an episode, k the rank of their normals, c the candidate rows of a KKT test, s the
# rows of a support, t the rows tight at its point, B the cycles of a block of the modal scan,
# and j the exponent of a matrix power, with bits(j) + ones(j) - 2 products of p x p matrices.
# A flop is one addition or one multiplication. Every count is the leading order read from the
# code of the stage. The Python counts are the functions at the end of lti_numerics.py, which a
# profiled LTISolver adds up stage by stage, and cost_model_fits.csv fits each stage's measured
# time to them. The C solver does not time its stages, hence its counts are read from
# C/accelerated.c and C/dykstra.c and checked only through the time per call.
#
# Dykstra cycle. Python, DykstraProjectionSolver: 6np + 2mp in n interpreted visits. C,
#   dykstra.c: 3np + 3mp, and a copy and a comparison of the n x p corrections.
# Exact cycle of the accelerated solver. Python and C: 4np + 2mp, the Python in n interpreted
#   visits.
# Cycle map, A_m, B_m, R, s and the span. Python: 4np(p + 1) + m(2p^3 + 4p^2) + 4p m min(p, m),
#   a dense p x p product per active row and the SVD of the active normals. C: 4np^2 + 4mp^2, a
#   rank-one update per active row, and a Jacobi SVD of the active normals only for a stepped or
#   deflated episode.
# Resolvent, I - A_m and its condition number. Python: 4p^3 by SVD. C: 2p^3/3 for the LU, and
#   4p^2 per power iteration for the norm of I - A_m, up to 500, and for the norm of its inverse,
#   up to 200.
# Closed-form setup. Python: 4p^3/3 + 2np^2 + 14np for x_inf, RIA, and the levels and drifts, and
#   4p^3 for the contraction rate by SVD. C: 2np^2 + 14np, and 4p^2 per power iteration for the
#   contraction rate, up to 500.
# Deflation of a singular episode. Python: 16p^3/3 + 2np^2 + 2kp^2 + 14np for the deflated
#   closed form, then 2kp(p + k) + 25k^3 + 19k^3 + 8npk + 10nk^2 for the eigendecomposition of
#   T = Q^T A_m Q, the condition number of its eigenvectors, and the modal coefficients. C: the
#   span, the resolvent of I - A_m + P_1 as above, 2np^2 + 14np, and 2kp^2 + 2k^2 p + 8npk for T
#   and the coefficients in the coordinates of the span, without an eigendecomposition.
# Evaluated cycle. Python, closed form: 2p^2 + 8np. Python, modal scan: B(18nk + 40k) per block
#   of B cycles, all B evaluated even when the switch comes earlier. C, either form:
#   2q^2 + 2nq + 2mq + 4(n - m)q with q = p for a regular and q = k for a deflated episode.
#   Python and C, stepped: 4np + 2p^2 + 4pk.
# Envelope jump. Python and C, at every evaluated cycle of a regular episode: about 1440m for
#   the horizons by bisection, unless an inactive half-space blocks the jump first. Python, per
#   landing tried: 2p^3 (bits(j) + ones(j) - 2) + 4p^2 + 8np. C, per landing: the same in the q
#   coordinates of the scan.
# Stall fast-forward. Python and C: 4np.
# KKT test. Python: 4np for the candidate rows; the dual, 2cp(c + 1) for its Gram matrix, then 4c^2
#   per Lawson-Hanson iteration and 4s^3 per least-squares solve; the cone, 4 max(c, p) min(c, p)^2
#   for its apex, then 8pc per iteration and 4 max(p, s) min(p, s)^2 per solve; and per support
#   verified, 8np + 2sp + 4 max(s, p) min(s, p)^2 and a nonnegative least-squares problem over
#   the t tight rows, 8pt per iteration and 4 max(p, s) min(p, s)^2 per solve. C: the same
#   supports in the same order, with the Gram matrix in c^2 p, and every Lawson-Hanson solve by
#   QR updates of about 8ps as a row enters or leaves.
# History. Python: 4mp per replayed cycle, and copies of n p per recorded cycle and of
#   (CAP - settled_at) n p at a settlement. C: none.

import argparse
import csv
import os
import statistics
import sys
import tempfile
import time

import numpy as np

from dykstra import c_solvers, paper_figures, performance_timing
from dykstra.convex_projection_solver import ConvexProjectionSolver, DykstraProjectionSolver
from dykstra.lti_solver import COUNTS, STAGES, LTIVer5Solver
from dykstra.paper_figures import quiet_accelerate_warnings, save_tex
from dykstra.problem_generators import WedgeTestPlanes, problem, problem_seed


TIERS = performance_timing.TABLE_TIERS
SEEDS = performance_timing.SEEDS
CAP = performance_timing.CAP
TOLERANCE = performance_timing.TOLERANCE

# Chosen here: the squared error at which Dykstra is compared with a certified settlement as
# well, the square of the KKT test's accuracy of 1e-9 for a unit distance moved
TIGHT_TOLERANCE = 1e-18

# The wedge cells, p and the angle theta in rad, and their problems
WEDGE_DIMENSIONS = (3, 8)
WEDGE_ANGLES = (0.1, 0.03, 0.01)
WEDGE_PROBLEMS = 5

# Chosen here: the seed of a wedge problem lies in a block of its own, set by p, the angle's
# index in WEDGE_ANGLES, and the problem's index
WEDGE_SEED = 12_000_000

# Chosen here: every time is the median of at least REPEATS runs, continued until they total
# REPEAT_SECONDS or number MAX_REPEATS, and the cycles of the runs that time a Dykstra cycle
REPEATS, REPEAT_SECONDS, MAX_REPEATS = 5, 0.2, 200
PYTHON_DYKSTRA_CYCLES, C_DYKSTRA_CYCLES = 100, 2000

# The stages a profiled solve charges to an episode's entry, split into the cycle map, its
# setup and the KKT test, and those charged to the cycles within an episode, apart from the
# history the Python solver records and the C solver does not
SETUP_STAGES = ("resolvent", "closed_form_setup", "deflation")
ENTRY_STAGES = ("cycle_maps",) + SETUP_STAGES + ("kkt_test",)
WITHIN_STAGES = ("closed_form_evaluation", "stepped_evaluation", "modal_scan", "envelope_jumps",
                 "budget_jumps", "stall_fast_forwards")

# The count each stage's time is fitted to beside its flops: the stage's entries, or for the KKT
# test its least-squares solves, each a call of numpy.linalg.lstsq
FIT_COUNT = {"kkt_test": "kkt_solves"}

QUICK_TIERS, QUICK_SEEDS = ((12, 4), (24, 8)), 1
QUICK_WEDGE_CELLS, QUICK_WEDGE_PROBLEMS = ((3, 0.1), (8, 0.01)), 1


def wedge_seed(num_dimensions: int, theta: float, index: int) -> int:
    """The seed of problem index of a wedge cell."""
    return WEDGE_SEED + 10_000 * num_dimensions + 1000 * WEDGE_ANGLES.index(theta) + index


def median_seconds(run) -> tuple[float, int]:
    """Median time of repeated calls of run, and their number."""
    timings = []
    while len(timings) < REPEATS or (sum(timings) < REPEAT_SECONDS and len(timings) < MAX_REPEATS):
        start = time.perf_counter()
        run()
        timings.append(time.perf_counter() - start)
    return statistics.median(timings), len(timings)


def c_dykstra_cycles_to_tolerance(z: np.ndarray, A: np.ndarray, b: np.ndarray,
                                  reference: np.ndarray, tolerance: float) -> int | None:
    """First budget whose C Dykstra iterate is within the squared error tolerance of reference,
    or None within CAP.

    The budget doubles from one cycle and is then bisected, the protocol of the supplement's
    benchmark section, which takes the error as monotone between the last two budgets.
    """
    def within(budget: int) -> bool:
        point = c_solvers.dykstra_c_projection(z, A, b, budget).projection
        return float(np.sum((point - reference) ** 2)) <= tolerance

    low, high = 0, 1
    while not within(high):
        if high >= CAP:
            return None
        low, high = high, min(2 * high, CAP)
    while high - low > 1:
        middle = (low + high) // 2
        low, high = (low, middle) if within(middle) else (middle, high)
    return high


def measure_dykstra(z: np.ndarray, A: np.ndarray, b: np.ndarray, reference: np.ndarray) -> dict:
    """Times of a Dykstra cycle in Python and in C, the fixed cost of a C call, and Dykstra's
    cycles to the tolerance."""
    python, _ = median_seconds(DykstraProjectionSolver(z, A, b, PYTHON_DYKSTRA_CYCLES).solve)
    one, _ = median_seconds(lambda: c_solvers.dykstra_c_projection(z, A, b, 1))
    cycles = c_solvers.dykstra_c_projection(z, A, b, C_DYKSTRA_CYCLES).cycles
    many, _ = median_seconds(lambda: c_solvers.dykstra_c_projection(z, A, b, C_DYKSTRA_CYCLES))
    # a plain C run stops early at a state that a cycle leaves unchanged bit for bit
    per_cycle = (many - one) / max(cycles - 1, 1)
    record = {"python_dykstra_seconds_per_cycle": python / PYTHON_DYKSTRA_CYCLES,
              "c_dykstra_seconds_per_cycle": per_cycle, "c_call_seconds": one - per_cycle,
              "c_dykstra_cycles_timed": cycles}
    for name, tolerance in (("tolerance", TOLERANCE), ("tight", TIGHT_TOLERANCE)):
        reached = c_dykstra_cycles_to_tolerance(z, A, b, reference, tolerance)
        budget = CAP if reached is None else reached
        seconds, _ = median_seconds(lambda: c_solvers.dykstra_c_projection(z, A, b, budget))
        record.update({f"dykstra_reached_{name}": reached is not None,
                       f"dykstra_cycles_to_{name}": budget,
                       f"c_dykstra_seconds_to_{name}": seconds})
    return record


def measure_accelerated_python(z: np.ndarray, A: np.ndarray, b: np.ndarray) -> tuple[dict, list]:
    """Medians of the profiled stages of LTIVer5Solver and of its unprofiled solve, its counts,
    and its KKT tests."""
    profiled = LTIVer5Solver(z, A, b, CAP, profile=True)
    plain = LTIVer5Solver(z, A, b, CAP)
    profiles, profiled_times, plain_times = [], [], []
    # the two solvers alternate, so that a change in the machine's load reaches both alike
    while len(profiles) < REPEATS or (sum(profiled_times) < REPEAT_SECONDS
                                      and len(profiles) < MAX_REPEATS):
        start = time.perf_counter()
        plain.solve()
        plain_times.append(time.perf_counter() - start)
        start = time.perf_counter()
        result = profiled.solve()
        profiled_times.append(time.perf_counter() - start)
        profiles.append(profiled.profile)
    last = profiles[-1]
    if any(p["counts"] != last["counts"] or p["flops"] != last["flops"] for p in profiles):
        raise RuntimeError("Repeated solves of one problem disagree on their counts or flops.")
    record = {"python_settled_at": -1 if result.settled_at is None else result.settled_at,
              "python_certificate": result.certificate or "",
              "python_settled_by": last["settled_by"] or "",
              "python_squared_error": float(np.sum((result.projection - profiled.actual_projection) ** 2)),
              "python_seconds": statistics.median(plain_times),
              "python_profiled_seconds": statistics.median(profiled_times),
              "python_repeats": len(profiles)}
    for stage in STAGES:
        record[f"seconds_{stage}"] = statistics.median(p["seconds"][stage] for p in profiles)
        record[f"flops_{stage}"] = last["flops"][stage]
        record[f"entries_{stage}"] = last["entries"][stage]
    record.update({f"count_{name}": last["counts"][name] for name in COUNTS})
    tests = []
    for number, test in enumerate(last["kkt_tests"]):
        test = dict(test)
        test["seconds"] = statistics.median(p["kkt_tests"][number]["seconds"] for p in profiles)
        tests.append({"test": number, **test})
    record["kkt_solves"] = sum(test["solves"] for test in tests)
    record["kkt_flops_without_memo"] = sum(test["flops_without_memo"] for test in tests)
    return record, tests


def measure_accelerated_c(z: np.ndarray, A: np.ndarray, b: np.ndarray) -> dict:
    """Median time of a call of the C accelerated solver, its counters, and the times of its
    first runs of one and two cycles."""
    result = c_solvers.accelerated_c_projection(z, A, b, CAP)
    seconds, repeats = median_seconds(lambda: c_solvers.accelerated_c_projection(z, A, b, CAP))
    one, _ = median_seconds(lambda: c_solvers.accelerated_c_projection(z, A, b, 1))
    two, _ = median_seconds(lambda: c_solvers.accelerated_c_projection(z, A, b, 2))
    return {"c_seconds": seconds, "c_repeats": repeats, "c_settled": result.settled,
            "c_cycles": result.cycles, "c_exact_cycles": result.exact_cycles,
            "c_cycles_skipped": result.cycles_skipped, "c_episodes": result.episodes,
            "c_step_episodes": result.step_episodes, "c_switches": result.switches,
            "c_skips": result.skips,
            "c_evaluated_cycles": result.cycles - result.exact_cycles - result.cycles_skipped,
            "c_one_cycle_seconds": one, "c_two_cycle_seconds": two}


def measure_problem(family: str, num_planes: int, num_dimensions: int, theta: float, index: int,
                    seed: int) -> tuple[dict, list]:
    """Every measurement of one problem."""
    if family == "tier":
        A, b, z = problem(num_planes, num_dimensions, seed)
    else:
        A, b, z = WedgeTestPlanes(theta, num_dimensions).problem(seed)
    record = {"family": family, "n": A.shape[0], "p": num_dimensions,
              "theta": theta if family == "wedge" else "", "index": index, "seed": seed}
    accelerated, tests = measure_accelerated_python(z, A, b)
    reference = ConvexProjectionSolver._find_optimal_solution(z, A, b)
    record.update(measure_dykstra(z, A, b, reference))
    record.update(accelerated)
    record.update(measure_accelerated_c(z, A, b))
    for test in tests:
        test.update({"family": family, "n": record["n"], "p": num_dimensions,
                     "theta": record["theta"], "index": index})
    return record, tests


def problem_list(tiers: tuple, seeds: int, wedge_cells: tuple, wedge_problems: int) -> list:
    """The problems to measure, as the arguments of measure_problem."""
    problems = [("tier", n, p, 0.0, index, problem_seed(n, index))
                for n, p in tiers for index in range(seeds)]
    problems += [("wedge", 2 * p - 2, p, theta, index, wedge_seed(p, theta, index))
                 for p, theta in wedge_cells for index in range(wedge_problems)]
    return problems
