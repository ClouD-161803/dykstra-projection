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


def derived(record: dict) -> dict:
    """The quantities of the cost model for one problem."""
    episodes = record["count_episodes"]
    exact = record["count_exact_cycles"]
    covered = record["count_evaluated_cycles"] + record["count_skipped_cycles"]
    c_cycle, python_cycle = record["c_dykstra_seconds_per_cycle"], record["python_dykstra_seconds_per_cycle"]
    c_episode = (record["c_seconds"] - record["c_call_seconds"]
                 - c_cycle * record["c_exact_cycles"]) / record["c_exact_cycles"]
    python_episode = (record["python_seconds"] - python_cycle * exact) / episodes
    stage = {name: record[f"seconds_{name}"] / episodes for name in STAGES}
    return {
        "c_us_per_cycle": 1e6 * c_cycle, "python_us_per_cycle": 1e6 * python_cycle,
        "episodes": episodes, "exact_cycles": exact,
        "evaluated_cycles": record["count_evaluated_cycles"],
        "skipped_cycles": record["count_skipped_cycles"],
        "kkt_attempts": record["count_kkt_attempts"],
        "cycles_per_episode": covered / episodes,
        "python_us_cycle_map": 1e6 * stage["cycle_maps"],
        "python_us_setup": 1e6 * sum(stage[name] for name in SETUP_STAGES),
        "python_us_kkt": 1e6 * stage["kkt_test"],
        "python_us_within": 1e6 * sum(stage[name] for name in WITHIN_STAGES),
        "python_us_history": 1e6 * stage["history"],
        "python_us_entry": 1e6 * sum(stage[name] for name in ENTRY_STAGES),
        "python_us_per_episode": 1e6 * python_episode,
        "python_ms_per_run": 1e3 * record["python_seconds"],
        "c_ms_per_call": 1e3 * record["c_seconds"],
        "c_us_per_episode": 1e6 * c_episode,
        "c_us_cycle_map": 1e6 * (record["c_one_cycle_seconds"] - record["c_call_seconds"] - c_cycle),
        "c_us_first_entry": 1e6 * (record["c_two_cycle_seconds"] - record["c_one_cycle_seconds"]),
        "c_breakeven_cycles": c_episode / c_cycle,
        "python_breakeven_cycles": python_episode / python_cycle,
        "python_entry_breakeven_cycles": sum(stage[name] for name in ENTRY_STAGES) / python_cycle,
        "dykstra_cycles_to_tolerance": record["dykstra_cycles_to_tolerance"],
        "settle_cycle": record["python_settled_at"] if record["python_settled_at"] >= 0 else CAP,
        "c_speedup": record["c_dykstra_seconds_to_tolerance"] / record["c_seconds"],
        "python_speedup": python_cycle * record["dykstra_cycles_to_tolerance"] / record["python_seconds"],
        "dykstra_cycles_to_tight": record["dykstra_cycles_to_tight"],
        "c_speedup_tight": record["c_dykstra_seconds_to_tight"] / record["c_seconds"],
        "python_speedup_tight": python_cycle * record["dykstra_cycles_to_tight"] / record["python_seconds"],
    }


SUMMARY_COLUMNS = ("c_us_per_cycle", "python_us_per_cycle", "episodes", "exact_cycles",
                   "evaluated_cycles", "skipped_cycles", "kkt_attempts", "python_us_cycle_map",
                   "python_us_setup", "python_us_kkt", "python_us_within", "python_us_history",
                   "c_ms_per_call",
                   "c_us_per_episode", "c_breakeven_cycles", "cycles_per_episode")
EXTRA_COLUMNS = ("python_us_entry", "python_us_per_episode", "python_ms_per_run",
                 "python_breakeven_cycles", "python_entry_breakeven_cycles", "c_us_cycle_map",
                 "c_us_first_entry", "dykstra_cycles_to_tolerance", "settle_cycle", "c_speedup",
                 "python_speedup", "dykstra_cycles_to_tight", "c_speedup_tight",
                 "python_speedup_tight")


def summarise(records: list[dict]) -> list[dict]:
    """One row per tier and per wedge cell: medians over its problems of every quantity of
    derived, and the counts that reached the tolerance and settled."""
    rows = {}
    for record in records:
        key = (record["family"], record["n"], record["p"], record["theta"])
        rows.setdefault(key, []).append(record)
    summary = []
    for (family, n, p, theta), group in rows.items():
        values = [derived(record) for record in group]
        row = {"family": family, "n": n, "p": p, "theta": theta, "problems": len(group),
               "dykstra_reached": sum(bool(r["dykstra_reached_tolerance"]) for r in group),
               "dykstra_reached_tight": sum(bool(r["dykstra_reached_tight"]) for r in group),
               "settled": sum(r["python_certificate"] == "kkt" for r in group),
               "c_settled": sum(bool(r["c_settled"]) for r in group)}
        for column in SUMMARY_COLUMNS + EXTRA_COLUMNS:
            row[column] = statistics.median(v[column] for v in values)
        row["episodes_paying_c"] = sum(v["cycles_per_episode"] > v["c_breakeven_cycles"] for v in values)
        row["runs_paying_c"] = sum(v["c_speedup"] > 1.0 for v in values)
        row["runs_paying_python"] = sum(v["python_speedup"] > 1.0 for v in values)
        row["runs_paying_c_tight"] = sum(v["c_speedup_tight"] > 1.0 for v in values)
        summary.append(row)
    return summary


def fit(stage: str, count: str, seconds: np.ndarray, flops: np.ndarray, counts: np.ndarray) -> dict:
    """The fit seconds = alpha flops + beta counts in relative least squares, so that the slowest
    problems do not decide it alone, with its residuals relative to the measured times."""
    design = np.column_stack([flops, counts]).astype(float)
    alpha, beta = np.linalg.lstsq(design / seconds[:, None], np.ones(len(seconds)), rcond=None)[0]
    # a negative coefficient is dropped and the other refitted alone
    if min(alpha, beta) < 0.0:
        column = 0 if alpha > beta else 1
        single = np.linalg.lstsq(design[:, [column]] / seconds[:, None], np.ones(len(seconds)),
                                 rcond=None)[0][0]
        alpha, beta = (single, 0.0) if column == 0 else (0.0, single)
    residual = np.abs(design @ np.array([alpha, beta]) / seconds - 1.0)
    return {"stage": stage, "count": count, "problems": len(seconds), "ns_per_flop": 1e9 * alpha,
            "gflops": 1e-9 / alpha if alpha > 0 else 0.0, "us_per_count": 1e6 * beta,
            "median_relative_residual": float(np.median(residual)),
            "largest_relative_residual": float(np.max(residual))}


def fit_stages(records: list[dict]) -> list[dict]:
    """For every profiled stage, the fit of its time to its flops and its count over the problems
    that ran it, then the fit of a Dykstra cycle to its flops and half-spaces in Python and C."""
    fits = []
    for stage in STAGES[:-1]:
        count = FIT_COUNT.get(stage, f"entries_{stage}")
        used = [r for r in records if r[f"seconds_{stage}"] > 0.0 and r[f"flops_{stage}"] > 0]
        if len(used) >= 3:
            fits.append(fit(stage, count.replace(f"_{stage}", ""),
                            np.array([r[f"seconds_{stage}"] for r in used]),
                            np.array([r[f"flops_{stage}"] for r in used]),
                            np.array([r[count] for r in used])))
    for implementation, flops in (("python", 6), ("c", 3)):
        used = [r for r in records if r[f"{implementation}_dykstra_seconds_per_cycle"] > 0.0]
        if len(used) >= 3:
            fits.append(fit(f"dykstra_cycle_{implementation}", "half-spaces",
                            np.array([r[f"{implementation}_dykstra_seconds_per_cycle"] for r in used]),
                            np.array([flops * r["n"] * r["p"] for r in used]),
                            np.array([r["n"] for r in used])))
    return fits


def number(value: float) -> str:
    """Three significant figures, whole numbers from 100 up."""
    if value >= 100 or value == int(value):
        return f"{value:.0f}"
    return f"{value:.3g}"


def table_rows(summary: list[dict]) -> str:
    """Rows of tab:cost-model, one per tier and per wedge cell."""
    lines = []
    for row in summary:
        if row["family"] == "tier":
            label = f"${row['n']} \\times {row['p']}$"
        else:
            label = f"wedge, $p = {row['p']}$, $\\theta = {row['theta']:g}$"
        cells = [number(row[column]) for column in SUMMARY_COLUMNS]
        lines.append(f"  {label:<34} & " + " & ".join(f"${cell}$" for cell in cells) + r" \\")
    return "\n".join(lines) + "\n"


def table_header(summary: list[dict]) -> str:
    """The LaTeX comment that names every column of tab:cost-model and its source."""
    counts = "; ".join(f"{row['n']} x {row['p']}" + (f" theta {row['theta']:g}" if row["family"] == "wedge" else "")
                       + f": Dykstra reached {row['dykstra_reached']}/{row['problems']}, settled "
                       f"{row['settled']}/{row['problems']} in Python and {row['c_settled']}/{row['problems']} in C, "
                       f"Dykstra reached {TIGHT_TOLERANCE:g} {row['dykstra_reached_tight']}/{row['problems']}"
                       for row in summary)
    return ("% an episode is the paper's phase. medians over the problems of each row, every "
            "time the median of at least "
            f"{REPEATS} runs, budget cap {CAP} cycles\n"
            "% columns: problem; Dykstra us per cycle in C; Dykstra us per cycle in Python; "
            "episodes; exact cycles; evaluated cycles; skipped cycles; KKT tests; Python us per "
            "episode in the cycle map, the setup (resolvent, closed-form setup and deflation), the "
            "KKT test, the cycles within the episode (evaluation, jumps and fast-forwards), and the "
            "history, the limit written into the rows after a settlement included; C ms "
            "per call; C us per episode, inferred as (call - fixed call cost - Dykstra cycle x exact "
            "cycles) / episodes; that cost in C Dykstra cycles, the break-even; cycles covered per "
            "episode beyond its exact cycle\n"
            f"% {counts}\n")


def write_csv(rows: list[dict], name: str) -> str:
    """Write rows as CSV in the output folder and return the path."""
    os.makedirs(paper_figures.OUTPUT_DIR, exist_ok=True)
    path = os.path.join(paper_figures.OUTPUT_DIR, name + ".csv")
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote {path}")
    return path


def print_method() -> None:
    """Print the machine and the timing method."""
    for key, value in performance_timing.machine_description().items():
        print(f"{key}: {value}")
    print(f"load averages: {', '.join(f'{load:.2f}' for load in os.getloadavg())}")
    print(f"method: every time the median of at least {REPEATS} runs, continued until they total "
          f"{REPEAT_SECONDS} s or number {MAX_REPEATS}; Dykstra cycles timed over "
          f"{PYTHON_DYKSTRA_CYCLES} cycles in Python and as the difference of {C_DYKSTRA_CYCLES} "
          f"cycles and one in C; accelerated runs to settlement or {CAP} cycles, Python in solve() "
          f"alone and profiled in alternation with an unprofiled solve, C as whole calls")


def report(records: list[dict], tests: list[dict], command: str) -> list[dict]:
    """Write every output, and print the rows, the fits and the break-even of each row."""
    summary = summarise(records)
    fits = fit_stages(records)
    write_csv(records, "cost_model_problems")
    write_csv(tests, "cost_model_kkt")
    write_csv(summary, "cost_model")
    write_csv(fits, "cost_model_fits")
    text = table_header(summary) + table_rows(summary)
    save_tex(text, "tab-cost-model", command)
    print(text)
    for fit in fits:
        print(f"fit {fit['stage']}: {fit['ns_per_flop']:.3g} ns per flop ({fit['gflops']:.3g} "
              f"GFlop/s) and {fit['us_per_count']:.3g} us per {fit['count']}, median relative "
              f"residual {fit['median_relative_residual']:.2f}, largest "
              f"{fit['largest_relative_residual']:.2f}, over {fit['problems']} problems")
    for row in summary:
        name = (f"{row['n']} x {row['p']}" if row["family"] == "tier"
                else f"wedge p = {row['p']}, theta = {row['theta']:g}")
        print(f"{name}: an episode costs {row['c_us_per_episode']:.3g} us in C, "
              f"{row['c_breakeven_cycles']:.3g} C Dykstra cycles, and "
              f"{row['python_us_per_episode']:.3g} us in Python, "
              f"{row['python_breakeven_cycles']:.3g} Python Dykstra cycles, of which its entry "
              f"takes {row['python_entry_breakeven_cycles']:.3g}; it covers "
              f"{row['cycles_per_episode']:.3g} cycles beyond its exact cycle, and pays in C on "
              f"{row['episodes_paying_c']} of {row['problems']} problems. Runs to settlement beat "
              f"Dykstra to the tolerance on {row['runs_paying_c']} problems in C and "
              f"{row['runs_paying_python']} in Python, median speedups "
              f"{row['c_speedup']:.3g} and {row['python_speedup']:.3g}, and Dykstra to squared error "
              f"{TIGHT_TOLERANCE:g} on {row['runs_paying_c_tight']} in C, median speedups "
              f"{row['c_speedup_tight']:.3g} and {row['python_speedup_tight']:.3g}, with Dykstra at "
              f"a median of {row['dykstra_cycles_to_tolerance']:.0f} and "
              f"{row['dykstra_cycles_to_tight']:.0f} cycles, "
              f"{row['dykstra_reached_tight']} of {row['problems']} within {CAP}")
    return summary


def read_csv(name: str) -> list[dict]:
    """Rows written by write_csv, with their numbers and flags restored."""
    def value(text: str):
        if text in ("True", "False"):
            return text == "True"
        for kind in (int, float):
            try:
                return kind(text)
            except ValueError:
                pass
        return text

    with open(os.path.join(paper_figures.OUTPUT_DIR, name + ".csv"), newline="",
              encoding="utf-8") as handle:
        return [{key: value(text) for key, text in row.items()} for row in csv.DictReader(handle)]


def warm_up() -> None:
    """Load the C libraries and run every solver once, so that no timing includes a first call."""
    A, b, z = problem(12, 4, problem_seed(12, 0))
    c_solvers.dykstra_c_projection(z, A, b, 2)
    c_solvers.accelerated_c_projection(z, A, b, 2)
    DykstraProjectionSolver(z, A, b, 2).solve()
    LTIVer5Solver(z, A, b, 2, profile=True).solve()


def run(problems: list, command: str) -> list[dict]:
    """Measure every problem, printing each as it finishes, and write the outputs."""
    print_method()
    warm_up()
    records, tests = [], []
    for arguments in problems:
        started = time.perf_counter()
        record, problem_tests = measure_problem(*arguments)
        records.append(record)
        tests += problem_tests
        print(f"{arguments[0]} {record['n']} x {record['p']}"
              + (f" theta {record['theta']:g}" if arguments[0] == "wedge" else "")
              + f" index {record['index']}: settled at {record['python_settled_at']}, "
              f"{record['count_episodes']} episodes, Python {record['python_seconds']:.3g} s, C "
              f"{record['c_seconds']:.3g} s ({time.perf_counter() - started:.1f} s)", flush=True)
    return report(records, tests, command)


def main() -> None:
    """Parse the command line and run."""
    global REPEATS, REPEAT_SECONDS
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--quick", action="store_true",
                        help="two small tiers and two wedge cells, one problem each and one run "
                             "per time, written to a temporary directory")
    parser.add_argument("--from-csv", action="store_true",
                        help="rebuild the outputs from results/paper/cost_model_problems.csv and "
                             "cost_model_kkt.csv")
    args = parser.parse_args()
    if not c_solvers.libraries_built():
        sys.exit(f"cost_model needs the C solvers: build them with make -C {c_solvers.C_FOLDER}")
    quiet_accelerate_warnings()
    command = " ".join(["python -m dykstra.cost_model", *sys.argv[1:]])
    began = time.perf_counter()
    if args.quick:
        REPEATS, REPEAT_SECONDS = 1, 0.0
        paper_figures.OUTPUT_DIR = tempfile.mkdtemp(prefix="cost_model_")
        run(problem_list(QUICK_TIERS, QUICK_SEEDS, QUICK_WEDGE_CELLS, QUICK_WEDGE_PROBLEMS), command)
    elif args.from_csv:
        print_method()
        report(read_csv("cost_model_problems"), read_csv("cost_model_kkt"), command)
    else:
        cells = tuple((p, theta) for p in WEDGE_DIMENSIONS for theta in WEDGE_ANGLES)
        run(problem_list(TIERS, SEEDS, cells, WEDGE_PROBLEMS), command)
    print(f"Finished in {time.perf_counter() - began:.0f} s: tab-cost-model.tex, cost_model.csv, "
          f"cost_model_problems.csv, cost_model_kkt.csv and cost_model_fits.csv in "
          f"{paper_figures.OUTPUT_DIR}")


if __name__ == "__main__":
    main()
