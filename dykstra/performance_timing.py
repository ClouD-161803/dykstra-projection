"""Time Dykstra's algorithm and the accelerated solver to a fixed accuracy, the protocol
behind the SIAM paper's tab:dykvsacc and the right panel of fig:performance.

Run from the repository root with `python -m dykstra.performance_timing`, or with `--tiers
small` and `--seeds 2` for a short run, or with `--quick` for a shorter one written to a
temporary directory, or with `--from-csv` to rebuild the outputs from the records of the last
run without timing again. Every problem is a Generator 2 polyhedron from
problem_generators with its infeasible start, problems 0 to seeds - 1 of each size tier.
`--family wedge` times the wedge family of problem_generators.WedgeTestPlanes instead, in the
cells of WEDGE_CELLS. `--schedule` sets where the accelerated solver tries the KKT test, in
Python and in C alike, and a run of another family or schedule than the paper's writes its
outputs under names that end in them. `--compare-schedules` sets the records of the runs of
the schedules work, every and doubling side by side.

For each problem the first cycle at which an iterate is within TOLERANCE in squared
distance of the quadprog projection is read off the error history the Python solvers
track, which gives the first crossing even where the error is not monotone. The
accelerated solver follows Dykstra's path until it settles, hence its first cycle is
Dykstra's or its settlement, whichever is earlier. Each solver is then timed on a fresh run
cut at that cycle. The certified column times the accelerated solver on a fresh run cut at its
settle cycle plus one, which an untimed run of the same implementation within CAP cycles finds.
After a settlement the Python solver writes the limit into every remaining row of its history,
which is not the algorithm's work. The C solver returns at its settlement and does the same
work under either budget. A problem that misses the tolerance or does not settle within CAP is
counted at CAP cycles and the time of a CAP-cycle run. The C solvers are timed as whole calls.
The Python solvers are timed in solve() alone, because their constructor runs the quadprog
reference solve. Each time is the median of repeated runs, repeated until they total
REPEAT_SECONDS or number REPEATS. Probes, the reference solve and the error checks are not
timed.

Every cell of a table reads the median over the problems of a tier and, in brackets, the
first and third quartiles, interpolated linearly between the order statistics, and the scaling
panel draws the same quartiles as bars. Every record names its implementation and schedule.
Results go to results/paper at the repository root.
"""

import argparse
import csv
import os
import platform
import statistics
import subprocess
import sys
import tempfile
import time
from collections import namedtuple

import numpy as np
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt

from dykstra import c_solvers, paper_figures
from dykstra.convex_projection_solver import ConvexProjectionSolver, DykstraProjectionSolver
from dykstra.lti_numerics import DEFAULT_SCHEDULE
from dykstra.lti_solver import LTIVer5Solver
from dykstra.paper_figures import (
    INK,
    RED,
    TEAL,
    quiet_accelerate_warnings,
    save_figure,
    save_tex,
    style_axes,
)
from dykstra.problem_generators import WedgeTestPlanes, problem, problem_seed


# Size tiers as (half-spaces, dimensions). TABLE_TIERS are the rows of tab:dykvsacc
TABLE_TIERS = ((12, 4), (24, 8), (48, 16), (96, 32), (160, 48), (252, 64))
SMALL_TIERS = ((12, 4), (24, 8))
SEEDS = 20

# The wedge family's cells as (dimensions, angle), the tiers and angles of benchmark.py, and its
# problems per cell, chosen here. Problem i of the k-th angle in p dimensions has the seed
# WEDGE_SEED + 1000 p + 100 k + i of benchmark.py's block, hence a cell's first five problems
# are the benchmark's
WEDGE_CELLS = ((3, 0.1), (3, 0.03), (3, 0.01), (8, 0.1), (8, 0.03), (8, 0.01))
WEDGE_ANGLES = (0.1, 0.03, 0.01)
WEDGE_PROBLEMS = 20
WEDGE_SEED = 5_000_000

TOLERANCE = 1e-3
CAP = 20000
MAX_CYCLES = CAP
REPEAT_SECONDS, REPEATS = 0.5, 5

# The schedules --compare-schedules sets side by side, the paper's first
COMPARED_SCHEDULES = ("work", "every", "doubling")

# Columns of tab:dykvsacc as (name, solver timed, colour, marker)
SOLVERS = (("Dykstra", "dykstra", INK, "o"),
           ("accelerated", "accelerated", RED, "s"),
           ("accelerated, certified", "accelerated", TEAL, "^"))

# Small counts are spelled out in the paper's prose
COUNTS = {2: "two", 3: "three", 4: "four", 5: "five", 6: "six", 7: "seven", 8: "eight",
          9: "nine"}

PYTHON_SOLVERS = {"dykstra": DykstraProjectionSolver, "accelerated": LTIVer5Solver}
C_SOLVERS = {"dykstra": c_solvers.dykstra_c_projection,
             "accelerated": c_solvers.accelerated_c_projection}
IMPLEMENTATION_NAMES = {"c": "C", "python": "Python"}

# One problem to time: its family, its tier's key and label, and how to draw it
Problem = namedtuple("Problem", "family key label num_planes num_dimensions theta index seed")


def dense_problems(tiers: tuple, seeds: int) -> list[Problem]:
    """Problems 0 to seeds - 1 of each Generator 2 size tier."""
    return [Problem("dense", (n, p), f"${n} \\times {p}$", n, p, 0.0, index, problem_seed(n, index))
            for n, p in tiers for index in range(seeds)]


def wedge_problems(cells: tuple, problems: int) -> list[Problem]:
    """Problems 0 to problems - 1 of each wedge cell."""
    return [Problem("wedge", (p, theta), f"$p = {p}$, $\\theta = {theta:g}$", 2 * p - 2, p, theta,
                    index, WEDGE_SEED + 1000 * p + 100 * WEDGE_ANGLES.index(theta) + index)
            for p, theta in cells for index in range(problems)]


def problem_data(spec: Problem) -> tuple:
    """The constraints and start of a problem, as (A, b, z)."""
    if spec.family == "wedge":
        return WedgeTestPlanes(spec.theta, spec.num_dimensions).problem(spec.seed)
    return problem(spec.num_planes, spec.num_dimensions, spec.seed)


def output_name(base: str, family: str = "dense", schedule: str = DEFAULT_SCHEDULE) -> str:
    """base for the paper's run, else base followed by the run's family and schedule."""
    return (base + ("" if family == "dense" else f"-{family}")
            + ("" if schedule == DEFAULT_SCHEDULE else f"-{schedule}"))


def machine_description() -> dict:
    """The machine, compiler, and package versions behind a timing."""
    import importlib.metadata as metadata

    chip = platform.processor()
    if sys.platform == "darwin":
        chip = subprocess.run(["sysctl", "-n", "machdep.cpu.brand_string"],
                              capture_output=True, text=True).stdout.strip() or chip
        system = f"macOS {platform.mac_ver()[0]} ({platform.machine()})"
    else:
        system = f"{platform.system()} {platform.release()} ({platform.machine()})"
    try:
        compiler = subprocess.run(["cc", "--version"], capture_output=True,
                                  text=True).stdout.splitlines()[0]
    except (OSError, IndexError):
        compiler = "unknown"
    return {"machine": chip, "system": system, "python": platform.python_version(),
            "numpy": metadata.version("numpy"), "quadprog": metadata.version("quadprog"),
            "C compiler": compiler,
            "C flags": "-std=c11 -O2 -Wall -Wextra -fPIC -ffp-contract=off (C/Makefile)"}


def print_method(schedule: str = DEFAULT_SCHEDULE) -> None:
    """Print the machine and the timing method."""
    for key, value in machine_description().items():
        print(f"{key}: {value}")
    print(f"method: first cycle within squared error {TOLERANCE:g} from the tracked history, "
          f"budget cap {CAP} cycles, the certified run cut at the settle cycle plus one of an "
          f"untimed run within the cap, each time the median of runs repeated until they total "
          f"{REPEAT_SECONDS} s or number {REPEATS}; C timed as whole calls, Python in solve() "
          f"alone; the accelerated solver tries the KKT test under the schedule {schedule}")


def cycles_to_tolerance(solver_type: type, z: np.ndarray, A: np.ndarray, b: np.ndarray,
                        dimensions: int, cap: int = CAP, **options) -> int | None:
    """First cycle whose iterate is within TOLERANCE, or None if none is within cap. options go
    to the solver's constructor."""
    budget = min(64, cap)
    while True:
        solver = solver_type(z, A, b, budget, track_error=True,
                             min_error=TOLERANCE, dimensions=dimensions, **options)
        errors = np.asarray(solver.solve().squared_errors, dtype=float)
        reached = np.flatnonzero(errors <= TOLERANCE)
        if reached.size:
            return int(reached[0])
        if budget >= cap:
            return None
        budget = min(2 * budget, cap)


def timed_run(implementation: str, solver: str, z: np.ndarray, A: np.ndarray, b: np.ndarray,
              budget: int, schedule: str = DEFAULT_SCHEDULE) -> tuple[float, np.ndarray, int | None, dict]:
    """One timed solve, returning seconds, the point, the first settled cycle or None, and
    the accelerated solver's counts: its KKT tests, and in C its exact and skipped cycles."""
    options = {"schedule": schedule} if solver == "accelerated" else {}
    if implementation == "python":
        instance = PYTHON_SOLVERS[solver](z, A, b, budget, **options)
        start = time.perf_counter()
        result = instance.solve()
        seconds = time.perf_counter() - start
        settled = result.settled_at if result.certificate == "kkt" else None
        counters = ({"KKT attempts": instance.kkt_record["kkt_attempts"],
                     "entry tests": instance.kkt_record["entry_tests"]}
                    if solver == "accelerated" else {})
        return seconds, result.projection, settled, counters
    start = time.perf_counter()
    result = C_SOLVERS[solver](z, A, b, budget, **options)
    seconds = time.perf_counter() - start
    settled = result.settled_at if solver == "accelerated" and result.settled else None
    counters = ({"KKT attempts": result.kkt_attempts, "entry tests": result.entry_tests,
                 "exact cycles": result.exact_cycles, "cycles skipped": result.cycles_skipped}
                if solver == "accelerated" else {})
    return seconds, result.projection, settled, counters


def median_seconds(implementation: str, solver: str, z: np.ndarray, A: np.ndarray,
                   b: np.ndarray, budget: int, schedule: str = DEFAULT_SCHEDULE
                   ) -> tuple[float, int, np.ndarray, int | None, dict]:
    """Median time of repeated runs, with their count and the last run's other results."""
    timings = []
    while not timings or (sum(timings) < REPEAT_SECONDS and len(timings) < REPEATS):
        seconds, point, settled, counters = timed_run(implementation, solver, z, A, b, budget,
                                                      schedule)
        timings.append(seconds)
    return statistics.median(timings), len(timings), point, settled, counters


def measure_problem(spec: Problem, implementations: tuple[str, ...],
                    schedule: str = DEFAULT_SCHEDULE) -> list[dict]:
    """Probe and time one problem, one record per implementation asked for. The certified
    column is timed at the settle cycle plus one of an untimed run of the same implementation
    within CAP, and a timed run that does not settle on that cycle raises RuntimeError."""
    A, b, z = problem_data(spec)
    reference = ConvexProjectionSolver._find_optimal_solution(z, A, b)
    first_dykstra = cycles_to_tolerance(DykstraProjectionSolver, z, A, b, spec.num_dimensions)
    first_accelerated = cycles_to_tolerance(LTIVer5Solver, z, A, b, spec.num_dimensions,
                                            cap=first_dykstra or CAP, schedule=schedule)
    budgets = {"Dykstra": first_dykstra or CAP, "accelerated": first_accelerated or CAP}
    records = []
    for implementation in implementations:
        # the two implementations can settle on different cycles, e.g. at a tie in the rounding
        # floor of the finality test, hence each is cut after a settle cycle of its own
        settle = timed_run(implementation, "accelerated", z, A, b, CAP, schedule)[2]
        budgets["accelerated, certified"] = CAP if settle is None else settle + 1
        record = {"family": spec.family,
                  "tier": (f"{spec.num_planes}x{spec.num_dimensions}" if spec.family == "dense"
                           else f"p={spec.num_dimensions} theta={spec.theta:g}"),
                  "num_planes": spec.num_planes, "num_dimensions": spec.num_dimensions,
                  "theta": spec.theta, "index": spec.index, "seed": spec.seed,
                  "implementation": implementation, "schedule": schedule,
                  "Dykstra reached": first_dykstra is not None,
                  "accelerated reached": first_accelerated is not None}
        for name, solver, _, _ in SOLVERS:
            seconds, runs, point, settled, counters = median_seconds(
                implementation, solver, z, A, b, budgets[name], schedule)
            if name == "accelerated, certified":
                if settled != settle:
                    raise RuntimeError(
                        f"{record['tier']} index {spec.index}: the "
                        f"{IMPLEMENTATION_NAMES[implementation]} accelerated solver settles at "
                        f"cycle {settled} within {budgets[name]} cycles, and the untimed run "
                        f"at cycle {settle} within {CAP}")
                record[f"{name} reached"] = settled is not None
                record[f"{name} cycles"] = settled if settled is not None else CAP
            else:
                record[f"{name} cycles"] = budgets[name]
            record[f"{name} seconds"] = seconds
            record[f"{name} runs"] = runs
            record[f"{name} squared error"] = float(np.sum((point - reference) ** 2))
            record.update({f"{name} {counter}": count for counter, count in counters.items()})
        records.append(record)
    return records


def quartiles(values: list[float]) -> tuple[float, float, float]:
    """The first quartile, the median and the third quartile, interpolated linearly."""
    first, median, third = np.percentile(np.asarray(values, dtype=float), [25, 50, 75])
    return float(first), float(median), float(third)


def summarise(records: list[dict], implementation: str) -> dict:
    """Medians and quartiles over the problems of a tier, per column, with the count that
    reached the column's target and, for the certified column, the KKT tests per run."""
    records = [r for r in records if r["implementation"] == implementation]
    summary = {}
    for name, _, _, _ in SOLVERS:
        cycles = quartiles([r[f"{name} cycles"] for r in records])
        seconds = quartiles([r[f"{name} seconds"] for r in records])
        summary[name] = {"cycles": cycles[1], "cycles quartiles": (cycles[0], cycles[2]),
                         "seconds": seconds[1], "seconds quartiles": (seconds[0], seconds[2]),
                         "reached": sum(bool(r[f"{name} reached"]) for r in records),
                         "seeds": len(records)}
    attempts = [r["accelerated, certified KKT attempts"] for r in records
                if r.get("accelerated, certified KKT attempts", "") != ""]
    if attempts:
        summary["accelerated, certified"]["KKT attempts"] = quartiles(attempts)
    return summary


def time_tier(num_planes: int, num_dimensions: int, seeds: int,
              implementation: str = "python", schedule: str = DEFAULT_SCHEDULE) -> dict:
    """Medians for one size tier in one implementation."""
    records = [record for spec in dense_problems(((num_planes, num_dimensions),), seeds)
               for record in measure_problem(spec, (implementation,), schedule)]
    return summarise(records, implementation)


def format_ms(seconds: float) -> str:
    """Milliseconds to three significant figures, whole above 100."""
    ms = 1000.0 * seconds
    return f"{ms:.0f}" if ms >= 100 else f"{ms:.3g}"


def format_cycles(cycles: float) -> str:
    """A cycle count, with a half where the median falls between two problems."""
    return f"{cycles:.0f}" if float(cycles).is_integer() else f"{cycles:.1f}"


def table_rows(summaries: dict) -> str:
    """Rows of tab:dykvsacc in the paper's tabular format, one per tier, every cell the median
    and in brackets the first and third quartiles."""
    lines = []
    for key, summary in summaries.items():
        cells = []
        for name, _, _, _ in SOLVERS:
            column = summary[name]
            (low, high), (fast, slow) = column["cycles quartiles"], column["seconds quartiles"]
            cells.append(f"${format_cycles(column['cycles'])}$ [${format_cycles(low)}$, "
                         f"${format_cycles(high)}$] & ${format_ms(column['seconds'])}$ "
                         f"[${format_ms(fast)}$, ${format_ms(slow)}$] ms")
        lines.append(f"  {summary.get('label', key):<16} & " + " & ".join(cells) + r" \\")
    return "\n".join(lines) + "\n"


def reach_comment(summaries: dict) -> str:
    """A LaTeX comment per tier giving how many problems reached each column's target and the
    KKT tests of the certified runs."""
    lines = []
    for key, summary in summaries.items():
        counts = ", ".join(f"{name} {summary[name]['reached']}/{summary[name]['seeds']}"
                           for name, _, _, _ in SOLVERS)
        attempts = summary["accelerated, certified"].get("KKT attempts")
        tests = (f"; KKT tests per certified run {format_cycles(attempts[1])} "
                 f"[{format_cycles(attempts[0])}, {format_cycles(attempts[2])}]" if attempts else "")
        lines.append(f"% {summary.get('tier', key)}: {counts}{tests}")
    return "\n".join(lines) + "\n"


def scaling_fit(summaries: dict, name: str) -> tuple[list[int], list[float], float]:
    """Sizes, median seconds to tolerance, and the log-log least-squares gradient."""
    sizes = [n for (n, _) in summaries]
    seconds = [summary[name]["seconds"] for summary in summaries.values()]
    gradient = float(np.polyfit(np.log(sizes), np.log(seconds), 1)[0])
    return sizes, seconds, gradient


def scaling_pgfplots(summaries: dict) -> str:
    """The right panel of fig:performance as pgfplots, with bars from the first to the third
    quartile."""
    plots = []
    for name, colour, mark in (("Dykstra", "figink", "*"), ("accelerated", "figred", "square*")):
        sizes, seconds, gradient = scaling_fit(summaries, name)
        bars = [summary[name]["seconds quartiles"] for summary in summaries.values()]
        points = " ".join(f"({n},{s:.4g}) += (0,{high - s:.4g}) -= (0,{s - low:.4g})"
                          for n, s, (low, high) in zip(sizes, seconds, bars))
        plots.append(f"      \\addplot[{colour}, thick, mark={mark}, mark size=1.5pt, "
                     f"error bars/.cd, y dir=both, y explicit] coordinates {{{points}}};\n"
                     f"      \\addlegendentry{{{name} $(\\sim n^{{{gradient:.2f}}})$}}\n")
    sizes = [n for (n, _) in summaries]
    lows = [s[name]["seconds quartiles"][0] for s in summaries.values()
            for name in ("Dykstra", "accelerated")]
    highs = [s[name]["seconds quartiles"][1] for s in summaries.values()
             for name in ("Dykstra", "accelerated")]
    ticks = ",".join(str(n) for n in sizes)
    labels = ",".join(f"${n}$" for n in sizes)
    return rf"""  \begin{{tikzpicture}}
    \begin{{axis}}[paperaxis, width=0.47\linewidth, height=0.40\linewidth,
                 xmode=log, ymode=log, log basis x=10,
                 xmin={0.8 * min(sizes):.3g}, xmax={1.25 * max(sizes):.3g}, ymin={0.5 * min(lows):.2g}, ymax={2.0 * max(highs):.2g},
                 xtick={{{ticks}}}, xticklabels={{{labels}}},
                 xlabel={{number of half-spaces $n$}},
                 ylabel={{median time [s]}},
                 legend pos=north west]
{"".join(plots)}    \end{{axis}}
  \end{{tikzpicture}}
"""


def performance_scaling_figure(summaries: dict, name: str = "performance_scaling") -> None:
    """Draw median time to tolerance against size, with bars from the first to the third
    quartile."""
    fig, ax = plt.subplots(figsize=(6.2, 4.4))
    for column, colour, marker in (("Dykstra", INK, "o"), ("accelerated", RED, "s")):
        sizes, seconds, gradient = scaling_fit(summaries, column)
        bars = np.array([summary[column]["seconds quartiles"] for summary in summaries.values()])
        ax.errorbar(sizes, seconds, yerr=[np.subtract(seconds, bars[:, 0]), bars[:, 1] - seconds],
                    marker=marker, color=colour, lw=1.6, ms=5, capsize=2.5, elinewidth=1.0,
                    label=rf"{column}  ($\sim n^{{{gradient:.2f}}}$)")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("number of half-spaces $n$")
    ax.set_ylabel(rf"median time to squared error ${TOLERANCE:g}$ in s")
    ax.legend(fontsize=9, frameon=False, loc="upper left")
    style_axes(ax)
    fig.tight_layout()
    save_figure(fig, name)


def write_records(records: list[dict], name: str) -> str:
    """Write every per-problem record as CSV and return the path."""
    os.makedirs(paper_figures.OUTPUT_DIR, exist_ok=True)
    path = os.path.join(paper_figures.OUTPUT_DIR, name + ".csv")
    # the C records carry counters the Python ones do not
    fields = list(dict.fromkeys(key for record in records for key in record))
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(records)
    print(f"Wrote {path}")
    return path


def performance_figure(summaries: dict, curves: dict, seeds: int) -> str:
    """The whole of fig:performance: the error curves on the left, the scaling on the right."""
    tiers = ", ".join(f"{n} x {p}" for n, p in summaries)
    left = paper_figures.performance_error_pgfplots(curves, paper_figures.PERFORMANCE_CYCLES)
    num_planes, num_dimensions = paper_figures.PERFORMANCE_SIZE
    dykstra = (f"Dykstra reaches the tolerance at cycle {curves['first_dykstra']}"
               if curves["first_dykstra"] is not None else
               f"Dykstra stays above the tolerance through cycle "
               f"{paper_figures.PERFORMANCE_CYCLES}")
    return (f"% right panel from the tiers {tiers}\n"
            "\\begin{figure}[htbp]\n  \\centering\n" + left + "  \\hfill\n"
            + scaling_pgfplots(summaries)
            + f"  \\caption{{Left: squared error against cycle on one ${num_planes} \\times "
            f"{num_dimensions}$ problem, where the accelerated solver follows the Dykstra path "
            f"up to rounding and then certifies the projection at cycle {curves['settle']}. "
            f"{dykstra}. Right: median time to squared error $10^{{-3}}$ against the number of "
            f"half-spaces, over the {COUNTS.get(len(summaries), len(summaries))} size tiers of "
            f"\\cref{{tab:dykvsacc}} with {seeds} problems each, the bars running from the first "
            "to the third quartile, and the fitted growth order in the legend.}%\n"
            "  \\label{fig:performance}\n\\end{figure}\n")


def measure(specs: list[Problem], implementations: tuple[str, ...],
            schedule: str = DEFAULT_SCHEDULE) -> list[dict]:
    """Probe and time every problem, printing each as it finishes, and return the records. The
    records file of the family and schedule keeps the records of every other implementation it
    holds, which --from-csv and --compare-schedules read beside the new ones."""
    records = []
    for spec in specs:
        started = time.perf_counter()
        measured = measure_problem(spec, implementations, schedule)
        records.extend(measured)
        for record in measured:
            cells = ", ".join(f"{key} {record[key]}" for key in record if key.endswith("cycles"))
            print(f"{record['tier']} index {spec.index}, {record['implementation']}: {cells} "
                  f"({time.perf_counter() - started:.1f} s)", flush=True)
    family = specs[0].family if specs else "dense"
    name = output_name("performance_timing", family, schedule)
    path = os.path.join(paper_figures.OUTPUT_DIR, name + ".csv")
    kept = ([r for r in read_records(path) if r["implementation"] not in implementations]
            if os.path.exists(path) else [])
    write_records(kept + records, name)
    return records


def read_records(path: str) -> list[dict]:
    """Records written by write_records, with their numbers and flags restored."""
    def value(text: str):
        if text in ("True", "False"):
            return text == "True"
        for kind in (int, float):
            try:
                return kind(text)
            except ValueError:
                pass
        return text

    with open(path, newline="", encoding="utf-8") as handle:
        return [{key: value(text) for key, text in row.items()} for row in csv.DictReader(handle)]


def tier_keys(records: list[dict]) -> dict:
    """The tiers of the records in order, keyed as the summaries are, with their labels."""
    tiers = {}
    for r in records:
        if r.get("family", "dense") == "wedge":
            key = (r["num_dimensions"], float(r["theta"]))
            label = f"$p = {key[0]}$, $\\theta = {key[1]:g}$"
        else:
            key = (r["num_planes"], r["num_dimensions"])
            label = f"${key[0]} \\times {key[1]}$"
        tiers.setdefault(key, (label, r["tier"]))
    return tiers


def summaries_by_tier(records: list[dict], implementation: str) -> dict:
    """The summary of every tier of the records in one implementation."""
    summaries = {}
    for key, (label, tier) in tier_keys(records).items():
        summary = summarise([r for r in records if r["tier"] == tier], implementation)
        summary["label"], summary["tier"] = label, tier
        summaries[key] = summary
    return summaries


def report(records: list[dict], implementations: tuple[str, ...], command: str,
           full_figure: bool = False) -> dict:
    """Write the table, the scaling panel and, if asked, the whole figure per implementation."""
    family = records[0].get("family", "dense")
    schedule = records[0].get("schedule", DEFAULT_SCHEDULE)
    seeds = max(sum(r["tier"] == tier for r in records if r["implementation"] == implementations[0])
                for _, tier in tier_keys(records).values())
    curves = None
    if full_figure and family == "dense" and schedule == DEFAULT_SCHEDULE:
        num_planes, num_dimensions = paper_figures.PERFORMANCE_SIZE
        curves = paper_figures.error_curves(num_planes, num_dimensions,
                                            paper_figures.PERFORMANCE_INDEX,
                                            paper_figures.PERFORMANCE_CYCLES)
    summaries = {}
    for implementation in implementations:
        summaries[implementation] = summaries_by_tier(records, implementation)
        rows = table_rows(summaries[implementation])
        header = (f"% {IMPLEMENTATION_NAMES[implementation]} implementation, "
                  f"{'the wedge family' if family == 'wedge' else 'Generator 2'}, {seeds} problems "
                  f"per tier, budget cap {CAP} cycles, squared error {TOLERANCE:g}, KKT test "
                  f"schedule {schedule}\n"
                  "% the certified column times a fresh run cut at the settle cycle plus one, which "
                  f"an untimed run within the cap finds, or a run of {CAP} cycles for a problem "
                  "that does not settle within the cap\n"
                  "% each cell reads the median over the problems of the tier and, in brackets, the "
                  "first and third quartiles, interpolated linearly between order statistics\n"
                  + reach_comment(summaries[implementation]))
        print(header + rows)
        save_tex(header + rows, output_name(f"tab-dykvsacc-{implementation}", family, schedule),
                 command)
        if family == "dense" and len(summaries[implementation]) >= 2:
            save_tex(scaling_pgfplots(summaries[implementation]),
                     output_name(f"fig-performance-right-{implementation}", family, schedule),
                     command)
            performance_scaling_figure(summaries[implementation],
                                       output_name(f"performance_scaling_{implementation}",
                                                   family, schedule))
            for column in ("Dykstra", "accelerated", "accelerated, certified"):
                print(f"{implementation} {column}: time grows as "
                      f"n^{scaling_fit(summaries[implementation], column)[2]:.2f}")
            if curves is not None:
                save_tex(performance_figure(summaries[implementation], curves, seeds),
                         f"fig-performance-{implementation}", command)
    return summaries


def compare_schedules(command: str) -> list[dict]:
    """Set the runs of COMPARED_SCHEDULES side by side, from their records in results/paper, and
    write the rows as kkt_schedules.csv and, for C, tab-kkt-schedules.tex."""
    rows = []
    for family in ("dense", "wedge"):
        runs = {}
        for schedule in COMPARED_SCHEDULES:
            path = os.path.join(paper_figures.OUTPUT_DIR,
                                output_name("performance_timing", family, schedule) + ".csv")
            if os.path.exists(path):
                runs[schedule] = read_records(path)
        if not runs:
            continue
        for implementation in IMPLEMENTATION_NAMES:
            for schedule, records in runs.items():
                if not any(r["implementation"] == implementation for r in records):
                    continue
                summaries = summaries_by_tier(records, implementation)
                exponents = ({column: scaling_fit(summaries, column)[2]
                              for column in ("accelerated", "accelerated, certified")}
                             if family == "dense" else {})
                for key, summary in summaries.items():
                    certified, reached = summary["accelerated, certified"], summary["accelerated"]
                    attempts = certified.get("KKT attempts", (np.nan,) * 3)
                    rows.append({
                        "family": family, "implementation": implementation, "schedule": schedule,
                        "tier": summary["tier"], "label": summary["label"],
                        "settle cycles": certified["cycles"],
                        "settle cycles q1": certified["cycles quartiles"][0],
                        "settle cycles q3": certified["cycles quartiles"][1],
                        "settled": certified["reached"], "problems": certified["seeds"],
                        "seconds to settle": certified["seconds"],
                        "seconds to settle q1": certified["seconds quartiles"][0],
                        "seconds to settle q3": certified["seconds quartiles"][1],
                        "cycles to tolerance": reached["cycles"],
                        "seconds to tolerance": reached["seconds"],
                        "seconds to tolerance q1": reached["seconds quartiles"][0],
                        "seconds to tolerance q3": reached["seconds quartiles"][1],
                        "KKT attempts": attempts[1], "KKT attempts q1": attempts[0],
                        "KKT attempts q3": attempts[2],
                        "exponent to tolerance": exponents.get("accelerated", np.nan),
                        "exponent to settle": exponents.get("accelerated, certified", np.nan)})
    if not rows:
        raise FileNotFoundError(f"No records of the schedules in {paper_figures.OUTPUT_DIR}.")
    write_records(rows, "kkt_schedules")
    lines = ["% the C implementation under each schedule of the KKT test, per tier and wedge cell: "
             "the median cycle of settlement and its time, the median time to squared error "
             f"{TOLERANCE:g}, and the median KKT tests per certified run, each with its first and "
             "third quartiles in brackets\n"]
    for row in rows:
        if row["implementation"] != "c":
            continue
        print(f"{row['family']} {row['tier']} {row['schedule']}: settles at "
              f"{format_cycles(row['settle cycles'])} [{format_cycles(row['settle cycles q1'])}, "
              f"{format_cycles(row['settle cycles q3'])}] ({row['settled']}/{row['problems']}) in "
              f"{format_ms(row['seconds to settle'])} ms, to tolerance in "
              f"{format_ms(row['seconds to tolerance'])} ms, KKT tests {row['KKT attempts']:g}")
        lines.append(f"  {row['label']:<24} & {row['schedule']:<8} & "
                     f"${format_cycles(row['settle cycles'])}$ [${format_cycles(row['settle cycles q1'])}$, "
                     f"${format_cycles(row['settle cycles q3'])}$] & "
                     f"${format_ms(row['seconds to settle'])}$ [${format_ms(row['seconds to settle q1'])}$, "
                     f"${format_ms(row['seconds to settle q3'])}$] ms & "
                     f"${format_ms(row['seconds to tolerance'])}$ [${format_ms(row['seconds to tolerance q1'])}$, "
                     f"${format_ms(row['seconds to tolerance q3'])}$] ms & "
                     f"${format_cycles(row['KKT attempts'])}$ \\\\\n")
    for row in rows:
        if row["tier"] == rows[0]["tier"] and row["family"] == "dense":
            print(f"{row['implementation']} {row['schedule']}: time to tolerance grows as "
                  f"n^{row['exponent to tolerance']:.2f}, to settlement as n^{row['exponent to settle']:.2f}")
    save_tex("".join(lines), "tab-kkt-schedules", command)
    return rows


def run(tiers: tuple = TABLE_TIERS, seeds: int = SEEDS,
        implementations: tuple[str, ...] = ("c", "python"), command: str | None = None,
        schedule: str = DEFAULT_SCHEDULE, family: str = "dense") -> dict:
    """Time every tier in every implementation, print and write the table and the scaling."""
    print_method(schedule)
    specs = dense_problems(tiers, seeds) if family == "dense" else wedge_problems(tiers, seeds)
    records = measure(specs, implementations, schedule)
    return report(records, implementations, command or "python -m dykstra.performance_timing",
                  full_figure=family == "dense" and tiers == TABLE_TIERS)


def main() -> None:
    """Parse the command line and run."""
    global REPEATS
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--tiers", choices=("table", "small"), default="table")
    parser.add_argument("--seeds", type=int, default=None,
                        help=f"problems per tier, {SEEDS} by default and {WEDGE_PROBLEMS} per wedge cell")
    parser.add_argument("--implementation", choices=("c", "python", "both"), default="both")
    parser.add_argument("--schedule", choices=COMPARED_SCHEDULES, default=DEFAULT_SCHEDULE,
                        help="where the accelerated solver tries the KKT test")
    parser.add_argument("--family", choices=("dense", "wedge"), default="dense")
    parser.add_argument("--from-csv", action="store_true",
                        help="rebuild the outputs from the run's records in results/paper")
    parser.add_argument("--compare-schedules", action="store_true",
                        help="set the records of the runs of the schedules side by side")
    parser.add_argument("--quick", action="store_true",
                        help="the small tiers or two wedge cells, one problem each and one timed "
                             "run per solver, written to a temporary directory")
    args = parser.parse_args()
    implementations = ("c", "python") if args.implementation == "both" else (args.implementation,)
    quiet_accelerate_warnings()
    command = " ".join(["python -m dykstra.performance_timing", *sys.argv[1:]])
    began = time.perf_counter()
    if args.compare_schedules:
        compare_schedules(command)
        print(f"Finished in {time.perf_counter() - began:.0f} s: kkt_schedules.csv and "
              f"tab-kkt-schedules.tex in {paper_figures.OUTPUT_DIR}")
        return
    if args.quick:
        REPEATS = 1
        paper_figures.OUTPUT_DIR = tempfile.mkdtemp(prefix="performance_timing_")
        tiers = SMALL_TIERS if args.family == "dense" else WEDGE_CELLS[:2]
        run(tiers, 1, implementations, command, args.schedule, args.family)
    elif args.from_csv:
        print_method(args.schedule)
        records = read_records(os.path.join(paper_figures.OUTPUT_DIR, output_name(
            "performance_timing", args.family, args.schedule) + ".csv"))
        report(records, implementations, command, full_figure=True)
    else:
        if args.family == "dense":
            tiers, seeds = (TABLE_TIERS if args.tiers == "table" else SMALL_TIERS), args.seeds or SEEDS
        else:
            tiers, seeds = WEDGE_CELLS, args.seeds or WEDGE_PROBLEMS
        run(tiers, seeds, implementations, command, args.schedule, args.family)
    print(f"Finished in {time.perf_counter() - began:.0f} s: "
          f"{output_name('performance_timing', args.family, args.schedule)}.csv, and per "
          f"implementation tab-dykvsacc, fig-performance-right and fig-performance as .tex and "
          f"performance_scaling as PNG and PDF, in {paper_figures.OUTPUT_DIR}")


if __name__ == "__main__":
    main()
