"""Time Dykstra's algorithm and the accelerated solver to a fixed accuracy, the protocol
behind the SIAM paper's tab:dykvsacc and the right panel of fig:performance.

Run from this folder with `python performance_timing.py`, or with `--tiers small` and
`--seeds 2` for a short run, or with `--from-csv` to rebuild the outputs from the records of
the last run without timing again. Every problem is a Generator 2 polyhedron from
problem_generators with its infeasible start, problems 0 to seeds - 1 of each size tier.

For each problem the first cycle at which an iterate is within TOLERANCE in squared
distance of the quadprog projection is read off the error history the Python solvers
track, which gives the first crossing even where the error is not monotone. The
accelerated solver follows Dykstra's path until it settles, hence its first cycle is
Dykstra's or its settlement, whichever is earlier. Each solver is then timed on a fresh run
cut at that cycle, and the certified column times the accelerated solver run until it
settles, all under a budget of CAP cycles. A problem that misses the tolerance or does not
settle within CAP is counted at CAP cycles and the time of a CAP-cycle run. The C solvers are
timed as whole calls. The Python solvers are timed in solve() alone, because their
constructor runs the quadprog reference solve. Each time is the median of repeated runs,
repeated until they total REPEAT_SECONDS or number REPEATS. Probes, the reference solve and
the error checks are not timed. Results go to results/paper beside this script.
"""

import argparse
import csv
import os
import platform
import statistics
import subprocess
import sys
import time

import numpy as np
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt

import c_solvers
import paper_figures
from convex_projection_solver import ConvexProjectionSolver, DykstraProjectionSolver
from lti_solver import LTIVer5Solver
from paper_figures import (
    INK,
    RED,
    TEAL,
    quiet_accelerate_warnings,
    save_figure,
    save_tex,
    style_axes,
)
from problem_generators import problem, problem_seed


# Size tiers as (half-spaces, dimensions). TABLE_TIERS are the rows of tab:dykvsacc
TABLE_TIERS = ((12, 4), (24, 8), (48, 16), (96, 32), (160, 48), (252, 64))
SMALL_TIERS = ((12, 4), (24, 8))
SEEDS = 5

TOLERANCE = 1e-3
CAP = 20000
MAX_CYCLES = CAP
REPEAT_SECONDS, REPEATS = 0.5, 5

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


def print_method() -> None:
    """Print the machine and the timing method."""
    for key, value in machine_description().items():
        print(f"{key}: {value}")
    print(f"method: first cycle within squared error {TOLERANCE:g} from the tracked history, "
          f"budget cap {CAP} cycles, each time the median of runs repeated until they total "
          f"{REPEAT_SECONDS} s or number {REPEATS}; C timed as whole calls, Python in solve() "
          f"alone")


def cycles_to_tolerance(solver_type: type, z: np.ndarray, A: np.ndarray, b: np.ndarray,
                        dimensions: int, cap: int = CAP) -> int | None:
    """First cycle whose iterate is within TOLERANCE, or None if none is within cap."""
    budget = min(64, cap)
    while True:
        solver = solver_type(z, A, b, budget, track_error=True,
                             min_error=TOLERANCE, dimensions=dimensions)
        errors = np.asarray(solver.solve().squared_errors, dtype=float)
        reached = np.flatnonzero(errors <= TOLERANCE)
        if reached.size:
            return int(reached[0])
        if budget >= cap:
            return None
        budget = min(2 * budget, cap)


def timed_run(implementation: str, solver: str, z: np.ndarray, A: np.ndarray, b: np.ndarray,
              budget: int) -> tuple[float, np.ndarray, int | None, dict]:
    """One timed solve, returning seconds, the point, the first settled cycle or None, and
    the C accelerated solver's counts of exact and skipped cycles."""
    if implementation == "python":
        instance = PYTHON_SOLVERS[solver](z, A, b, budget)
        start = time.perf_counter()
        result = instance.solve()
        seconds = time.perf_counter() - start
        settled = result.settled_at if result.certificate == "kkt" else None
        return seconds, result.projection, settled, {}
    start = time.perf_counter()
    result = C_SOLVERS[solver](z, A, b, budget)
    seconds = time.perf_counter() - start
    # The C solver counts the cycles it ran, and the settled point holds from the next one
    settled = result.cycles + 1 if solver == "accelerated" and result.settled else None
    counters = ({"exact cycles": result.exact_cycles, "cycles skipped": result.cycles_skipped}
                if solver == "accelerated" else {})
    return seconds, result.projection, settled, counters


def median_seconds(implementation: str, solver: str, z: np.ndarray, A: np.ndarray,
                   b: np.ndarray, budget: int) -> tuple[float, int, np.ndarray, int | None, dict]:
    """Median time of repeated runs, with their count and the last run's other results."""
    timings = []
    while not timings or (sum(timings) < REPEAT_SECONDS and len(timings) < REPEATS):
        seconds, point, settled, counters = timed_run(implementation, solver, z, A, b, budget)
        timings.append(seconds)
    return statistics.median(timings), len(timings), point, settled, counters


def measure_problem(num_planes: int, num_dimensions: int, index: int,
                    implementations: tuple[str, ...]) -> dict:
    """Probe and time one problem in every implementation asked for."""
    A, b, z = problem(num_planes, num_dimensions, problem_seed(num_planes, index))
    reference = ConvexProjectionSolver._find_optimal_solution(z, A, b)
    first_dykstra = cycles_to_tolerance(DykstraProjectionSolver, z, A, b, num_dimensions)
    first_accelerated = cycles_to_tolerance(LTIVer5Solver, z, A, b, num_dimensions,
                                            cap=first_dykstra or CAP)
    record = {"tier": f"{num_planes}x{num_dimensions}", "num_planes": num_planes,
              "num_dimensions": num_dimensions, "index": index,
              "seed": problem_seed(num_planes, index),
              "Dykstra reached": first_dykstra is not None,
              "accelerated reached": first_accelerated is not None}
    budgets = {"Dykstra": first_dykstra or CAP, "accelerated": first_accelerated or CAP,
               "accelerated, certified": CAP}
    for implementation in implementations:
        for name, solver, _, _ in SOLVERS:
            seconds, runs, point, settled, counters = median_seconds(implementation, solver,
                                                                     z, A, b, budgets[name])
            key = f"{implementation} {name}"
            if name == "accelerated, certified":
                record[f"{key} reached"] = settled is not None
                record[f"{key} cycles"] = settled if settled is not None else CAP
            else:
                record[f"{key} cycles"] = budgets[name]
            record[f"{key} seconds"] = seconds
            record[f"{key} runs"] = runs
            record[f"{key} squared error"] = float(np.sum((point - reference) ** 2))
            record.update({f"{key} {counter}": count for counter, count in counters.items()})
    return record


def summarise(records: list[dict], implementation: str) -> dict:
    """Medians over the problems of a tier, per column, with the count that reached."""
    summary = {}
    for name, _, _, _ in SOLVERS:
        key = f"{implementation} {name}"
        reached_key = f"{key} reached" if name == "accelerated, certified" else f"{name} reached"
        summary[name] = {
            "cycles": statistics.median(r[f"{key} cycles"] for r in records),
            "seconds": statistics.median(r[f"{key} seconds"] for r in records),
            "reached": sum(bool(r[reached_key]) for r in records),
            "seeds": len(records),
        }
    return summary


def time_tier(num_planes: int, num_dimensions: int, seeds: int,
              implementation: str = "python") -> dict:
    """Medians for one size tier in one implementation."""
    records = [measure_problem(num_planes, num_dimensions, index, (implementation,))
               for index in range(seeds)]
    return summarise(records, implementation)


def format_ms(seconds: float) -> str:
    """Milliseconds to three significant figures, whole above 100."""
    ms = 1000.0 * seconds
    return f"{ms:.0f}" if ms >= 100 else f"{ms:.3g}"


def table_rows(summaries: dict) -> str:
    """Rows of tab:dykvsacc in the paper's tabular format, one per tier."""
    lines = []
    for (num_planes, num_dimensions), summary in summaries.items():
        cells = [f"${summary[name]['cycles']:.0f}$ & ${format_ms(summary[name]['seconds'])}$ ms"
                 for name, _, _, _ in SOLVERS]
        size = f"${num_planes} \\times {num_dimensions}$"
        lines.append(f"  {size:<16} & " + " & ".join(cells) + r" \\")
    return "\n".join(lines) + "\n"


def reach_comment(summaries: dict) -> str:
    """A LaTeX comment per tier giving how many problems reached each column's target."""
    lines = []
    for (num_planes, num_dimensions), summary in summaries.items():
        counts = ", ".join(f"{name} {summary[name]['reached']}/{summary[name]['seeds']}"
                           for name, _, _, _ in SOLVERS)
        lines.append(f"% {num_planes} x {num_dimensions}: {counts}")
    return "\n".join(lines) + "\n"


def scaling_fit(summaries: dict, name: str) -> tuple[list[int], list[float], float]:
    """Sizes, median seconds to tolerance, and the log-log least-squares gradient."""
    sizes = [n for (n, _) in summaries]
    seconds = [summary[name]["seconds"] for summary in summaries.values()]
    gradient = float(np.polyfit(np.log(sizes), np.log(seconds), 1)[0])
    return sizes, seconds, gradient


def scaling_pgfplots(summaries: dict) -> str:
    """The right panel of fig:performance as pgfplots."""
    plots = []
    for name, colour, mark in (("Dykstra", "figink", "*"), ("accelerated", "figred", "square*")):
        sizes, seconds, gradient = scaling_fit(summaries, name)
        points = " ".join(f"({n},{s:.4g})" for n, s in zip(sizes, seconds))
        plots.append(f"      \\addplot[{colour}, thick, mark={mark}, mark size=1.5pt] "
                     f"coordinates {{{points}}};\n"
                     f"      \\addlegendentry{{{name} $(\\sim n^{{{gradient:.2f}}})$}}\n")
    sizes = [n for (n, _) in summaries]
    all_seconds = [s[name]["seconds"] for s in summaries.values()
                   for name in ("Dykstra", "accelerated")]
    ticks = ",".join(str(n) for n in sizes)
    labels = ",".join(f"${n}$" for n in sizes)
    return rf"""  \begin{{tikzpicture}}
    \begin{{axis}}[paperaxis, width=0.47\linewidth, height=0.40\linewidth,
                 xmode=log, ymode=log, log basis x=10,
                 xmin={0.8 * min(sizes):.3g}, xmax={1.25 * max(sizes):.3g}, ymin={0.5 * min(all_seconds):.2g}, ymax={2.0 * max(all_seconds):.2g},
                 xtick={{{ticks}}}, xticklabels={{{labels}}},
                 xlabel={{number of half-spaces $n$}},
                 ylabel={{median time [s]}},
                 legend pos=north west]
{"".join(plots)}    \end{{axis}}
  \end{{tikzpicture}}
"""


def performance_scaling_figure(summaries: dict, name: str = "performance_scaling") -> None:
    """Draw median time to tolerance against size."""
    fig, ax = plt.subplots(figsize=(6.2, 4.4))
    for column, colour, marker in (("Dykstra", INK, "o"), ("accelerated", RED, "s")):
        sizes, seconds, gradient = scaling_fit(summaries, column)
        ax.loglog(sizes, seconds, marker=marker, color=colour, lw=1.6, ms=5,
                  label=rf"{column}  ($\sim n^{{{gradient:.2f}}}$)")
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
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)
    print(f"Wrote {path}")
    return path


def performance_figure(summaries: dict, curves: dict) -> str:
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
            f"\\cref{{tab:dykvsacc}}, with "
            "the fitted growth order in the legend.}%\n  \\label{fig:performance}\n"
            "\\end{figure}\n")


def measure(tiers: tuple, seeds: int, implementations: tuple[str, ...]) -> list[dict]:
    """Probe and time every problem, printing each as it finishes."""
    records = []
    for num_planes, num_dimensions in tiers:
        for index in range(seeds):
            started = time.perf_counter()
            record = measure_problem(num_planes, num_dimensions, index, implementations)
            records.append(record)
            cells = ", ".join(f"{key} {record[key]}"
                              for key in record if key.endswith("cycles"))
            print(f"{num_planes} x {num_dimensions} index {index}: {cells} "
                  f"({time.perf_counter() - started:.1f} s)", flush=True)
    write_records(records, "performance_timing")
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


def report(records: list[dict], implementations: tuple[str, ...], command: str,
           full_figure: bool = False) -> dict:
    """Write the table, the scaling panel and, if asked, the whole figure per implementation."""
    tiers = tuple(dict.fromkeys((r["num_planes"], r["num_dimensions"]) for r in records))
    seeds = max(sum(r["num_planes"] == n for r in records) for n, _ in tiers)
    curves = None
    if full_figure:
        num_planes, num_dimensions = paper_figures.PERFORMANCE_SIZE
        curves = paper_figures.error_curves(num_planes, num_dimensions,
                                            paper_figures.PERFORMANCE_INDEX,
                                            paper_figures.PERFORMANCE_CYCLES)
    summaries = {}
    for implementation in implementations:
        summaries[implementation] = {
            (n, p): summarise([r for r in records if r["num_planes"] == n], implementation)
            for n, p in tiers}
        rows = table_rows(summaries[implementation])
        header = (f"% {implementation} implementation, {seeds} problems per tier, budget cap "
                  f"{CAP} cycles, squared error {TOLERANCE:g}\n"
                  + reach_comment(summaries[implementation]))
        print(header + rows)
        save_tex(header + rows, f"tab-dykvsacc-{implementation}", command)
        if len(tiers) >= 2:
            save_tex(scaling_pgfplots(summaries[implementation]),
                     f"fig-performance-right-{implementation}", command)
            performance_scaling_figure(summaries[implementation],
                                       f"performance_scaling_{implementation}")
            for column in ("Dykstra", "accelerated"):
                print(f"{implementation} {column}: time to tolerance grows as "
                      f"n^{scaling_fit(summaries[implementation], column)[2]:.2f}")
            if curves is not None:
                save_tex(performance_figure(summaries[implementation], curves),
                         f"fig-performance-{implementation}", command)
    return summaries


def run(tiers: tuple = TABLE_TIERS, seeds: int = SEEDS,
        implementations: tuple[str, ...] = ("c", "python"), command: str | None = None) -> dict:
    """Time every tier in every implementation, print and write the table and the scaling."""
    print_method()
    records = measure(tiers, seeds, implementations)
    return report(records, implementations, command or "python performance_timing.py",
                  full_figure=tiers == TABLE_TIERS)


def main() -> None:
    """Parse the command line and run."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--tiers", choices=("table", "small"), default="table")
    parser.add_argument("--seeds", type=int, default=SEEDS)
    parser.add_argument("--implementation", choices=("c", "python", "both"), default="both")
    parser.add_argument("--from-csv", action="store_true",
                        help="rebuild the outputs from results/paper/performance_timing.csv")
    args = parser.parse_args()
    implementations = ("c", "python") if args.implementation == "both" else (args.implementation,)
    quiet_accelerate_warnings()
    command = " ".join(["python performance_timing.py", *sys.argv[1:]])
    if args.from_csv:
        print_method()
        records = read_records(os.path.join(paper_figures.OUTPUT_DIR, "performance_timing.csv"))
        report(records, implementations, command, full_figure=True)
    else:
        run(TABLE_TIERS if args.tiers == "table" else SMALL_TIERS, args.seeds, implementations,
            command)


if __name__ == "__main__":
    main()
