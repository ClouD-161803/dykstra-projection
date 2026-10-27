"""The phases whose active normals do not span R^p, and the 2-norm and the spectral radius of the
block T of their cycle map on the span of those normals, over the phases of the activity study's
curve battery and of the dense tiers of tab:dykvsacc.

Run from the repository root with `python -m dykstra.singular_phases`, with `--quick` for small
sizes written to a temporary directory, and with `--workers` for the number of worker processes.
The curve battery is drawn from Generator 1, which needs scipy from the dependency group named
benchmark.

A phase is a run of cycles with a constant active set, which the code calls an episode. Its cycle
map A_m is the identity off the span S of its active unit normals, and the field span of the
CycleMap that lti_numerics.cycle_map returns holds an orthonormal basis Q of S in its r columns.
A phase is singular when its active normals do not span R^p, i.e. when r < p, and a phase with an
empty active set counts as singular. For every other phase the script reads the 2-norm of
T = Q^T A_m Q, the block that the deflated solve scans, and its spectral radius, the largest
modulus of the eigenvalues that activity_study.span_eigenvalues returns. On every phase it checks
that p - r equals the unit eigenvalues that span_eigenvalues leaves out, all p of them for an
empty active set.

The two sets of phases are the following.
- activity study: the curve battery of activity_study.py, Generator 1 at the sizes of CURVE_SIZES
  with CURVE_PROBLEMS problems each, recorded by record_runs within a budget of CURVE_HORIZON
  cycles. Its phases are the entries of each run from cycle 1 on, which the study counts as the
  phases of fig:am_eigs. A run whose active set in cycle 1 is the set of half-spaces z violates
  does not have an entry at cycle 1, hence its first phase is not counted.
- dense tiers: the tiers of tab:dykvsacc, Generator 2, problems 0 to DENSE_PROBLEMS - 1 of each,
  with the phases of the schedule that oracle.record_schedule records over DENSE_CYCLES cycles.

A phase holds the cycles from its entry to the next entry, and the last phase of a run holds them
to the last cycle run. The outputs go to results/paper: tab-singular-phases.tex, one row per size
and a total row per set, and singular_phases.csv, the same rows at full precision.
"""

import argparse
from concurrent.futures import ProcessPoolExecutor
import importlib.util
import math
import sys
import tempfile
import time

import numpy as np

from dykstra import activity_study
from dykstra.activity_study import (draw_problem, quiet_accelerate_warnings, record_runs,
                                    span_eigenvalues, study_seed)
from dykstra.lti_numerics import cycle_map, ignore_blas_flags, unit_constraints
from dykstra.oracle import record_schedule
from dykstra.problem_generators import problem, problem_seed


# Chosen here: the problems of each dense tier and the cycles oracle.record_schedule records on
# each
DENSE_PROBLEMS, DENSE_CYCLES = 5, 3000

# The settings of --quick and the test: the curve battery of activity_study.QUICK, and two small
# dense tiers with few problems and cycles, chosen here
QUICK_DENSE_TIERS, QUICK_DENSE_PROBLEMS, QUICK_DENSE_CYCLES = ((12, 4), (24, 8)), 2, 300

# The two sets of phases, as the outputs name them
SETS = {"activity": "activity study", "dense": "dense tiers"}

COUNTS = ("problems", "phases", "empty_phases", "singular_phases", "cycles", "singular_cycles",
          "unit_eigenvalues_left_out", "check_failures")
MAXIMA = ("largest_norm", "largest_spectral_radius", "largest_norm_minus_radius")
CSV_FIELDS = (("set", "size", "n", "p") + COUNTS[:4] + ("singular_share",) + COUNTS[4:6]
              + ("singular_cycle_share",) + COUNTS[6:] + MAXIMA)


@ignore_blas_flags
def phase_statistics(unit_A: np.ndarray, unit_b: np.ndarray, active: np.ndarray) -> tuple:
    """Statistics of the phase of one active set.

    Returns the rank deficiency p - r of the active unit normals, the unit eigenvalues that
    span_eigenvalues leaves out, and the 2-norm and the spectral radius of T, both nan for an
    empty active set.
    """
    cmap = cycle_map(unit_A, unit_b, tuple(bool(a) for a in active))
    values, left_out = span_eigenvalues(unit_A, unit_b, active)
    span = cmap.span
    deficiency = unit_A.shape[1] - span.shape[1]
    if span.shape[1] == 0:
        return deficiency, left_out, math.nan, math.nan
    T = span.T @ cmap.A_m @ span
    return deficiency, left_out, float(np.linalg.norm(T, 2)), float(np.max(np.abs(values)))


class PhaseTally:
    """Counts and maxima over the phases of one size."""

    def __init__(self) -> None:
        self.counts = dict.fromkeys(COUNTS, 0)
        self.maxima = dict.fromkeys(MAXIMA, 0.0)

    def add(self, unit_A: np.ndarray, unit_b: np.ndarray, active: np.ndarray, cycles: int) -> None:
        """Add the phase of an active set that holds the given cycles."""
        deficiency, left_out, norm, radius = phase_statistics(unit_A, unit_b, active)
        counts = self.counts
        counts["phases"] += 1
        counts["cycles"] += cycles
        counts["unit_eigenvalues_left_out"] += left_out
        counts["check_failures"] += deficiency != left_out
        if deficiency > 0:
            counts["singular_phases"] += 1
            counts["singular_cycles"] += cycles
        if math.isnan(norm):
            counts["empty_phases"] += 1
            return
        for key, value in (("largest_norm", norm), ("largest_spectral_radius", radius),
                           ("largest_norm_minus_radius", norm - radius)):
            self.maxima[key] = max(self.maxima[key], value)

    def merge(self, other: "PhaseTally") -> None:
        """Add the phases of another tally."""
        for key, value in other.counts.items():
            self.counts[key] += value
        for key, value in other.maxima.items():
            self.maxima[key] = max(self.maxima[key], value)


def curve_phases(num_planes: int, num_dimensions: int, seeds: list, horizon: int) -> PhaseTally:
    """The phases of the curve-battery problems of the given seeds."""
    problems = [draw_problem(1, num_planes, num_dimensions, seed) for seed in seeds]
    tally = PhaseTally()
    for (A, b, _), run in zip(problems, record_runs(problems, horizon, seeds)):
        unit_A, unit_b = unit_constraints(A, b)
        tally.counts["problems"] += 1
        ends = np.append(run.cycles[1:], run.cycles_run + 1)
        for cycle, end, mask in zip(run.cycles, ends, run.masks):
            # the entry at cycle 0 holds the set z violates, which eq:activeset does not define
            # as the active set of a cycle
            if cycle >= 1:
                tally.add(unit_A, unit_b, mask, int(end - cycle))
    return tally


def dense_phases(num_planes: int, num_dimensions: int, index: int, cycles: int) -> PhaseTally:
    """The phases of problem index of a dense tier over its first cycles."""
    A, b, z = problem(num_planes, num_dimensions, problem_seed(num_planes, index))
    schedule, _ = record_schedule(z, A, b, cycles)
    unit_A, unit_b = unit_constraints(A, b)
    tally = PhaseTally()
    tally.counts["problems"] = 1
    for active, length in schedule:
        tally.add(unit_A, unit_b, np.array(active), length)
    return tally


def _task(task: tuple) -> tuple:
    """One task of either set, returned with its set and size."""
    kind, num_planes, num_dimensions, *arguments = task
    phases = curve_phases if kind == "activity" else dense_phases
    return kind, (num_planes, num_dimensions), phases(num_planes, num_dimensions, *arguments)


def tally_phases(curve_sizes: tuple, curve_problems: int, horizon: int, batch: int,
                 dense_tiers: tuple, dense_problems: int, dense_cycles: int,
                 workers: int) -> dict:
    """The tallies of both sets, keyed by set and then by size in the order given."""
    tasks = [("activity", n, p, [study_seed(n, p, index)
                                 for index in range(start, min(start + batch, curve_problems))],
              horizon)
             for n, p in curve_sizes for start in range(0, curve_problems, batch)]
    tasks += [("dense", n, p, index, dense_cycles)
              for n, p in dense_tiers for index in range(dense_problems)]
    # The costliest tasks start first, so that none is left running alone at the end
    tasks.sort(key=lambda task: -task[1] * task[-1])
    if workers > 1:
        with ProcessPoolExecutor(workers, initializer=quiet_accelerate_warnings) as pool:
            results = list(pool.map(_task, tasks))
    else:
        results = [_task(task) for task in tasks]
    tallies = {"activity": {size: PhaseTally() for size in curve_sizes},
               "dense": {size: PhaseTally() for size in dense_tiers}}
    for kind, size, tally in results:
        tallies[kind][size].merge(tally)
    return tallies


def row(kind: str, size: tuple | None, tally: PhaseTally) -> dict:
    """The output row of one size, or of a set's total when size is None."""
    counts = tally.counts
    n, p = size if size else ("", "")
    values = {"set": kind, "size": f"{n}x{p}" if size else "total", "n": n, "p": p,
              **counts, **tally.maxima,
              "singular_share": counts["singular_phases"] / max(counts["phases"], 1),
              "singular_cycle_share": counts["singular_cycles"] / max(counts["cycles"], 1)}
    return {field: values[field] for field in CSV_FIELDS}


def rows_of(tallies: dict) -> list[dict]:
    """One row per size and a total row per set."""
    rows = []
    for kind, sizes in tallies.items():
        total = PhaseTally()
        for size, tally in sizes.items():
            rows.append(row(kind, size, tally))
            total.merge(tally)
        rows.append(row(kind, None, total))
    return rows


def percent(share: float) -> str:
    """A share as a percentage, to one decimal from 1% up and two significant figures below."""
    value = 100.0 * share
    return f"{value:.1f}" if value >= 1.0 else f"{value:.2g}"


def one_minus(value: float) -> str:
    """LaTeX for a value below 1 as 1 minus a mantissa of three figures times a power of ten."""
    gap = 1.0 - value
    if gap <= 0.0:
        return f"${value:g}$"
    exponent = math.floor(math.log10(gap))
    return f"$1 - {gap / 10.0 ** exponent:.2f} \\times 10^{{{exponent}}}$"


def table_rows(rows: list[dict]) -> str:
    """Rows of tab-singular-phases.tex, one per size and a total per set."""
    lines = []
    for r in rows:
        size = "total" if r["size"] == "total" else f"${r['n']} \\times {r['p']}$"
        cells = [SETS[r["set"]], size, f"${r['phases']}$", f"${r['singular_phases']}$",
                 f"${percent(r['singular_share'])}\\%$",
                 f"${percent(r['singular_cycle_share'])}\\%$",
                 one_minus(r["largest_norm"]), one_minus(r["largest_spectral_radius"])]
        lines.append("  " + " & ".join(cells) + r" \\")
    return "\n".join(lines) + "\n"


def table_header(rows: list[dict], curve_problems: int, horizon: int, dense_problems: int,
                 dense_cycles: int) -> str:
    """The LaTeX comment naming every column of tab-singular-phases.tex and its source."""
    failures = sum(r["check_failures"] for r in rows if r["size"] == "total")
    return ("% a phase is singular when its active normals do not span R^p, an empty active set "
            "included, and T = Q^T A_m Q is its cycle map on the span of those normals\n"
            "% columns: set; size n x p; phases; singular phases; their share of the phases; "
            "their share of the cycles; largest 2-norm of T; largest spectral radius of T\n"
            f"% activity study: Generator 1, {curve_problems} problems per size, cycle budget "
            f"{horizon}, phases from cycle 1 on as activity_study counts them. dense tiers: "
            f"Generator 2, problems 0 to {dense_problems - 1} per tier, the first {dense_cycles} "
            "cycles\n"
            f"% the rank deficiency p - r differs from the unit eigenvalues that span_eigenvalues "
            f"leaves out on {failures} phases\n")


def print_rows(rows: list[dict]) -> None:
    """Print every number of every row."""
    for r in rows:
        print(f"{SETS[r['set']]} {r['size']}: {r['singular_phases']} of {r['phases']} phases of "
              f"{r['problems']} problems are singular ({100.0 * r['singular_share']:.3g}%), "
              f"{r['empty_phases']} of them empty, and they hold {r['singular_cycles']} of "
              f"{r['cycles']} cycles ({100.0 * r['singular_cycle_share']:.3g}%). Unit eigenvalues "
              f"left out: {r['unit_eigenvalues_left_out']}. Largest 2-norm of T: "
              f"{r['largest_norm']!r}, i.e. 1 - {1.0 - r['largest_norm']:.3g}. Largest spectral "
              f"radius: {r['largest_spectral_radius']!r}, i.e. 1 - "
              f"{1.0 - r['largest_spectral_radius']:.3g}. Largest excess of the norm over the "
              f"radius: {r['largest_norm_minus_radius']:.3g}. Check failures: "
              f"{r['check_failures']}.", flush=True)


def run(quick: bool = False, workers: int = activity_study.WORKERS,
        command: str = "python -m dykstra.singular_phases") -> list[dict]:
    """Tally both sets of phases, write the outputs, and print every number they hold."""
    # paper_figures and performance_timing load matplotlib, which the workers do not need
    from dykstra import paper_figures
    from dykstra.cost_model import write_csv
    from dykstra.performance_timing import TABLE_TIERS

    quiet_accelerate_warnings()
    if quick:
        curve = activity_study.QUICK
        dense = QUICK_DENSE_TIERS, QUICK_DENSE_PROBLEMS, QUICK_DENSE_CYCLES
    else:
        curve = {name: getattr(activity_study, name)
                 for name in ("CURVE_SIZES", "CURVE_PROBLEMS", "CURVE_HORIZON", "BATCH")}
        dense = TABLE_TIERS, DENSE_PROBLEMS, DENSE_CYCLES
    tallies = tally_phases(curve["CURVE_SIZES"], curve["CURVE_PROBLEMS"], curve["CURVE_HORIZON"],
                           curve["BATCH"], *dense, workers)
    rows = rows_of(tallies)
    write_csv(rows, "singular_phases")
    text = (table_header(rows, curve["CURVE_PROBLEMS"], curve["CURVE_HORIZON"], *dense[1:])
            + table_rows(rows))
    paper_figures.save_tex(text, "tab-singular-phases", command)
    print(text)
    print_rows(rows)
    return rows


def main() -> None:
    """Parse the command line and run."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--quick", action="store_true",
                        help="small sizes, written to a temporary directory")
    parser.add_argument("--workers", type=int, default=activity_study.WORKERS,
                        help=f"processes, {activity_study.WORKERS} by default")
    args = parser.parse_args()
    if importlib.util.find_spec("scipy") is None:
        sys.exit("singular_phases needs scipy for Generator 1: uv sync --group benchmark")
    from dykstra import paper_figures

    began = time.perf_counter()
    if args.quick:
        paper_figures.OUTPUT_DIR = tempfile.mkdtemp(prefix="singular_phases_")
    run(args.quick, args.workers, " ".join(["python -m dykstra.singular_phases", *sys.argv[1:]]))
    print(f"Finished in {time.perf_counter() - began:.0f} s: tab-singular-phases.tex and "
          f"singular_phases.csv in {paper_figures.OUTPUT_DIR}")


if __name__ == "__main__":
    main()
