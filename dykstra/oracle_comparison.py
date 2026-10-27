"""The oracle of oracle.py against the Python accelerated solver on the dense problems of
benchmark.py, in time and in the work each does: the episodes and closed forms of the oracle, and
the counts of a profiled solve of the accelerated solver.

Run from the repository root with `python -m dykstra.oracle_comparison` after building the C
solvers with `make -C C`, or with `--quick` for problem 0 of the first two tiers, timed once and
written to a temporary directory.

The problems are the dense family of benchmark.py, problems 0 to SEEDS - 1 of each tier of TIERS,
drawn by benchmark.family_problem. On each of them benchmark.run_projection_method runs C Dykstra
under the benchmark's budget of BUDGET cycles, and then the oracle as the benchmark runs it: the
schedule of active sets that plain Python Dykstra records over C Dykstra's cycles is replayed, and
the oracle solves the problem when the replay lands on C Dykstra's settled point to within 1e-9
relative to that point. The schedule is recorded once more, untimed, for the work of the oracle.
oracle_lti_projection enters every episode with one exact cycle and evaluates the rest of an
episode of two or more cycles in closed form, or steps it when the resolvent of its cycle map is
too ill-conditioned for the closed form.

On every problem the oracle solves, LTIVer5Solver, the accelerated solver of the paper, runs a
profiled solve under its default schedule with a budget of ACCELERATED_BUDGET cycles, which gives
its counts, and it solves the problem when it settles with the KKT certificate. After a settlement
the solver writes the limit into every remaining row of its history, which within that budget
can take most of the time of a solve. It is therefore timed at its settle cycle plus one, where
it settles on the same cycle with the same point, which the script checks. Every time is the median of
benchmark.repeated without the profiler. The accelerated solver is timed in solve() alone, since
its constructor runs the quadprog reference solve, and the oracle in its call of
oracle_lti_projection.

The outputs go to results/paper: tab-oracle-comparison.tex, one row of medians per tier and one
over every problem both solve, then a line with the total times and the per-problem ratio of the
times, and oracle_comparison.csv, one record per problem.
"""

import argparse
import os
import statistics
import sys
import tempfile
import time

import numpy as np

from dykstra import benchmark, c_solvers, paper_figures, performance_timing
from dykstra.cost_model import number, write_csv
from dykstra.lti_numerics import (DEFAULT_SCHEDULE, RESOLVENT_COND_CAP, cycle_map,
                                  ignore_blas_flags, touched_resolvent, unit_constraints)
from dykstra.lti_solver import COUNTS, LTIVer5Solver
from dykstra.oracle import record_schedule
from dykstra.paper_figures import quiet_accelerate_warnings, save_tex


TIERS, SEEDS = benchmark.TIERS, benchmark.SEEDS

# Chosen here: the budget within which the accelerated solver must settle, far above the settle
# cycles of the problems the oracle solves
ACCELERATED_BUDGET = 100_000

QUICK_TIERS, QUICK_SEEDS = TIERS[:2], 1

# The medians of a row of tab-oracle-comparison.tex after its tier and the problems both solve,
# with the ratio of the two times third
COLUMNS = ("oracle_ms", "accelerated_ms", "ratio", "oracle_episodes", "oracle_closed_forms",
           "accelerated_episodes", "accelerated_exact_cycles", "accelerated_evaluated_cycles",
           "accelerated_skipped_cycles", "accelerated_kkt_attempts", "accelerated_settled_at")

FIELDS = (("tier", "num_planes", "num_dimensions", "index", "seed", "dykstra_settled",
           "dykstra_cycles", "dykstra_seconds", "oracle_solved", "oracle_seconds", "oracle_runs",
           "oracle_replay_difference", "oracle_cycles", "oracle_episodes", "oracle_closed_forms",
           "oracle_stepped_episodes", "accelerated_settled", "accelerated_certificate",
           "accelerated_settled_at", "accelerated_budget", "accelerated_seconds",
           "accelerated_runs") + tuple(f"accelerated_{name}" for name in COUNTS)
          + ("both_solved",))


@ignore_blas_flags
def oracle_work(z: np.ndarray, A: np.ndarray, b: np.ndarray, cycles: int) -> dict:
    """The episodes of the schedule the oracle replays over the given cycles, and those it
    evaluates in closed form and steps, as oracle_lti_projection decides between them."""
    schedule, _ = record_schedule(z, A, b, cycles)
    unit_A, unit_b = unit_constraints(A, b)
    closed = stepped = 0
    for active, length in schedule:
        if length < 2:
            continue
        resolvent = touched_resolvent(cycle_map(unit_A, unit_b, active))
        if np.linalg.cond(resolvent) < RESOLVENT_COND_CAP:
            closed += 1
        else:
            stepped += 1
    return {"oracle_episodes": len(schedule), "oracle_closed_forms": closed,
            "oracle_stepped_episodes": stepped,
            "schedule_cycles": sum(length for _, length in schedule)}


def measure_accelerated(z: np.ndarray, A: np.ndarray, b: np.ndarray) -> dict:
    """The counts of a profiled solve of LTIVer5Solver within ACCELERATED_BUDGET cycles, and,
    when it settles, the median time of its solve at its settle cycle plus one."""
    solver = LTIVer5Solver(z, A, b, ACCELERATED_BUDGET, profile=True)
    result = solver.solve()
    settled = result.certificate == "kkt"
    record = {"accelerated_settled": settled, "accelerated_certificate": result.certificate or "",
              "accelerated_settled_at": result.settled_at if settled else "",
              "accelerated_budget": "", "accelerated_seconds": "", "accelerated_runs": "",
              **{f"accelerated_{name}": solver.profile["counts"][name] for name in COUNTS}}
    if not settled:
        return record
    budget = result.settled_at + 1
    seconds, runs, timed = benchmark.repeated(LTIVer5Solver(z, A, b, budget).solve)
    if ((timed.settled_at, timed.certificate) != (result.settled_at, "kkt")
            or not np.array_equal(timed.projection, result.projection)):
        raise RuntimeError(f"LTIVer5Solver settles otherwise within {budget} cycles than within "
                           f"{ACCELERATED_BUDGET}")
    record.update(accelerated_budget=budget, accelerated_seconds=seconds, accelerated_runs=runs)
    return record


def measure_problem(tier: tuple, index: int) -> dict:
    """Every measurement of problem index of a dense tier, as one record of FIELDS."""
    A, b, z, seed = benchmark.family_problem("dense", tier, index)
    record = dict.fromkeys(FIELDS, "")
    record.update(tier=f"{tier[0]}x{tier[1]}", num_planes=tier[0], num_dimensions=tier[1],
                  index=index, seed=seed)
    dykstra = benchmark.run_projection_method("Dykstra", z, A, b, None)
    oracle = benchmark.run_projection_method("oracle", z, A, b, dykstra)
    record.update(dykstra_settled=dykstra["solved"], dykstra_cycles=dykstra["cycles"],
                  dykstra_seconds=dykstra["seconds"], oracle_solved=oracle["solved"])
    if dykstra["solved"]:
        work = oracle_work(z, A, b, dykstra["cycles"])
        if work.pop("schedule_cycles") != oracle["cycles"]:
            raise RuntimeError(f"the schedule recorded again for {record['tier']} index {index} "
                               "differs from the one the oracle replayed")
        record.update(work, oracle_seconds=oracle["seconds"], oracle_runs=oracle["runs"],
                      oracle_replay_difference=oracle["replay difference"],
                      oracle_cycles=oracle["cycles"])
    if oracle["solved"]:
        record.update(measure_accelerated(z, A, b))
    record["both_solved"] = bool(oracle["solved"]) and record["accelerated_settled"] is True
    return record


def summary_row(records: list[dict], tier: str) -> dict:
    """The medians of COLUMNS over the problems of records both solve, under the name tier."""
    both = [r for r in records if r["both_solved"]]
    row = {"tier": tier, "problems": len(records), "both_solved": len(both)}
    if not both:
        return row
    row["oracle_ms"] = 1e3 * statistics.median(r["oracle_seconds"] for r in both)
    row["accelerated_ms"] = 1e3 * statistics.median(r["accelerated_seconds"] for r in both)
    row["ratio"] = row["accelerated_ms"] / row["oracle_ms"]
    for column in COLUMNS[3:]:
        row[column] = statistics.median(r[column] for r in both)
    return row


def totals(records: list[dict]) -> dict:
    """The total times over the problems both solve, their ratio, and the median, smallest and
    largest ratio of the accelerated solver's time to the oracle's over those problems."""
    both = [r for r in records if r["both_solved"]]
    oracle = sum(r["oracle_seconds"] for r in both)
    accelerated = sum(r["accelerated_seconds"] for r in both)
    ratios = [r["accelerated_seconds"] / r["oracle_seconds"] for r in both]
    return {"problems": len(records), "both_solved": len(both), "oracle_seconds": oracle,
            "accelerated_seconds": accelerated, "ratio_of_totals": accelerated / oracle,
            "median_ratio": statistics.median(ratios), "smallest_ratio": min(ratios),
            "largest_ratio": max(ratios)}


def summarise(records: list[dict]) -> list[dict]:
    """One summary row per tier in the order of records, then one over every problem."""
    tiers = dict.fromkeys(r["tier"] for r in records)
    return ([summary_row([r for r in records if r["tier"] == tier], tier) for tier in tiers]
            + [summary_row(records, "all")])


def totals_text(total: dict) -> str:
    """The sentence of the last line of tab-oracle-comparison.tex."""
    return (f"over the ${total['both_solved']}$ problems both solve, the oracle takes "
            f"${number(total['oracle_seconds'])}$ s and the accelerated solver "
            f"${number(total['accelerated_seconds'])}$ s in total, a ratio of "
            f"${number(total['ratio_of_totals'])}$, and the median ratio per problem is "
            f"${number(total['median_ratio'])}$, from ${number(total['smallest_ratio'])}$ to "
            f"${number(total['largest_ratio'])}$")


def table_rows(summary: list[dict], total: dict) -> str:
    """Rows of tab-oracle-comparison.tex, one per tier and one over every problem, and the line
    of totals."""
    lines = []
    for row in summary:
        if row["tier"] == "all":
            label = "all tiers"
        else:
            n, p = row["tier"].split("x")
            label = f"${n} \\times {p}$"
        cells = [label, f"${row['both_solved']}/{row['problems']}$"]
        cells += ([f"${number(row[column])}$" for column in COLUMNS] if row["both_solved"]
                  else ["---"] * len(COLUMNS))
        lines.append("  " + " & ".join(cells) + r" \\")
    lines.append(f"  \\multicolumn{{{2 + len(COLUMNS)}}}{{l}}{{{totals_text(total)}}} \\\\")
    return "\n".join(lines) + "\n"


def describe() -> list[str]:
    """The machine, the package versions and the method, as lines."""
    lines = [f"{key}: {value}" for key, value in performance_timing.machine_description().items()]
    lines.append(f"method: each time the median of runs repeated until they total "
                 f"{benchmark.REPEAT_SECONDS} s or number {benchmark.REPEATS}, by "
                 f"benchmark.repeated; C Dykstra under {benchmark.BUDGET} cycles and the oracle as "
                 f"benchmark.run_projection_method runs them; LTIVer5Solver under the schedule "
                 f"{DEFAULT_SCHEDULE}, settled within {ACCELERATED_BUDGET} cycles by a profiled "
                 f"solve and timed unprofiled in solve() alone at its settle cycle plus one")
    return lines


def table_header(header: list[str], total: dict) -> str:
    """The LaTeX comment that names the machine, the method, and every column."""
    return ("".join(f"% {line}\n" for line in header)
            + "% an episode is the paper's phase. columns: tier; problems both solve of the tier's "
            "problems; medians over the problems both solve of the oracle's ms, the accelerated "
            "solver's ms, the ratio of these two medians, the oracle's episodes and closed forms, "
            "and the accelerated solver's episodes, exact, evaluated, and skipped cycles, KKT "
            "tests, and settle cycle\n"
            f"% {total['both_solved']} of {total['problems']} problems solved by both\n")


def print_record(record: dict, seconds: float) -> None:
    """Print the measurements of one problem."""
    line = f"dense {record['tier']} index {record['index']}: C Dykstra "
    if not record["dykstra_settled"]:
        print(line + f"did not settle within {benchmark.BUDGET} cycles ({seconds:.1f} s)",
              flush=True)
        return
    line += (f"settled at {record['dykstra_cycles']} cycles. The oracle "
             f"{'solved' if record['oracle_solved'] else 'did not solve'} it in "
             f"{1e3 * record['oracle_seconds']:.3g} ms over {record['oracle_episodes']} "
             f"episodes, {record['oracle_closed_forms']} in closed form and "
             f"{record['oracle_stepped_episodes']} stepped.")
    if record["oracle_solved"]:
        line += (f" LTIVer5Solver {'settled' if record['accelerated_settled'] else 'did not settle'}"
                 + (f" at cycle {record['accelerated_settled_at']} in "
                    f"{1e3 * record['accelerated_seconds']:.3g} ms" if record["accelerated_settled"]
                    else "")
                 + f" after {record['accelerated_episodes']} episodes and "
                 f"{record['accelerated_kkt_attempts']} KKT tests.")
    print(line + f" ({seconds:.1f} s)", flush=True)


def report(records: list[dict], header: list[str], command: str) -> tuple[list[dict], dict]:
    """Write both outputs, and print the rows and the totals."""
    summary, total = summarise(records), totals(records)
    write_csv(records, "oracle_comparison")
    text = table_header(header, total) + table_rows(summary, total)
    save_tex(text, "tab-oracle-comparison", command)
    print(text)
    for row in summary:
        name = "all tiers" if row["tier"] == "all" else row["tier"]
        if not row["both_solved"]:
            print(f"{name}: none of {row['problems']} problems solved by both")
            continue
        print(f"{name}: {row['both_solved']} of {row['problems']} problems solved by both, "
              f"median oracle {row['oracle_ms']:.3g} ms and accelerated {row['accelerated_ms']:.3g} "
              f"ms, ratio {row['ratio']:.3g}. Oracle medians: {row['oracle_episodes']:g} episodes, "
              f"{row['oracle_closed_forms']:g} closed forms. Accelerated medians: "
              f"{row['accelerated_episodes']:g} episodes, {row['accelerated_exact_cycles']:g} exact, "
              f"{row['accelerated_evaluated_cycles']:g} evaluated and "
              f"{row['accelerated_skipped_cycles']:g} skipped cycles, "
              f"{row['accelerated_kkt_attempts']:g} KKT tests, settle cycle "
              f"{row['accelerated_settled_at']:g}")
    print(f"Totals {totals_text(total).replace('$', '')}", flush=True)
    return summary, total


def warm_up() -> None:
    """Load the C libraries and run every method once, so that no timing includes a first call."""
    A, b, z, _ = benchmark.family_problem("dense", TIERS[0], 0)
    dykstra = benchmark.run_projection_method("Dykstra", z, A, b, None)
    benchmark.run_projection_method("oracle", z, A, b, dykstra)
    LTIVer5Solver(z, A, b, 2, profile=True).solve()


def run(tiers: tuple, seeds: int, command: str) -> list[dict]:
    """Measure problems 0 to seeds - 1 of each tier, printing each as it finishes, and write the
    outputs."""
    header = describe()
    for line in header:
        print(line)
    print(f"load averages: {', '.join(f'{load:.2f}' for load in os.getloadavg())}", flush=True)
    warm_up()
    records = []
    for tier in tiers:
        for index in range(seeds):
            started = time.perf_counter()
            records.append(measure_problem(tier, index))
            print_record(records[-1], time.perf_counter() - started)
    report(records, header, command)
    return records


def main() -> None:
    """Parse the command line and run."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--quick", action="store_true",
                        help="problem 0 of the first two tiers and one run per time, written to a "
                             "temporary directory")
    args = parser.parse_args()
    if not c_solvers.libraries_built():
        sys.exit(f"oracle_comparison needs the C solvers: build them with make -C {c_solvers.C_FOLDER}")
    quiet_accelerate_warnings()
    command = " ".join(["python -m dykstra.oracle_comparison", *sys.argv[1:]])
    began = time.perf_counter()
    if args.quick:
        benchmark.REPEATS = 1
        paper_figures.OUTPUT_DIR = tempfile.mkdtemp(prefix="oracle_comparison_")
        run(QUICK_TIERS, QUICK_SEEDS, command)
    else:
        run(TIERS, SEEDS, command)
    print(f"Finished in {time.perf_counter() - began:.0f} s: tab-oracle-comparison.tex and "
          f"oracle_comparison.csv in {paper_figures.OUTPUT_DIR}")


if __name__ == "__main__":
    main()
