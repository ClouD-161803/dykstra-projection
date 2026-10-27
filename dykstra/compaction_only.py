"""Compaction alone: the accelerated solver without its KKT test, the schedule never, against
Dykstra's algorithm, for the SIAM paper's section on the accelerated algorithm, and a check of
the closed-form length of a stalling period on the stall family.

Run from the repository root with `python -m dykstra.compaction_only`, or with `--quick` for a
short run written to a temporary directory. Under the schedule never the deflated_modal preset
runs no KKT test, and since that test is the only way it settles, every run ends at its
budget: inside a phase, or after the finality test has proven the active set final and the
solver has jumped to the budget. What remains is compaction, i.e. the cycles a phase evaluates
in closed form, its envelope jumps, the stall fast-forward and that last jump, which the
certificate at the first phase entry otherwise hides on the wedge and stall families.

The families are the wedge family of problem_generators.WedgeTestPlanes in the cells of
WEDGE_CELLS, the stall family of BoxSlabTestPlanes in STALL_DIMENSIONS, both with the seeds of
benchmark.py, and the six size tiers of tab:dykvsacc from Generator 2, PROBLEMS problems each.
The budget of a problem is the first cycle at which Dykstra's iterate is within squared error
TOLERANCE of the quadprog projection, or CAP. LTIVer5Solver, profiled, gives the counts and the
path, which is compared cycle by cycle with DykstraProjectionSolver's, and the C solvers give
the times, each the median of runs repeated until they total REPEAT_SECONDS or number REPEATS.
The rows are medians over a cell's or a tier's problems, except the largest deviation, which is
the largest over them.

The stall-length check takes problem STALL_CHECK_INDEX of each stall tier. From the state after
the first cycle it computes N_stall = min over the draining half-spaces m of
ceil(y_m / -beta_m), as square_line_example.stalling_state does for the paper's example, and
sets it beside the cycles plain Dykstra's state stays stationary and beside the stalling period
that the relative detector of stall_grid.run_batch finds.

The outputs go to results/paper: tab-compaction-only.tex and compaction_only.csv, one row per
problem, and tab-stall-length.tex and stall_length.csv, one row per instance.
"""

import argparse
import csv
import os
import statistics
import sys
import tempfile
import time
from concurrent.futures import ProcessPoolExecutor

import numpy as np

from dykstra import c_solvers, paper_figures, performance_timing, stall_grid
from dykstra.convex_projection_solver import ConvexProjectionSolver, DykstraProjectionSolver
from dykstra.lti_solver import LTIVer5Solver
from dykstra.paper_figures import quiet_accelerate_warnings, save_tex
from dykstra.problem_generators import (BoxSlabTestPlanes, WedgeTestPlanes, problem,
                                        problem_seed)
from dykstra.square_line_example import (dykstra_cycles, first_inactive_cycle,
                                         first_moved_cycle, stalling_state)

# The cells and seeds of benchmark.py's wedge and stall families, five problems each as there,
# and the size tiers of tab:dykvsacc with the same count of problems, chosen here
WEDGE_CELLS = performance_timing.WEDGE_CELLS
WEDGE_ANGLES = performance_timing.WEDGE_ANGLES
WEDGE_SEED = performance_timing.WEDGE_SEED
STALL_DIMENSIONS = (2, 4, 8)
STALL_SEED = 6_000_000
DENSE_TIERS = performance_timing.TABLE_TIERS
PROBLEMS = 5

TOLERANCE = performance_timing.TOLERANCE
CAP = performance_timing.CAP
REPEAT_SECONDS, REPEATS = performance_timing.REPEAT_SECONDS, performance_timing.REPEATS

# The stall-length check's instance in each stall tier, and the cycles it runs beyond the
# predicted switch, chosen here
STALL_CHECK_INDEX = 0
STALL_CHECK_MARGIN = 10

# Worker processes, chosen here: the paths of the largest tier at the cap hold 2.6 GB each
WORKERS = 4

# Under --quick, chosen here
QUICK_WEDGE_CELLS, QUICK_STALL_DIMENSIONS, QUICK_DENSE_TIERS, QUICK_PROBLEMS = (
    WEDGE_CELLS[:1], STALL_DIMENSIONS[:1], DENSE_TIERS[:1], 1)


def problem_list(wedge_cells: tuple, stall_dimensions: tuple, dense_tiers: tuple,
                 problems: int) -> list[dict]:
    """Every problem of the study, as the family, its row and the seed."""
    specs = [{"family": "wedge", "row": f"wedge, p = {p}, theta = {theta:g}", "p": p, "theta": theta,
              "num_planes": 2 * p - 2, "index": i,
              "seed": WEDGE_SEED + 1000 * p + 100 * WEDGE_ANGLES.index(theta) + i}
             for p, theta in wedge_cells for i in range(problems)]
    specs += [{"family": "stall", "row": f"stall, p = {p}", "p": p, "theta": 0.0,
               "num_planes": 2 * p + 2, "index": i, "seed": STALL_SEED + 1000 * p + i}
              for p in stall_dimensions for i in range(problems)]
    specs += [{"family": "dense", "row": f"dense, {n} x {p}", "p": p, "theta": 0.0,
               "num_planes": n, "index": i, "seed": problem_seed(n, i)}
              for n, p in dense_tiers for i in range(problems)]
    return specs


def problem_data(spec: dict) -> tuple:
    """The constraints and the start of a problem, as (A, b, z)."""
    if spec["family"] == "wedge":
        return WedgeTestPlanes(spec["theta"], spec["p"]).problem(spec["seed"])
    if spec["family"] == "stall":
        return BoxSlabTestPlanes(spec["p"]).problem(spec["seed"])
    return problem(spec["num_planes"], spec["p"], spec["seed"])


def first_within(errors: np.ndarray) -> int | None:
    """The first cycle whose squared error is within TOLERANCE, or None."""
    reached = np.flatnonzero(np.asarray(errors, dtype=float) <= TOLERANCE)
    return int(reached[0]) if reached.size else None


def analyse(spec: dict) -> dict:
    """The Python half of a problem: Dykstra's budget, and the counts and path of LTIVer5Solver
    under the schedule never against Dykstra's path."""
    quiet_accelerate_warnings()
    A, b, z = problem_data(spec)
    p = spec["p"]
    first_dykstra = performance_timing.cycles_to_tolerance(DykstraProjectionSolver, z, A, b, p)
    budget = CAP if first_dykstra is None else first_dykstra
    # one path at a time, since at the cap of the largest tier each holds 2.6 GB
    dykstra = DykstraProjectionSolver(z, A, b, budget, dimensions=p).solve().path[:, -1].copy()
    solver = LTIVer5Solver(z, A, b, budget, dimensions=p, profile=True, track_error=True,
                           min_error=TOLERANCE, schedule="never")
    result = solver.solve()
    accelerated = result.path[:, -1]
    scale = float(np.abs(z).max())
    deviation = float(np.abs(accelerated - dykstra).max())
    counts = solver.profile["counts"]
    first_accelerated = first_within(solver.squared_errors)
    del result
    return {**spec, "budget": budget, "dykstra_reached": first_dykstra is not None,
            "dykstra_first_within": -1 if first_dykstra is None else first_dykstra,
            "accelerated_first_within": -1 if first_accelerated is None else first_accelerated,
            "settled": solver.settled_at is not None,
            "ended_by": "finality, then the budget" if counts["budget_jumps"] else "budget",
            "exact_cycles": counts["exact_cycles"], "evaluated_cycles": counts["evaluated_cycles"],
            "envelope_skipped_cycles": counts["envelope_skipped_cycles"],
            "stall_skipped_cycles": counts["stall_skipped_cycles"],
            "budget_skipped_cycles": counts["budget_skipped_cycles"],
            "envelope_jumps": counts["envelope_jumps"],
            "stall_fast_forwards": counts["stall_fast_forwards"],
            "episodes": counts["episodes"], "kkt_attempts": counts["kkt_attempts"],
            "largest_deviation": deviation, "largest_relative_deviation": deviation / scale}


def median_seconds(call) -> float:
    """Median time of repeated calls, until they total REPEAT_SECONDS or number REPEATS."""
    timings = []
    while not timings or (sum(timings) < REPEAT_SECONDS and len(timings) < REPEATS):
        start = time.perf_counter()
        call()
        timings.append(time.perf_counter() - start)
    return statistics.median(timings)


def time_in_c(record: dict) -> dict:
    """The C half of a problem: both solvers timed at the budget, one after the other."""
    A, b, z = problem_data(record)
    budget = record["budget"]
    dykstra = c_solvers.dykstra_c_projection(z, A, b, budget)
    accelerated = c_solvers.accelerated_c_projection(z, A, b, budget, schedule="never")
    reference = ConvexProjectionSolver._find_optimal_solution(z, A, b)
    return {**record,
            "c_dykstra_ms": 1000.0 * median_seconds(
                lambda: c_solvers.dykstra_c_projection(z, A, b, budget)),
            "c_accelerated_ms": 1000.0 * median_seconds(
                lambda: c_solvers.accelerated_c_projection(z, A, b, budget, schedule="never")),
            "c_settled": accelerated.settled, "c_cycles_skipped": accelerated.cycles_skipped,
            "c_exact_cycles": accelerated.exact_cycles,
            "c_deviation": float(np.abs(accelerated.projection - dykstra.projection).max()),
            "c_squared_error": float(np.sum((accelerated.projection - reference) ** 2))}


def median(records: list[dict], key: str) -> float:
    """The median of a column over records."""
    return float(np.median([record[key] for record in records]))


def number(value: float) -> str:
    """A median as the table prints it, with a half where it falls between two problems."""
    return f"{value:.0f}" if float(value).is_integer() else f"{value:.1f}"


def summary_rows(records: list[dict]) -> list[dict]:
    """One row per wedge cell, stall tier and dense tier, of medians over its problems."""
    rows = []
    for row in dict.fromkeys(record["row"] for record in records):
        group = [record for record in records if record["row"] == row]
        ended = {}
        for record in group:
            ended[record["ended_by"]] = ended.get(record["ended_by"], 0) + 1
        rows.append({
            "row": row, "problems": len(group),
            "dykstra_reached": sum(record["dykstra_reached"] for record in group),
            **{key: median(group, key) for key in (
                "dykstra_first_within", "accelerated_first_within", "exact_cycles",
                "evaluated_cycles", "envelope_skipped_cycles", "stall_skipped_cycles",
                "stall_fast_forwards", "budget_skipped_cycles", "c_dykstra_ms",
                "c_accelerated_ms")},
            "first_within_equal": sum(record["dykstra_first_within"]
                                      == record["accelerated_first_within"] for record in group),
            "largest_relative_deviation": max(record["largest_relative_deviation"]
                                              for record in group),
            "largest_c_deviation": max(record["c_deviation"] for record in group),
            "settled": sum(record["settled"] or record["c_settled"] for record in group),
            "ended_by": "; ".join(f"{rule} {count}" for rule, count in ended.items())})
    return rows


def compaction_table(rows: list[dict]) -> str:
    """The rows of tab-compaction-only.tex, headed by a comment naming every column."""
    lines = ["% under the schedule never, per wedge cell, stall tier and dense tier: problems; the "
             "median cycle at which Dykstra and the accelerated solver reach squared error "
             f"{TOLERANCE:g}; the median exact cycles, cycles evaluated one by one, cycles skipped "
             "by envelope jumps, cycles skipped by stall fast-forwards, fast-forwards, and cycles "
             "skipped by the jump to the budget; the median C times of Dykstra and of the "
             "accelerated solver at Dykstra's budget; the largest deviation of the accelerated "
             "iterate from Dykstra's at the same cycle over the run, relative to the largest "
             "coordinate of z; and the rule that ended the runs\n"]
    for row in rows:
        lines.append(
            f"  {row['row']:<28} & {row['problems']} & ${number(row['dykstra_first_within'])}$ & "
            f"${number(row['accelerated_first_within'])}$ & ${number(row['exact_cycles'])}$ & "
            f"${number(row['evaluated_cycles'])}$ & ${number(row['envelope_skipped_cycles'])}$ & "
            f"${number(row['stall_skipped_cycles'])}$ & ${number(row['stall_fast_forwards'])}$ & "
            f"${number(row['budget_skipped_cycles'])}$ & "
            f"${performance_timing.format_ms(row['c_dykstra_ms'] / 1000.0)}$ ms & "
            f"${performance_timing.format_ms(row['c_accelerated_ms'] / 1000.0)}$ ms & "
            f"${row['largest_relative_deviation']:.1e}$ & {row['ended_by']} \\\\\n")
    return "".join(lines)


def stall_length(p: int) -> dict:
    """The closed-form stall length of one stall instance against plain Dykstra's run."""
    seed = STALL_SEED + 1000 * p + STALL_CHECK_INDEX
    A, b, z = BoxSlabTestPlanes(p).problem(seed)
    x, y = dykstra_cycles(z, A, b, 1)[1][:2]
    state = stalling_state(A, b, x, y)
    n_stall, m_switch = state["n_stall"], state["m_switch"]
    horizon = 1 + n_stall + STALL_CHECK_MARGIN
    states = dykstra_cycles(z, A, b, horizon)
    reference = ConvexProjectionSolver._find_optimal_solution(z, A, b)
    batch = stall_grid.run_batch(z[None], A[None], b[None], reference[None], horizon)
    periods = batch["relative_period_lengths"][0]
    # the states after cycles 1 to stationary_through coincide
    stationary_through = first_moved_cycle(states, np.array(state["points"][1:]), 1) - 1
    leaves = first_inactive_cycle(states, m_switch, 1)
    active = np.array(state["active"]) - 1
    return {"p": p, "seed": seed, "n_stall": n_stall, "m_switch": m_switch,
            "t_switch": 1 + n_stall, "stationary_through": stationary_through,
            "switch_half_space_leaves": leaves,
            "relative_period": periods[0] if periods else 0,
            "relative_stalling_cycles": int(batch["relative_cycles"][0]),
            "matches": stationary_through == n_stall and leaves == 1 + n_stall,
            "entry_state": " ".join(f"{v:.6g}" for v in x),
            "active": " ".join(str(m) for m in state["active"]),
            "auxiliaries": " ".join(f"{y[m]:.6g}" for m in active),
            "slacks": " ".join(f"{state['beta'][m]:.6g}" for m in active),
            "ratios": " ".join(f"{m}: {r:.6g}" for m, r in state["ratios"].items())}


def stall_length_table(rows: list[dict]) -> str:
    """The rows of tab-stall-length.tex, headed by a comment naming every column."""
    lines = ["% per stall tier, problem "
             f"{STALL_CHECK_INDEX} of benchmark.py's stall family from the state after cycle 1: "
             "p; the seed; N_stall = min over the draining half-spaces of ceil(y_m / -beta_m); "
             "m_switch; t_switch = 1 + N_stall; the last cycle after which plain Dykstra's state "
             "repeats that state; the cycle in which half-space m_switch leaves the active set; "
             "and the stalling period the relative detector of stall_grid.py finds, which counts "
             "cycles 2 to N_stall\n"]
    for row in rows:
        lines.append(f"  ${row['p']}$ & ${row['seed']}$ & ${row['n_stall']}$ & ${row['m_switch']}$ & "
                     f"${row['t_switch']}$ & ${row['stationary_through']}$ & "
                     f"${row['switch_half_space_leaves']}$ & ${row['relative_period']}$ \\\\\n")
    return "".join(lines)


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


def run(wedge_cells: tuple = WEDGE_CELLS, stall_dimensions: tuple = STALL_DIMENSIONS,
        dense_tiers: tuple = DENSE_TIERS, problems: int = PROBLEMS, workers: int = WORKERS,
        command: str = "python -m dykstra.compaction_only") -> tuple[list[dict], list[dict]]:
    """Run the study and the stall-length check, print every number and write the outputs."""
    specs = problem_list(wedge_cells, stall_dimensions, dense_tiers, problems)
    if workers > 1:
        with ProcessPoolExecutor(max_workers=workers) as pool:
            analysed = list(pool.map(analyse, specs))
    else:
        analysed = [analyse(spec) for spec in specs]
    # the times are taken one after the other in this process, after the parallel part
    records = [time_in_c(record) for record in analysed]
    for record in records:
        print(f"{record['row']} problem {record['index']}: budget {record['budget']}, first within "
              f"{record['dykstra_first_within']} and {record['accelerated_first_within']}, exact "
              f"{record['exact_cycles']}, evaluated {record['evaluated_cycles']}, skipped by "
              f"envelope {record['envelope_skipped_cycles']}, by stall {record['stall_skipped_cycles']} "
              f"in {record['stall_fast_forwards']}, by the budget jump {record['budget_skipped_cycles']}, "
              f"largest deviation {record['largest_deviation']:.3e} "
              f"({record['largest_relative_deviation']:.3e} of max |z|), C deviation "
              f"{record['c_deviation']:.3e}, ended by {record['ended_by']}")
    if any(record["settled"] or record["c_settled"] for record in records):
        raise RuntimeError("A run under the schedule never settled.")
    write_csv(records, "compaction_only")
    rows = summary_rows(records)
    save_tex(compaction_table(rows), "tab-compaction-only", command)
    for row in rows:
        print(f"{row['row']}: {row}")
    checks = [stall_length(p) for p in stall_dimensions]
    for check in checks:
        print(f"stall check p = {check['p']}, seed {check['seed']}: entry state {check['entry_state']}, "
              f"active {check['active']}, auxiliaries {check['auxiliaries']}, slacks {check['slacks']}, "
              f"ratios {check['ratios']}; N_stall {check['n_stall']}, m_switch {check['m_switch']}, "
              f"t_switch {check['t_switch']}; Dykstra stationary through cycle "
              f"{check['stationary_through']}, m_switch leaves in cycle "
              f"{check['switch_half_space_leaves']}, relative period {check['relative_period']} "
              f"cycles; matches {check['matches']}")
    write_csv(checks, "stall_length")
    save_tex(stall_length_table(checks), "tab-stall-length", command)
    return records, checks


def main() -> None:
    """Parse the command line and run."""
    global REPEATS
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--quick", action="store_true",
                        help="one problem of one cell per family and one run per time, written to "
                             "a temporary directory")
    parser.add_argument("--workers", type=int, default=WORKERS)
    args = parser.parse_args()
    quiet_accelerate_warnings()
    command = " ".join(["python -m dykstra.compaction_only", *sys.argv[1:]])
    began = time.perf_counter()
    if args.quick:
        REPEATS = 1
        paper_figures.OUTPUT_DIR = tempfile.mkdtemp(prefix="compaction_only_")
        run(QUICK_WEDGE_CELLS, QUICK_STALL_DIMENSIONS, QUICK_DENSE_TIERS, QUICK_PROBLEMS,
            args.workers, command)
    else:
        run(workers=args.workers, command=command)
    print(f"Finished in {time.perf_counter() - began:.0f} s: tab-compaction-only.tex, "
          f"compaction_only.csv, tab-stall-length.tex and stall_length.csv in "
          f"{paper_figures.OUTPUT_DIR}")


if __name__ == "__main__":
    main()
