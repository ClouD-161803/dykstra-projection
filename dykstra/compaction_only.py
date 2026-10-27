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
