"""The stall grid of the SIAM paper's supplement, section Additional stall studies: the share of
random problems per size (n, p) with a stalling period and with an approximate stall, which
fig:sm-stall-frozen and fig:sm-stall-plateau show.

Run from the repository root with `python -m dykstra.stall_grid`, or with `--quick` for a small
grid written to a temporary directory. Every problem is a Generator 2 polyhedron from
problem_generators with its infeasible start, run by plain Dykstra for GRID_HORIZON cycles. The
figures and CSV files go to results/paper at the repository root, and the numbers the supplement
quotes are printed at the end.

run_batch, the plain Dykstra run with both detectors applied cycle by cycle, is shared with
stall_angles.py. Every choice the supplement does not state is a module constant below.

A cycle t >= 2 is a stalling cycle when the state before it is a stalling state in the sense of
the article's def:frozen and def:stallingcond. Cycle t then repeats the active set of cycle
t - 1, leaves the state unchanged, and some active half-space has a negative signed distance,
i.e. its auxiliary drains. Two variants decide "unchanged" and "negative":

- absolute: every coordinate moves by at most STATIONARY_TOLERANCE, the old figure's "1e-12",
  and some active visit input lies more than STATIONARY_TOLERANCE inside its half-space.
- relative: the repository's own tests in lti_numerics decide, as LTISolver._stall_episode does.
  is_frozen holds on the cycle map of the active set, some active increment R x + s lies below
  minus its rounding floor, and every inactive increment stays below its floor.

A stalling period is a maximal run of stalling cycles. A cycle is an approximate stall when its
squared error to the quadprog projection repeats the previous cycle's, by either of two rules:

- relative: within ConvexProjectionSolver.RELATIVE_TOLERANCE times the larger of the two, the
  rule of the current _track_error and hence of ProjectionResult.stalled_errors.
- decimals: equal once both are rounded to REPEAT_DECIMALS places, the old figure's "10 dp".

A cycle whose squared error is below CONVERGED_SQUARED_ERROR is converged and counts as neither,
as _track_error treats min_error. A stalling period is also an approximate stall, since a state
that does not move repeats its error.

For the article's fig:stallfreq, drawn with pgfplots, the script also writes grids with one row
per n and one column per p, each as CSV: the stalling-period percentages and counts, the
approximate-stall percentages of both rules, the relative approximate-stall grid minus the
stalling-period grid, and the median length of a stalling period. The three percentage grids are
also written as fig-stallgrid-*.tex matrix plots.
"""

import argparse
import csv
import os
import sys
import tempfile
import time
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

from dykstra import paper_figures
from dykstra.paper_figures import (GRID, INK, RED, quiet_accelerate_warnings, save_figure,
                                   save_tex, tex_number)
from dykstra.convex_projection_solver import ConvexProjectionSolver, DykstraProjectionSolver
from dykstra.lti_numerics import cycle_map, is_frozen, rounding_floor, unit_constraints
from dykstra.problem_generators import problem


# The grid of the supplement, read off the old heat maps: nine values of n, seven of p, the
# cells with n >= p + 1, 100 problems per cell and a 2000-cycle horizon
GRID_HALF_SPACES = (4, 6, 8, 12, 16, 24, 32, 48, 64)
GRID_DIMENSIONS = (2, 3, 4, 6, 8, 12, 16)
GRID_PROBLEMS = 100
GRID_HORIZON = 2000

# Chosen here: problem i of cell (n, p) is problem_generators.problem(n, p, seed) with seed
# GRID_SEED + 10000 p + 100 n + i, distinct over the grid since n <= 64 and i < 100
GRID_SEED = 7_000_000

# The old figure's stationarity threshold, taken as absolute, in the units of the coordinates
STATIONARY_TOLERANCE = 1e-12

# The old figure's approximate-stall rule rounds the squared error to this many places
REPEAT_DECIMALS = 10

# Chosen here: a squared error below this is converged. Above it the squared error has at least
# two significant digits on the grid of REPEAT_DECIMALS places, hence a repeat there is a repeat
# of the value rather than of the grid. stall_angles.py takes "cycles to converge" from it too
CONVERGED_SQUARED_ERROR = 1e-9

# The article's accuracy criterion, reported beside the converged share
ARTICLE_TOLERANCE = 1e-3

# Chosen here: a cycle goes to the relative stalling test only if it repeats the active set and
# moves no coordinate by more than this times 1 + max |x|, far above any rounding floor. The
# cycle map is thus built only where the state has nearly stopped
CANDIDATE_MOTION = 1e-8

DETECTORS = ("absolute", "relative")
RULES = ("relative", "decimals")

# Chosen here: the heat maps draw the stalling periods of the relative detector, which tests
# stationarity to rounding as def:frozen asks, and the approximate stalls of the decimals rule
# of the old figure. The relative rule flags almost only stalling periods and would repeat the
# first heat map
FIGURE_DETECTOR, FIGURE_RULE = "relative", "decimals"

# The pgfplots grids of fig:stallfreq take the relative rule, the definition of def:plateau, and
# keep the decimals rule in files named tendecimal
PGFPLOTS_GRIDS = (("stationary", "stalling_cycles_relative"),
                  ("approximate", "approximate_cycles_relative"),
                  ("approximate_tendecimal", "approximate_cycles_decimals"))

# Chosen here: processes of the full run
WORKERS = 8

QUICK_HALF_SPACES, QUICK_DIMENSIONS, QUICK_PROBLEMS, QUICK_HORIZON = (4, 8), (2, 3), 4, 200


def grid_seed(num_planes: int, num_dimensions: int, index: int) -> int:
    """The seed of problem number index of cell (num_planes, num_dimensions)."""
    return GRID_SEED + 10000 * num_dimensions + 100 * num_planes + index


def approximate_stall_flags(errors: np.ndarray, rule: str,
                            converged: float = CONVERGED_SQUARED_ERROR) -> np.ndarray:
    """Flags of the approximate-stall cycles of each run.

    errors holds the squared errors of cycles 0 to T along its last axis. The flags have the same
    shape, and cycle 0 and converged cycles are never flagged. Raises ValueError for a rule
    outside RULES.
    """
    previous, current = errors[..., :-1], errors[..., 1:]
    if rule == "relative":
        repeats = (np.abs(current - previous)
                   <= ConvexProjectionSolver.RELATIVE_TOLERANCE * np.maximum(current, previous))
    elif rule == "decimals":
        repeats = np.round(current, REPEAT_DECIMALS) == np.round(previous, REPEAT_DECIMALS)
    else:
        raise ValueError(f"rule must be one of {', '.join(RULES)}.")
    flags = np.zeros(errors.shape, dtype=bool)
    flags[..., 1:] = repeats & (current >= converged)
    return flags


def relative_stalling_state(unit_A: np.ndarray, unit_b: np.ndarray, active: tuple,
                            x: np.ndarray, cache: dict) -> bool:
    """Whether x is a stalling state of the active set, by the tests of lti_numerics.

    cache maps an active set to its cycle map and is filled as sets are met.
    """
    if active not in cache:
        cache[active] = cycle_map(unit_A, unit_b, active)
    cmap = cache[active]
    if not is_frozen(cmap, x):
        return False
    rows = np.array(active)
    delta = cmap.R @ x + cmap.s
    floor = rounding_floor(cmap.R_scale @ np.abs(x) + cmap.s_scale)
    return bool(np.any(rows & (delta < -floor)) and not np.any(~rows & (delta > floor)))


def run_batch(z: np.ndarray, A: np.ndarray, b: np.ndarray, reference: np.ndarray,
              horizon: int) -> dict:
    """Plain Dykstra on a batch of problems of one shape, with the stall detectors.

    z is (B, p), A is (B, n, p), b is (B, n), and reference, the projections, is (B, p). Each
    visit is that of lti_numerics.exact_cycle, vectorised over the batch. Returns per-problem
    arrays: the unit normals, the squared errors of cycles 0 to horizon, and for each stalling
    detector the number of stalling cycles, the length and active set of the longest stalling
    period, the number of stalling cycles left out as converged, and, under the key
    f"{detector}_period_lengths", a list per problem of the lengths of its stalling periods.
    """
    batch, n, p = A.shape
    unit_A, unit_b = np.empty((batch, n, p)), np.empty((batch, n))
    for i in range(batch):
        unit_A[i], unit_b[i] = unit_constraints(A[i], b[i])

    x, y = z.astype(float).copy(), np.zeros((batch, n))
    errors = np.empty((batch, horizon + 1))
    errors[:, 0] = ((x - reference) ** 2).sum(axis=1)
    active, previous_active = np.zeros((batch, n), dtype=bool), np.zeros((batch, n), dtype=bool)
    beta = np.empty((batch, n))
    stats = {d: {"cycles": np.zeros(batch, dtype=int), "run": np.zeros(batch, dtype=int),
                 "longest": np.zeros(batch, dtype=int),
                 "longest_set": np.zeros((batch, n), dtype=bool),
                 "excluded": np.zeros(batch, dtype=int),
                 "period_lengths": [[] for _ in range(batch)]} for d in DETECTORS}
    caches = [dict() for _ in range(batch)]

    for cycle in range(1, horizon + 1):
        start, start_y = x.copy(), y.copy()
        for m in range(n):
            a, offset = unit_A[:, m], unit_b[:, m]
            beta[:, m] = np.einsum("ij,ij->i", a, x) - offset
            shifted = x + y[:, m, None] * a
            slack = np.einsum("ij,ij->i", a, shifted) - offset
            active[:, m] = slack > 0.0
            x = np.where(active[:, m, None], shifted - slack[:, None] * a, shifted)
            y[:, m] = np.where(active[:, m], slack, 0.0)
        errors[:, cycle] = ((x - reference) ** 2).sum(axis=1)

        motion = np.abs(x - start).max(axis=1)
        repeats = (active == previous_active).all(axis=1) & (cycle >= 2)
        flags = {"absolute": repeats & (motion <= STATIONARY_TOLERANCE)
                 & (active & (beta < -STATIONARY_TOLERANCE)).any(axis=1),
                 "relative": np.zeros(batch, dtype=bool)}
        # a cycle that repeats the state and every auxiliary bit for bit drains nothing
        fixed = (motion == 0.0) & (y == start_y).all(axis=1)
        candidates = (repeats & ~fixed
                      & (motion <= CANDIDATE_MOTION * (1.0 + np.abs(start).max(axis=1))))
        for i in np.flatnonzero(candidates):
            flags["relative"][i] = relative_stalling_state(
                unit_A[i], unit_b[i], tuple(active[i]), start[i], caches[i])

        converged = errors[:, cycle] < CONVERGED_SQUARED_ERROR
        for detector, flag in flags.items():
            record = stats[detector]
            record["excluded"] += flag & converged
            flag = flag & ~converged
            record["cycles"] += flag
            for i in np.flatnonzero(~flag & (record["run"] > 0)):
                record["period_lengths"][i].append(int(record["run"][i]))
            record["run"] = np.where(flag, record["run"] + 1, 0)
            longer = record["run"] > record["longest"]
            record["longest"][longer] = record["run"][longer]
            # a stalling cycle repeats the previous cycle's set, thus a period has one set
            record["longest_set"][longer] = active[longer]
        previous_active[:] = active

    result = {"errors": errors, "unit_A": unit_A}
    for detector, record in stats.items():
        # a period still running at the horizon is counted with the cycles it has had
        for i in np.flatnonzero(record["run"] > 0):
            record["period_lengths"][i].append(int(record["run"][i]))
        for key in ("cycles", "longest", "longest_set", "excluded", "period_lengths"):
            result[f"{detector}_{key}"] = record[key]
    return result


def first_cycle_below(errors: np.ndarray, threshold: float) -> np.ndarray:
    """First cycle of each run whose squared error is below threshold, or -1."""
    below = errors < threshold
    return np.where(below.any(axis=1), below.argmax(axis=1), -1)


def period_rank(unit_A: np.ndarray, active_set: np.ndarray) -> int:
    """Rank of the active normals of a stalling period, or 0 for an empty set."""
    return int(np.linalg.matrix_rank(unit_A[active_set])) if active_set.any() else 0


def summarise_runs(batch: dict, num_dimensions: int) -> list[dict]:
    """One record per problem from the arrays run_batch returns."""
    errors = batch["errors"]
    approximate = {rule: approximate_stall_flags(errors, rule).sum(axis=1) for rule in RULES}
    converged_at = first_cycle_below(errors, CONVERGED_SQUARED_ERROR)
    within_at = first_cycle_below(errors, ARTICLE_TOLERANCE)
    records = []
    for i in range(errors.shape[0]):
        record = {"converged_at": int(converged_at[i]), "within_tolerance_at": int(within_at[i])}
        for rule in RULES:
            record[f"approximate_cycles_{rule}"] = int(approximate[rule][i])
        for detector in DETECTORS:
            longest_set = batch[f"{detector}_longest_set"][i]
            stalled = batch[f"{detector}_cycles"][i] > 0
            rank = period_rank(batch["unit_A"][i], longest_set) if stalled else 0
            record[f"stalling_cycles_{detector}"] = int(batch[f"{detector}_cycles"][i])
            record[f"longest_period_{detector}"] = int(batch[f"{detector}_longest"][i])
            record[f"longest_period_size_{detector}"] = int(longest_set.sum()) if stalled else 0
            record[f"longest_period_rank_{detector}"] = rank
            record[f"spans_{detector}"] = int(stalled and rank == num_dimensions)
            record[f"excluded_cycles_{detector}"] = int(batch[f"{detector}_excluded"][i])
        records.append(record)
    return records


def reference_check(z: np.ndarray, A: np.ndarray, b: np.ndarray, errors: np.ndarray,
                    horizon: int) -> dict:
    """Compare one run of run_batch with DykstraProjectionSolver on the same problem.

    Returns the largest difference between the two squared-error histories relative to 1 plus
    the error, and the number of cycles whose relative approximate-stall flag differs from
    stalled_errors.
    """
    result = DykstraProjectionSolver(z, A, b, horizon, track_error=True,
                                     min_error=CONVERGED_SQUARED_ERROR).solve()
    solver_errors = np.asarray(result.squared_errors)
    solver_flags = ~np.isnan(np.asarray(result.stalled_errors))
    flags = approximate_stall_flags(errors, "relative")
    difference = np.abs(errors - solver_errors) / (1.0 + solver_errors)
    return {"error_difference": float(difference.max()),
            "flag_disagreements": int(np.count_nonzero(flags != solver_flags))}


def grid_cell(num_planes: int, num_dimensions: int, problems: int, horizon: int) -> list[dict]:
    """Run one cell of the grid and return its per-problem records."""
    quiet_accelerate_warnings()
    began = time.perf_counter()
    seeds = [grid_seed(num_planes, num_dimensions, i) for i in range(problems)]
    drawn = [problem(num_planes, num_dimensions, seed) for seed in seeds]
    A = np.array([d[0] for d in drawn])
    b = np.array([d[1] for d in drawn])
    z = np.array([d[2] for d in drawn])
    reference = np.array([ConvexProjectionSolver._find_optimal_solution(z[i], A[i], b[i])
                          for i in range(problems)])
    batch = run_batch(z, A, b, reference, horizon)
    seconds = time.perf_counter() - began
    check = reference_check(z[0], A[0], b[0], batch["errors"][0], horizon)
    records = summarise_runs(batch, num_dimensions)
    for i, record in enumerate(records):
        record.update({"n": num_planes, "p": num_dimensions, "index": i, "seed": seeds[i],
                       "seconds_per_problem": seconds / problems})
        if i == 0:
            record.update(check)
        # kept out of summarise_runs, whose records stall_angles.py writes as well
        for detector in DETECTORS:
            lengths = batch[f"{detector}_period_lengths"][i]
            record[f"period_lengths_{detector}"] = " ".join(str(length) for length in lengths)
    return records


def grid_cells(half_spaces: tuple, dimensions: tuple) -> list[tuple[int, int]]:
    """The admissible cells, those with n >= p + 1."""
    return [(n, p) for p in dimensions for n in half_spaces if n >= p + 1]


def run_grid(half_spaces: tuple = GRID_HALF_SPACES, dimensions: tuple = GRID_DIMENSIONS,
             problems: int = GRID_PROBLEMS, horizon: int = GRID_HORIZON,
             workers: int = WORKERS) -> list[dict]:
    """Every record of the grid, the cells run in parallel."""
    cells = grid_cells(half_spaces, dimensions)
    # the largest cells go first, so that no worker is left with one at the end
    cells.sort(key=lambda cell: -cell[0] * cell[1])
    if workers <= 1:
        batches = [grid_cell(n, p, problems, horizon) for n, p in cells]
    else:
        with ProcessPoolExecutor(max_workers=workers) as pool:
            batches = list(pool.map(grid_cell, *zip(*cells), [problems] * len(cells),
                                    [horizon] * len(cells)))
    records = [record for batch in batches for record in batch]
    records.sort(key=lambda r: (r["p"], r["n"], r["index"]))
    return records


def cell_percentages(records: list[dict], field: str, half_spaces: tuple,
                     dimensions: tuple) -> np.ndarray:
    """Percentage of problems per cell whose field is nonzero, NaN outside the grid."""
    grid = np.full((len(dimensions), len(half_spaces)), np.nan)
    for row, p in enumerate(dimensions):
        for column, n in enumerate(half_spaces):
            cell = [r for r in records if r["n"] == n and r["p"] == p]
            if cell:
                grid[row, column] = 100.0 * sum(bool(r[field]) for r in cell) / len(cell)
    return grid


def write_grid_csv(grid: np.ndarray, half_spaces: tuple, dimensions: tuple, name: str) -> str:
    """Write a grid of percentages, one row per p, and return the path."""
    os.makedirs(paper_figures.OUTPUT_DIR, exist_ok=True)
    path = os.path.join(paper_figures.OUTPUT_DIR, name + ".csv")
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["p \\ n", *half_spaces])
        for p, row in zip(dimensions, grid):
            writer.writerow([p, *("" if np.isnan(v) else f"{v:.0f}" for v in row)])
    print(f"Wrote {path}")
    return path


def n_by_p_grid(records: list[dict], half_spaces: tuple, dimensions: tuple,
                statistic) -> np.ndarray:
    """statistic of each cell's records, one row per n and one column per p, NaN outside the grid."""
    grid = np.full((len(half_spaces), len(dimensions)), np.nan)
    for row, n in enumerate(half_spaces):
        for column, p in enumerate(dimensions):
            cell = [r for r in records if r["n"] == n and r["p"] == p]
            if cell:
                grid[row, column] = statistic(cell)
    return grid


def share_of(field: str):
    """The statistic giving the percentage of a cell's records whose field is nonzero."""
    return lambda cell: 100.0 * sum(bool(r[field]) for r in cell) / len(cell)


def count_of(field: str):
    """The statistic giving the number of a cell's records whose field is nonzero."""
    return lambda cell: float(sum(bool(r[field]) for r in cell))


def pooled_lengths(cell: list[dict]) -> list[int]:
    """Lengths of every stalling period of the relative detector in a cell's records."""
    return [int(length) for r in cell for length in r["period_lengths_relative"].split()]


def median_period_length(cell: list[dict]) -> float:
    """Median length of the stalling periods of a cell's records, NaN if it has none."""
    lengths = pooled_lengths(cell)
    return float(np.median(lengths)) if lengths else np.nan


def write_n_by_p_csv(grid: np.ndarray, half_spaces: tuple, dimensions: tuple, name: str) -> str:
    """Write a grid of n_by_p_grid, blank outside the grid, and return the path."""
    os.makedirs(paper_figures.OUTPUT_DIR, exist_ok=True)
    path = os.path.join(paper_figures.OUTPUT_DIR, name + ".csv")
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["n/p", *dimensions])
        for n, row in zip(half_spaces, grid):
            writer.writerow([n, *("" if np.isnan(v) else tex_number(v, 2) for v in row)])
    print(f"Wrote {path}")
    return path


def matrix_plot_pgfplots(grid: np.ndarray, half_spaces: tuple, dimensions: tuple,
                         description: str) -> str:
    """A pgfplots matrix plot of a grid of n_by_p_grid, for pasting into an axis."""
    # a matrix plot needs every cell of the lattice, hence a cell outside the grid is nan
    # rather than left out, and unbounded coords=jump leaves it empty
    scanlines = ["\n".join(f"{p} {n} {'nan' if np.isnan(v) else tex_number(v, 2)}"
                           for p, v in zip(dimensions, row))
                 for n, row in zip(half_spaces, grid)]
    return (f"% {description}\n"
            "% x is p, y is n, and meta is the percentage. A cell with n < p + 1 is not part of\n"
            "% the grid and is written as nan, which unbounded coords=jump leaves empty. Each\n"
            "% scanline holds one n, and both axes in log mode give cells of nearly equal size\n"
            "\\addplot[matrix plot*, point meta=explicit, unbounded coords=jump]\n"
            "  table[x=p, y=n, meta=value] {\n"
            "p n value\n" + "\n\n".join(scanlines) + "\n};\n")


def write_records(records: list[dict], name: str) -> str:
    """Write every per-problem record as CSV and return the path."""
    os.makedirs(paper_figures.OUTPUT_DIR, exist_ok=True)
    path = os.path.join(paper_figures.OUTPUT_DIR, name + ".csv")
    fields = list(dict.fromkeys(key for record in records for key in record))
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(records)
    print(f"Wrote {path}")
    return path


def heatmap_figure(grid: np.ndarray, half_spaces: tuple, dimensions: tuple, title: str,
                   name: str) -> None:
    """Draw an annotated heat map of percentages, with the cells outside the grid in grey."""
    colours = LinearSegmentedColormap.from_list("stall", ["#ffffff", "#f2b8a8", RED, "#7f1718"])
    colours.set_bad(GRID)
    fig, ax = plt.subplots(figsize=(8.0, 4.6))
    image = ax.imshow(np.ma.masked_invalid(grid), cmap=colours, vmin=0, vmax=100, aspect="auto")
    for row in range(grid.shape[0]):
        for column in range(grid.shape[1]):
            value = grid[row, column]
            if not np.isnan(value):
                ax.text(column, row, f"{value:.0f}", ha="center", va="center", fontsize=8,
                        color="white" if value >= 50 else INK)
    ax.set_xticks(range(len(half_spaces)), [str(n) for n in half_spaces])
    ax.set_yticks(range(len(dimensions)), [str(p) for p in dimensions])
    ax.set_xlabel("number of half-spaces $n$")
    ax.set_ylabel("dimension $p$")
    ax.set_title(title, fontsize=11)
    for side in ax.spines.values():
        side.set_visible(False)
    bar = fig.colorbar(image, ax=ax)
    bar.set_label("% of problems")
    bar.outline.set_visible(False)
    fig.tight_layout()
    save_figure(fig, name)


def report(records: list[dict], half_spaces: tuple, dimensions: tuple,
           horizon: int = GRID_HORIZON) -> dict:
    """Write the CSV files and both heat maps, print every quoted number, and return them."""
    shape = (half_spaces, dimensions)
    stalling = {d: cell_percentages(records, f"stalling_cycles_{d}", *shape) for d in DETECTORS}
    approximate = {r: cell_percentages(records, f"approximate_cycles_{r}", *shape) for r in RULES}
    write_grid_csv(stalling["relative"], *shape, "stall_grid_stalling_period")
    write_grid_csv(stalling["absolute"], *shape, "stall_grid_stalling_period_absolute")
    write_grid_csv(approximate["decimals"], *shape, "stall_grid_approximate_stall")
    write_grid_csv(approximate["relative"], *shape, "stall_grid_approximate_stall_relative")
    write_records(records, "stall_grid_problems")
    heatmap_figure(stalling[FIGURE_DETECTOR], *shape, "Problems with a stalling period (%)",
                   "stall_grid_frozen")
    heatmap_figure(approximate[FIGURE_RULE], *shape, "Problems with an approximate stall (%)",
                   "stall_grid_plateau")

    total = len(records)
    numbers = {"problems": total, "stalling": stalling, "approximate": approximate}
    for detector in DETECTORS:
        stalled = [r for r in records if r[f"stalling_cycles_{detector}"]]
        spanning = sum(r[f"spans_{detector}"] for r in stalled)
        converged_only = sum(1 for r in records if r[f"excluded_cycles_{detector}"]
                             and not r[f"stalling_cycles_{detector}"])
        numbers[f"stalled_{detector}"] = len(stalled)
        numbers[f"spanning_{detector}"] = spanning
        numbers[f"converged_only_{detector}"] = converged_only
        print(f"stalling period, {detector} detector: {len(stalled)} of {total} problems "
              f"({100.0 * len(stalled) / total:.1f}%). The active normals of the longest period "
              f"span R^p in {spanning} of them. {converged_only} more problems stall only at a "
              f"converged state, which would make {len(stalled) + converged_only} if counted")
        if len(stalled) > spanning:
            ranks = sorted({(r[f"longest_period_rank_{detector}"], r["p"]) for r in stalled
                            if not r[f"spans_{detector}"]})
            print(f"  (rank, p) of the longest periods that do not span: {ranks}")
    both = sum(1 for r in records
               if r["stalling_cycles_absolute"] and r["stalling_cycles_relative"])
    print(f"problems with a stalling period under both detectors: {both}")
    for rule in RULES:
        flagged = sum(1 for r in records if r[f"approximate_cycles_{rule}"])
        unnested = sum(1 for r in records
                       if r["stalling_cycles_relative"] and not r[f"approximate_cycles_{rule}"])
        numbers[f"approximate_{rule}"] = flagged
        print(f"approximate stall, {rule} rule: {flagged} of {total} problems "
              f"({100.0 * flagged / total:.1f}%). {unnested} problems with a stalling period "
              f"(relative detector) have none")

    largest = [r for r in records if r["n"] == max(half_spaces) and r["p"] == max(dimensions)]
    for label, field in ((f"squared error below {CONVERGED_SQUARED_ERROR:g}", "converged_at"),
                         (f"squared error at most {ARTICLE_TOLERANCE:g}", "within_tolerance_at")):
        share = 100.0 * sum(r[field] >= 0 for r in largest) / len(largest)
        numbers[f"largest_{field}"] = share
        print(f"{max(half_spaces)} x {max(dimensions)} problems reaching {label} within "
              f"{horizon} cycles: {share:.0f}% of {len(largest)}")
    for ratio in (2, 4):
        figure_grid = stalling[FIGURE_DETECTOR]
        values = [f"{n}x{p} {figure_grid[dimensions.index(p), half_spaces.index(n)]:.0f}"
                  for p in dimensions for n in half_spaces if n == ratio * p]
        print(f"stalling-period percentages along n/p = {ratio}: {', '.join(values)}")
    checks = [r for r in records if "error_difference" in r]
    print(f"against DykstraProjectionSolver on problem 0 of each of {len(checks)} cells: "
          f"largest squared-error difference "
          f"{max(r['error_difference'] for r in checks):.2e} relative to 1 plus the error, and "
          f"{sum(r['flag_disagreements'] for r in checks)} cycles whose relative "
          f"approximate-stall flag differs from stalled_errors")
    seconds = [r["seconds_per_problem"] for r in records]
    numbers["seconds_per_problem"] = float(np.mean(seconds))
    print(f"batch run time per problem: mean {np.mean(seconds):.3f} s, largest cell "
          f"{max(seconds):.3f} s")
    numbers.update(report_stallfreq(records, half_spaces, dimensions, horizon))
    return numbers


def report_stallfreq(records: list[dict], half_spaces: tuple, dimensions: tuple,
                     horizon: int) -> dict:
    """Write the n-by-p grids and matrix plots of fig:stallfreq, print their numbers, return them."""
    shape = (half_spaces, dimensions)
    total, cells = len(records), len(grid_cells(*shape))
    per_cell = total // cells
    descriptions = {
        "stationary": "a stalling period (relative detector)",
        "approximate": f"an approximate stall, the squared error of consecutive cycles repeating to "
                       f"{ConvexProjectionSolver.RELATIVE_TOLERANCE:g} of the larger",
        "approximate_tendecimal": f"an approximate stall, the squared error of consecutive cycles "
                                  f"repeating once rounded to {REPEAT_DECIMALS} decimal places",
    }
    grids = {}
    for label, field in PGFPLOTS_GRIDS:
        grids[label] = n_by_p_grid(records, *shape, share_of(field))
        write_n_by_p_csv(grids[label], *shape, f"stall_grid_{label}_percent")
        description = (f"Percentage of the {per_cell} problems per cell with "
                       f"{descriptions[label]}, over a budget of {horizon} cycles")
        save_tex(matrix_plot_pgfplots(grids[label], *shape, description),
                 "fig-stallgrid-" + label.replace("_", "-"), "python -m dykstra.stall_grid")
    counts = n_by_p_grid(records, *shape, count_of("stalling_cycles_relative"))
    write_n_by_p_csv(counts, *shape, "stall_grid_stationary_counts")
    difference = grids["approximate"] - grids["stationary"]
    write_n_by_p_csv(difference, *shape, "stall_grid_approximate_minus_stationary")
    medians = n_by_p_grid(records, *shape, median_period_length)
    write_n_by_p_csv(medians, *shape, "stall_grid_median_period_length")

    flagged = {label: sum(1 for r in records if r[field]) for label, field in PGFPLOTS_GRIDS}
    absolute = np.abs(difference[~np.isnan(difference)])
    row, column = np.unravel_index(np.nanargmax(np.abs(difference)), difference.shape)
    lengths = pooled_lengths(records)
    numbers = {"stallfreq_counts": flagged,
               "stallfreq_largest_difference": float(absolute.max()),
               "stallfreq_mean_difference": float(absolute.mean()),
               "stallfreq_median_period": float(np.median(lengths)) if lengths else None,
               "stallfreq_periods": len(lengths)}
    print(f"fig:stallfreq, {total} problems in {cells} cells of {per_cell}: "
          f"{flagged['stationary']} with a stalling period (relative detector), "
          f"{flagged['approximate']} with an approximate stall under the relative rule "
          f"(RELATIVE_TOLERANCE {ConvexProjectionSolver.RELATIVE_TOLERANCE:g}), and "
          f"{flagged['approximate_tendecimal']} under the {REPEAT_DECIMALS}-decimal rule")
    print(f"relative approximate-stall grid minus stalling-period grid: largest absolute "
          f"difference {absolute.max():.0f} percentage points, at {half_spaces[row]} x "
          f"{dimensions[column]}, mean absolute difference {absolute.mean():.2f} points over "
          f"{absolute.size} cells, {int(np.count_nonzero(absolute))} cells differ")
    if lengths:
        print(f"stalling periods (relative detector): {len(lengths)} in {flagged['stationary']} "
              f"problems, median length {np.median(lengths):g} cycles, per-cell medians from "
              f"{np.nanmin(medians):g} to {np.nanmax(medians):g}")
    return numbers


def main() -> None:
    """Parse the command line and run."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--quick", action="store_true",
                        help="a small grid written to a temporary directory")
    parser.add_argument("--workers", type=int, default=WORKERS)
    args = parser.parse_args()
    quiet_accelerate_warnings()
    began = time.perf_counter()
    if args.quick:
        paper_figures.OUTPUT_DIR = tempfile.mkdtemp(prefix="stall_grid_")
        shape, problems, horizon = (QUICK_HALF_SPACES, QUICK_DIMENSIONS), QUICK_PROBLEMS, QUICK_HORIZON
    else:
        shape, problems, horizon = (GRID_HALF_SPACES, GRID_DIMENSIONS), GRID_PROBLEMS, GRID_HORIZON
    records = run_grid(*shape, problems, horizon, args.workers)
    report(records, *shape, horizon)
    print(f"Finished in {time.perf_counter() - began:.0f} s: stall_grid_frozen and "
          f"stall_grid_plateau as PNG and PDF, the three fig-stallgrid-*.tex matrix plots, and "
          f"the stall_grid CSV files, in {paper_figures.OUTPUT_DIR}")


if __name__ == "__main__":
    sys.exit(main())
