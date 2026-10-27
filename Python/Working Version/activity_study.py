"""The activity study of the SIAM paper's supplement, section sm-activity, and its nine figures.

Run from this folder with `python activity_study.py` for the full study, about four minutes on
ten worker processes of an Apple M5 Pro, or with `--quick` for a reduced run into a temporary
directory. activity_figures.py draws the figures into results/paper beside this script as PNG and
PDF, every number the supplement quotes from them is printed, and all of them are stored in
results/paper/activity_numbers.json.

Activity follows eq:activeset of the article: half-space i is active in cycle t when the point
it is visited with, shifted by its auxiliary, violates it, which leaves y_i > 0 after the visit.
A change of half-space i in cycle t is a difference between the sets of cycles t - 1 and t, for
t >= 2, since eq:activeset defines no set before cycle 1. The counts that also compare cycle 1
with the half-spaces z violates, the set both solver families record as cycle 0, are printed
beside as the counts with cycle 1.

The runs are plain Dykstra in the scalar auxiliaries of lti_numerics.exact_cycle, vectorised over
a batch of problems of one size by record_runs, which tests/test_paper_outputs.py holds to
exact_cycle, and which each task checks again on its first problem. The core solver's
active_half_spaces is not used, because it tests the cycle's end point rather than eq:activeset.
A run stops at an exact fixed point of its state (x, y), the stop rule of oracle.record_schedule,
after which every cycle repeats, or else at its horizon. It counts as converged when it ends
within CONVERGED_DISTANCE of the quadprog projection, relative to the distance from z.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
import json
import os
import tempfile
import time
import warnings

import numpy as np

from convex_projection_solver import ConvexProjectionSolver
from lti_numerics import cycle_map, exact_cycle, unit_constraints
from problem_generators import RandomTestPlanes, infeasible_start, problem


# Every constant below that the supplement does not state was chosen here.
# Sizes n x p of the activity curves and their horizon, both from the old figures' legend and
# axis, and the problems per size
CURVE_SIZES = ((6, 2), (8, 3), (10, 3), (12, 4), (16, 5), (20, 6))
CURVE_HORIZON = 10_000
CURVE_PROBLEMS = 1000
# The single run shows the first SINGLE_CYCLES cycles of the first 6 x 2 problem of the curve
# battery whose activity changes at least three times after cycle 1, the last time after cycle 5,
# as the old figure's run did, and of problem SINGLE_INDEX when none does
SINGLE_SIZE, SINGLE_CYCLES = (6, 2), 29
SINGLE_INDEX = 0
# The sweep behind the extreme example: the prose's 2400 problems, all at the old figure's n = 8
EXAMPLE_SIZE, EXAMPLE_PROBLEMS, EXAMPLE_HORIZON = (8, 2), 2400, 40_000
# The 2-D sweep: the old figure's n, problems per n and horizons. A run is counted to its fixed
# point or to LIMIT_HORIZON, and LIMIT_FIRST_HORIZON is where the old sweep stopped first
LIMIT_SIZES = (4, 8, 16, 32, 64, 128)
LIMIT_PROBLEMS, LIMIT_FIRST_HORIZON, LIMIT_HORIZON = 4000, 4000, 40_000
# The trend sweep: the old figures' grid of p and ratios n / p, problems per size, and horizon
TREND_DIMENSIONS = (2, 3, 4, 6, 8, 10, 12, 16, 20)
TREND_RATIOS = (2, 4)
TREND_PROBLEMS, TREND_HORIZON = 1000, 100_000
# Distance from the projection, relative to the distance from z, within which a run converged
CONVERGED_DISTANCE = 1e-8
# Slack at the projection, relative to the magnitudes summed in it, within which a row is tight
TIGHT_TOLERANCE = 1e-8
# Cycles of each task's first run that are checked against exact_cycle
CHECK_CYCLES = 2000
# Problems per worker task, and worker processes
BATCH, WORKERS = 250, 10

# The settings --quick and the test replace, for a run of a few seconds on one process
QUICK = {
    "CURVE_SIZES": ((6, 2), (8, 3)), "CURVE_HORIZON": 300, "CURVE_PROBLEMS": 24,
    "EXAMPLE_PROBLEMS": 40, "EXAMPLE_HORIZON": 2000, "LIMIT_SIZES": (4, 8),
    "LIMIT_PROBLEMS": 30, "LIMIT_FIRST_HORIZON": 200, "LIMIT_HORIZON": 2000,
    "TREND_DIMENSIONS": (2, 3, 4), "TREND_PROBLEMS": 12, "TREND_HORIZON": 3000,
    "CHECK_CYCLES": 200, "BATCH": 16,
}


def quiet_accelerate_warnings() -> None:
    """Silence the matmul warnings numpy 2.1 raises with Apple's Accelerate library."""
    # paper_figures holds the same filter, but importing it loads matplotlib into every worker
    warnings.filterwarnings("ignore", message=r".* encountered in matmul", category=RuntimeWarning)


def study_seed(num_planes: int, num_dimensions: int, index: int) -> int:
    """The seed of problem number index of size num_planes x num_dimensions, for index < 10000."""
    # problem_generators.problem_seed repeats a seed across p and past index 999
    return 10_000 * (1000 * num_dimensions + num_planes) + index


def draw_problem(generator: int, num_planes: int, num_dimensions: int, seed: int) -> tuple:
    """Constraints from Generator 1 or 2 and an infeasible start, from one seeded stream.

    Returns (A, b, z).
    """
    if generator == 2:
        return problem(num_planes, num_dimensions, seed)
    rng = np.random.RandomState(seed)
    A, b = RandomTestPlanes(num_planes, num_dimensions).draw(rng)
    return A, b, infeasible_start(A, b, rng)


@dataclass
class Run:
    """One recorded run of Dykstra's algorithm.

    masks[k] is the active set from cycle cycles[k] until the next entry, and cycles[0] is 0.
    tight marks the rows tight at the quadprog projection. converged_at is the first cycle from
    which the iterate stays within the convergence distance.
    """
    seed: int
    cycles: np.ndarray
    masks: np.ndarray
    cycles_run: int
    fixed_at: int | None
    converged: bool
    converged_at: int | None
    tight: np.ndarray


def record_runs(problems: list, horizon: int, seeds: list | None = None) -> list:
    """Run plain Dykstra on problems (A, b, z) of one size and record their activity.

    Each run stops at an exact fixed point of its state or after horizon cycles, and seeds label
    the runs.
    """
    count = len(problems)
    seeds = list(range(count)) if seeds is None else list(seeds)
    units = [unit_constraints(A, b) for A, b, _ in problems]
    # Row m of every problem is contiguous, since a visit reads one row of each
    UA = np.ascontiguousarray(np.stack([u[0] for u in units], axis=1))
    UB = np.ascontiguousarray(np.stack([u[1] for u in units], axis=1))
    Z = np.stack([np.asarray(z, dtype=float) for *_, z in problems])
    x_star = np.stack([ConvexProjectionSolver._find_optimal_solution(z, A, b)
                       for A, b, z in problems])
    tolerance = (CONVERGED_DISTANCE * np.linalg.norm(Z - x_star, axis=1)) ** 2
    slack = np.einsum("mbp,bp->bm", UA, x_star) - UB.T
    scale = np.einsum("mbp,bp->bm", np.abs(UA), np.abs(x_star)) + np.abs(UB.T)
    tight = np.abs(slack) <= TIGHT_TOLERANCE * scale
    n = UA.shape[0]

    X, Y = Z.copy(), np.zeros((n, count))
    active = np.einsum("mbp,bp->mb", UA, Z) - UB > 0.0
    entries = [[(0, active[:, j].copy())] for j in range(count)]
    last_above = np.zeros(count, dtype=int)
    fixed_at = np.full(count, -1)
    final_x = np.zeros_like(Z)
    alive, XS, TOL = np.arange(count), x_star.copy(), tolerance.copy()
    for t in range(1, horizon + 1):
        X_previous, Y_previous = X, Y.copy()
        for m in range(n):
            normal = UA[m]
            shifted = X + Y[m][:, None] * normal
            excess = np.einsum("ij,ij->i", normal, shifted) - UB[m]
            step = np.where(excess > 0.0, excess, 0.0)
            X = shifted - step[:, None] * normal
            Y[m] = step
        new_active = Y > 0.0
        for k in np.flatnonzero((new_active != active).any(axis=0)):
            entries[alive[k]].append((t, new_active[:, k].copy()))
        active = new_active
        last_above[alive[((X - XS) ** 2).sum(axis=1) > TOL]] = t

        # A fixed run repeats its state, hence it is dropped from the batch only now and then
        fixed = (X == X_previous).all(axis=1) & (Y == Y_previous).all(axis=0)
        newly = fixed & (fixed_at[alive] < 0)
        fixed_at[alive[newly]] = t
        final_x[alive[newly]] = X[newly]
        if fixed.all() or (fixed.any() and t % 64 == 0):
            keep = ~fixed
            X, Y, active = X[keep], Y[:, keep], active[:, keep]
            UA, UB, XS, TOL = UA[:, keep], UB[:, keep], XS[keep], TOL[keep]
            alive = alive[keep]
            if alive.size == 0:
                break
    final_x[alive] = X

    runs = []
    for j in range(count):
        converged = bool(((final_x[j] - x_star[j]) ** 2).sum() <= tolerance[j])
        runs.append(Run(seed=seeds[j], cycles=np.array([c for c, _ in entries[j]]),
                        masks=np.array([mask for _, mask in entries[j]]),
                        cycles_run=int(fixed_at[j]) if fixed_at[j] >= 0 else horizon,
                        fixed_at=int(fixed_at[j]) if fixed_at[j] >= 0 else None,
                        converged=converged,
                        converged_at=int(last_above[j]) + 1 if converged else None,
                        tight=tight[j]))
    return runs


def exact_entries(A: np.ndarray, b: np.ndarray, z: np.ndarray, cycles: int) -> tuple:
    """Active-set entries of a run of exact_cycle, as Run.cycles and Run.masks, and its points.

    points[t, m] is the point after visit m of cycle t, and points[0, -1] is z.
    """
    unit_A, unit_b = unit_constraints(A, b)
    x, y = np.asarray(z, dtype=float).copy(), np.zeros(len(b))
    previous = unit_A @ x - unit_b > 0.0
    entries, points = [(0, previous)], np.empty((cycles + 1, len(b), len(x)))
    points[0] = x
    for t in range(1, cycles + 1):
        x, y, points[t] = exact_cycle(x, y, unit_A, unit_b)
        if not np.array_equal(y > 0.0, previous):
            previous = y > 0.0
            entries.append((t, previous))
    return np.array([c for c, _ in entries]), np.array([mask for _, mask in entries]), points


def _changes(run: Run, opening: bool) -> np.ndarray:
    """Which half-spaces change at each entry after the first, with or without cycle 1."""
    changes = run.masks[1:] != run.masks[:-1]
    if not opening:
        changes = changes & (run.cycles[1:] >= 2)[:, None]
    return changes


def change_counts(run: Run, opening: bool = False) -> np.ndarray:
    """Activity changes of each half-space, from cycle 2 or, with opening, from cycle 1."""
    return _changes(run, opening).sum(axis=0)


def change_cycles(run: Run, half_space: int, opening: bool = False) -> np.ndarray:
    """The cycles in which one half-space changed activity."""
    return run.cycles[1:][_changes(run, opening)[:, half_space]]


def sampled(run: Run, cycles: np.ndarray, opening: bool = False) -> tuple:
    """The active count and the cumulative change count of a run at the given cycles."""
    index = np.searchsorted(run.cycles, cycles, side="right") - 1
    cumulative = np.concatenate([[0], np.cumsum(_changes(run, opening).sum(axis=1))])
    return run.masks.sum(axis=1)[index], cumulative[index]


def span_eigenvalues(unit_A: np.ndarray, unit_b: np.ndarray, active: np.ndarray) -> tuple:
    """Eigenvalues of A_m on the span of the active normals, and the unit eigenvalues left out.

    Off that span A_m is the identity, hence its other p - r eigenvalues are 1 exactly. The
    block on the span is the matrix T of the article's deflated solve, and it contracts.
    """
    cmap = cycle_map(unit_A, unit_b, tuple(bool(a) for a in active))
    span = cmap.span
    if span.shape[1] == 0:
        return np.empty(0, dtype=complex), unit_A.shape[1]
    values = np.linalg.eigvals(span.T @ cmap.A_m @ span).astype(complex)
    return values, unit_A.shape[1] - span.shape[1]


def _battery_task(task: tuple) -> dict:
    """Record one batch of problems, check its first run against exact_cycle, and optionally
    collect the eigenvalues of every episode."""
    quiet_accelerate_warnings()
    generator, n, p, seeds, horizon, eigenvalues, check_cycles = task
    started = time.perf_counter()
    problems = [draw_problem(generator, n, p, seed) for seed in seeds]
    drawn = time.perf_counter()
    runs = record_runs(problems, horizon, seeds)
    recorded = time.perf_counter()

    checked = min(check_cycles, runs[0].cycles_run)
    cycles, masks, _ = exact_entries(*problems[0], checked)
    kept = runs[0].cycles <= checked
    agrees = (np.array_equal(cycles, runs[0].cycles[kept])
              and np.array_equal(masks, runs[0].masks[kept]))

    result = {"runs": runs, "agrees": bool(agrees), "draw_seconds": drawn - started,
              "record_seconds": recorded - drawn}
    if eigenvalues:
        values, units, episodes, empty = [], 0, 0, 0
        for (A, b, _), run in zip(problems, runs):
            unit_A, unit_b = unit_constraints(A, b)
            for cycle, mask in zip(run.cycles, run.masks):
                if cycle == 0:
                    continue
                episodes += 1
                if not mask.any():
                    empty += 1
                    continue
                spectrum, left_out = span_eigenvalues(unit_A, unit_b, mask)
                values.append(spectrum)
                units += left_out
        result.update(eigenvalues=np.concatenate(values) if values else np.empty(0, complex),
                      unit_eigenvalues=units, episodes=episodes, empty_episodes=empty)
    return result


def run_batteries(specs: list, workers: int, batch: int, check_cycles: int) -> list:
    """Record batteries given as (generator, n, p, problems, horizon, eigenvalues).

    Returns, per battery, its runs in seed order, the merged task results, and the timings.
    """
    tasks, owners = [], []
    for k, (generator, n, p, count, horizon, eigenvalues) in enumerate(specs):
        seeds = [study_seed(n, p, index) for index in range(count)]
        for start in range(0, count, batch):
            tasks.append((generator, n, p, seeds[start:start + batch], horizon, eigenvalues,
                          check_cycles))
            owners.append(k)
    # The costliest tasks start first, so that none is left running alone at the end
    order = sorted(range(len(tasks)), key=lambda k: -tasks[k][1] * tasks[k][4])
    if workers > 1:
        with ProcessPoolExecutor(workers, initializer=quiet_accelerate_warnings) as pool:
            done = dict(zip(order, pool.map(_battery_task, [tasks[k] for k in order])))
    else:
        done = {k: _battery_task(tasks[k]) for k in order}

    batteries = [{"runs": [], "checks": 0, "check_failures": 0, "draw_seconds": 0.0,
                  "record_seconds": 0.0, "eigenvalues": [], "unit_eigenvalues": 0,
                  "episodes": 0, "empty_episodes": 0} for _ in specs]
    for k in range(len(tasks)):
        battery, result = batteries[owners[k]], done[k]
        battery["runs"] += result["runs"]
        battery["checks"] += 1
        battery["check_failures"] += not result["agrees"]
        battery["draw_seconds"] += result["draw_seconds"]
        battery["record_seconds"] += result["record_seconds"]
        if "eigenvalues" in result:
            battery["eigenvalues"].append(result["eigenvalues"])
            for key in ("unit_eigenvalues", "episodes", "empty_episodes"):
                battery[key] += result[key]
    for battery in batteries:
        battery["eigenvalues"] = (np.concatenate(battery["eigenvalues"])
                                  if battery["eigenvalues"] else np.empty(0, complex))
    return batteries


def settings(quick: bool) -> dict:
    """The study's parameters, reduced by QUICK for a quick run."""
    chosen = {name: globals()[name] for name in (
        "CURVE_SIZES", "CURVE_HORIZON", "CURVE_PROBLEMS", "SINGLE_SIZE", "SINGLE_CYCLES",
        "EXAMPLE_SIZE", "EXAMPLE_PROBLEMS", "EXAMPLE_HORIZON", "LIMIT_SIZES", "LIMIT_PROBLEMS",
        "LIMIT_FIRST_HORIZON", "LIMIT_HORIZON", "TREND_DIMENSIONS", "TREND_RATIOS",
        "TREND_PROBLEMS", "TREND_HORIZON", "CHECK_CYCLES", "BATCH", "WORKERS")}
    if quick:
        chosen.update(QUICK)
    return chosen


def battery_timing(battery: dict) -> dict:
    """Seconds per problem spent drawing and recording, summed over the worker processes."""
    count = len(battery["runs"])
    return {"problems": count, "draw_ms_per_problem": 1e3 * battery["draw_seconds"] / count,
            "record_ms_per_problem": 1e3 * battery["record_seconds"] / count,
            "recorder_checks": battery["checks"],
            "recorder_check_failures": battery["check_failures"]}


def _plain(value):
    """value with numpy scalars and arrays turned into JSON types."""
    if isinstance(value, dict):
        return {str(key): _plain(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, np.ndarray)):
        return [_plain(item) for item in value]
    if isinstance(value, np.generic):
        return value.item()
    return value


def run(quick: bool = False, workers: int | None = None) -> dict:
    """Record every battery, draw the nine figures, print and store the numbers they rest on.

    The batteries run cheapest first, and the numbers file is rewritten after each, hence an
    interrupted run leaves the figures and numbers of the batteries it finished.
    """
    import activity_figures as figures
    import paper_figures

    quiet_accelerate_warnings()
    cfg = settings(quick)
    workers = cfg["WORKERS"] if workers is None else workers
    numbers = {"settings": dict(cfg, CONVERGED_DISTANCE=CONVERGED_DISTANCE,
                                TIGHT_TOLERANCE=TIGHT_TOLERANCE), "seconds": {}}
    path = os.path.join(paper_figures.OUTPUT_DIR, "activity_numbers.json")

    def record(specs: list) -> list:
        return run_batteries(specs, workers, cfg["BATCH"], cfg["CHECK_CYCLES"])

    def finish(name: str, started: float) -> None:
        numbers["seconds"][name] = time.perf_counter() - started
        print(f"{name}: {json.dumps(_plain(numbers[name]))}", flush=True)
        os.makedirs(paper_figures.OUTPUT_DIR, exist_ok=True)
        with open(path, "w", encoding="utf-8") as handle:
            json.dump(_plain(numbers), handle, indent=1)

    started = time.perf_counter()
    sizes = cfg["CURVE_SIZES"]
    batteries = record([(1, n, p, cfg["CURVE_PROBLEMS"], cfg["CURVE_HORIZON"], True)
                        for n, p in sizes])
    numbers["seconds"]["curve battery"] = time.perf_counter() - started
    started = time.perf_counter()
    single_runs = batteries[sizes.index(cfg["SINGLE_SIZE"])]["runs"]
    single = figures.single_run(figures.single_run_index(single_runs), cfg)
    figures.single_run_figure(single)
    numbers["single"] = {k: v for k, v in single.items() if k not in ("residuals", "active")}
    finish("single", started)

    started = time.perf_counter()
    numbers["eigenvalues"], spectrum = figures.eigenvalue_numbers(batteries, sizes)
    figures.eigenvalue_figure(spectrum)
    finish("eigenvalues", started)

    started = time.perf_counter()
    numbers["curves"], curves = [], {}
    for (n, p), battery in zip(sizes, batteries):
        row, curves[(n, p)] = figures.curve_numbers(battery["runs"], n, p, cfg["CURVE_HORIZON"])
        numbers["curves"].append({**row, **battery_timing(battery)})
    figures.curve_figures(curves)
    finish("curves", started)

    started = time.perf_counter()
    n, p = cfg["EXAMPLE_SIZE"]
    battery, = record([(1, n, p, cfg["EXAMPLE_PROBLEMS"], cfg["EXAMPLE_HORIZON"], False)])
    numbers["example"], example = figures.example_numbers(battery["runs"], cfg)
    numbers["example"].update(battery_timing(battery))
    figures.example_figure(example)
    finish("example", started)

    started = time.perf_counter()
    batteries = record([(2, n, 2, cfg["LIMIT_PROBLEMS"], cfg["LIMIT_HORIZON"], False)
                        for n in cfg["LIMIT_SIZES"]])
    numbers["limit"] = [figures.limit_numbers(battery, n, cfg)
                        for n, battery in zip(cfg["LIMIT_SIZES"], batteries)]
    figures.limit_figure(numbers["limit"], cfg)
    finish("limit", started)

    started = time.perf_counter()
    sizes = [(ratio * p, p) for ratio in cfg["TREND_RATIOS"] for p in cfg["TREND_DIMENSIONS"]]
    batteries = record([(2, n, p, cfg["TREND_PROBLEMS"], cfg["TREND_HORIZON"], False)
                        for n, p in sizes])
    numbers["trend"] = [figures.trend_numbers(battery, n, p)
                        for (n, p), battery in zip(sizes, batteries)]
    numbers["trend_fits"], numbers["trend_fits_with_cycle_1"] = {}, {}
    for ratio in cfg["TREND_RATIOS"]:
        rows = [row for row in numbers["trend"] if row["n"] == ratio * row["p"]]
        for key, fits in (("total_mean", "trend_fits"),
                          ("total_mean_with_cycle_1", "trend_fits_with_cycle_1")):
            numbers[fits][f"n = {ratio}p"] = figures.power_fit([row["p"] for row in rows],
                                                               [row[key] for row in rows])
    figures.trend_figures(numbers["trend"], numbers["trend_fits"], cfg)
    finish("trend", started)
    figures.print_quoted(numbers)
    print(f"Wrote the nine activity figures and {path}", flush=True)
    return numbers


def main() -> None:
    """Run the study from the command line."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--quick", action="store_true",
                        help="reduced batteries, written to a temporary directory")
    parser.add_argument("--workers", type=int, default=None, help=f"processes, {WORKERS} by default")
    arguments = parser.parse_args()
    if arguments.quick:
        import paper_figures
        paper_figures.OUTPUT_DIR = tempfile.mkdtemp(prefix="activity_study_")
    run(arguments.quick, arguments.workers)


if __name__ == "__main__":
    main()
