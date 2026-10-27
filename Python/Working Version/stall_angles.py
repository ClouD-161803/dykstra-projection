"""The angle sweeps and the accelerated stall study of the SIAM paper's supplement, section
Additional stall studies: fig:sm-angle-cycles, fig:sm-angle-stallcycles, fig:sm-angle-capture and
fig:sm-acc-stall.

Run from this folder with `python stall_angles.py`, or with `--quick` for small sizes written to a
temporary directory, or with `--only accelerated` to redo the accelerated study alone. The
figures and CSV files go to results/paper beside this script, and the numbers the supplement
quotes are printed at the end. Plain Dykstra runs through stall_grid.run_batch, with the stall
detectors and the converged threshold of stall_grid.py.

The supplement does not define either family, and this module defines both. Each is built with
its corner at the origin, then turned by a Haar-distributed rotation Q and moved to a corner v
drawn uniformly from [-CORNER_HALF_WIDTH, CORNER_HALF_WIDTH]^p. The corner rows come first.

- wedge: two half-spaces whose unit normals (cos(theta/2), +-sin(theta/2), 0, ..., 0) meet at
  angle theta, as in the article's lem:wedge, and the box |x_j| <= BOX_HALF_WIDTH on each of the
  remaining p - 2 coordinates, 2 + 2 (p - 2) rows in all. The box keeps the region bounded off
  the wedge's plane and stays inactive, hence the active normals span only that plane and every
  episode is singular for p >= 3.
- vertex: p half-spaces through one point whose unit normals cos(alpha) u + sin(alpha) w_i lie
  around the axis u = (1, ..., 1) / sqrt(p), where the w_i are the vertices of a regular simplex
  orthogonal to u. alpha is set so that every two normals meet at angle theta, which is then the
  smallest angle between them, as for the wedge. The normals span R^p, the full-rank branch.

A canonical start is the corner plus START_DISTANCE times a unit direction in the corner's
normal cone, the normalised combination of the corner normals with weights uniform on the
probability simplex. The projection is then the corner by construction, and every corner row is
active at the limit. A wedge start also moves off the plane by an offset uniform in half the box,
which the projection keeps. The problems of one cell differ in Q, v, the weights, and the offset.

A corner-approaching start is the corner plus START_DISTANCE times a uniformly random unit
direction, redrawn until it lies outside at least one corner half-space. It is captured when its
projection is the corner, i.e. when it lies in the corner's normal cone, which a least-squares
solve in the corner normals decides without running Dykstra. A captured start is then run like a
canonical one.
"""

import argparse
import sys
import tempfile
import time
from collections import namedtuple
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

import c_solvers
import paper_figures
import performance_timing
from paper_figures import BLUE, GREY, INK, RED, quiet_accelerate_warnings, save_figure, style_axes
from convex_projection_solver import ConvexProjectionSolver, DykstraProjectionSolver
from lti_solver import LTIVer5Solver
from stall_grid import CONVERGED_SQUARED_ERROR, run_batch, summarise_runs, write_records


# The angle grid, read off the old figures, and the sweep of the supplement: p in {3, 8}, 40
# problems per cell, a 5000-cycle horizon and 2000 corner-approaching starts per angle
ANGLES = (0.01, 0.018, 0.03, 0.06, 0.1, 0.18, 0.3, 0.6, 1.0)
ANGLE_DIMENSIONS = (3, 8)
FAMILIES = ("wedge", "vertex")
ANGLE_PROBLEMS = 40
ANGLE_HORIZON = 5000
CAPTURE_SAMPLES = 2000

# Chosen here: the geometry of both families, in the units of the coordinates
START_DISTANCE = 1.0
BOX_HALF_WIDTH = 1.0
CORNER_HALF_WIDTH = 1.0

# Chosen here: corner-approaching starts per angle for the fit of the capture exponent, which
# needs no Dykstra run
CAPTURE_SAMPLES_FIT = 200_000

# The accelerated study of the supplement: p = 8, the squared-error criterion and cap of
# performance_timing, and the angles read off the old figure
ACCELERATED_DIMENSION = 8
ACCELERATED_ANGLES = (0.1, 0.03, 0.01, 0.003)
ACCELERATED_TOLERANCE = performance_timing.TOLERANCE
ACCELERATED_CAP = performance_timing.CAP

# Chosen here: canonical problems per angle and family of the accelerated study
ACCELERATED_PROBLEMS = 40

# Chosen here: every seed lies in a block of its own, set by the family, p, the angle's index in
# its grid, and the problem's index
ANGLE_SEED = 8_000_000

# Chosen here: zeros sit at these values on the log axes of the stall-cycle and capture figures
CYCLES_CLIP, PERCENT_CLIP = 0.5, 1e-3

WORKERS = 8

QUICK_ANGLE_INDICES, QUICK_PROBLEMS, QUICK_HORIZON = (6, 8), 3, 300
QUICK_SAMPLES, QUICK_SAMPLES_FIT = 200, 2000
QUICK_ACCELERATED_INDICES, QUICK_ACCELERATED_PROBLEMS, QUICK_CAP = (0, 1), 2, 2000

# A family member: its constraints A x <= b, its corner, the number of corner rows at the top of
# A, and the unit directions off the wedge's plane as columns, none for a vertex
Geometry = namedtuple("Geometry", "A b corner corner_rows free_directions")

STYLES = {("wedge", 3): (BLUE, "-"), ("wedge", 8): (BLUE, "--"),
          ("vertex", 3): (RED, "-"), ("vertex", 8): (RED, "--")}


def seed(study: int, family: str, num_dimensions: int, angle_index: int, index: int = 0) -> int:
    """The seed of one problem or sample set: study 0 is the sweep, 1 the capture, 2 the
    accelerated study."""
    return (ANGLE_SEED + 1_000_000 * study + 100_000 * FAMILIES.index(family)
            + 1000 * num_dimensions + 100 * angle_index + index)


def haar_rotation(rng, num_dimensions: int) -> np.ndarray:
    """A Haar-distributed orthogonal matrix, drawn as in BoundedTestPlanes."""
    basis, upper = np.linalg.qr(rng.standard_normal((num_dimensions, num_dimensions)))
    return basis * np.sign(np.diag(upper))


def wedge(theta: float, num_dimensions: int, rng) -> Geometry:
    """A wedge of angle theta, in rad, with its box, rotated and moved by rng."""
    p = num_dimensions
    normals = np.zeros((2, p))
    normals[:, 0] = np.cos(theta / 2)
    normals[:, 1] = (np.sin(theta / 2), -np.sin(theta / 2))
    box = np.vstack([np.eye(p)[2:], -np.eye(p)[2:]])
    rotation = haar_rotation(rng, p)
    corner = rng.uniform(-CORNER_HALF_WIDTH, CORNER_HALF_WIDTH, size=p)
    A = np.vstack([normals, box]) @ rotation.T
    b = A @ corner + np.concatenate([np.zeros(2), np.full(len(box), BOX_HALF_WIDTH)])
    return Geometry(A, b, corner, 2, rotation[:, 2:])


def vertex(theta: float, num_dimensions: int, rng) -> Geometry:
    """A vertex of p normals pairwise at angle theta, in rad, rotated and moved by rng."""
    p = num_dimensions
    axis = np.full(p, 1.0 / np.sqrt(p))
    spread = np.eye(p) - 1.0 / p
    spread /= np.linalg.norm(spread, axis=1, keepdims=True)
    # a_i . a_j = cos(alpha)^2 - sin(alpha)^2 / (p - 1) = cos(theta), from w_i . w_j = -1 / (p - 1)
    sin_alpha = np.sqrt((1.0 - np.cos(theta)) * (p - 1) / p)
    normals = np.sqrt(1.0 - sin_alpha ** 2) * axis + sin_alpha * spread
    rotation = haar_rotation(rng, p)
    corner = rng.uniform(-CORNER_HALF_WIDTH, CORNER_HALF_WIDTH, size=p)
    A = normals @ rotation.T
    return Geometry(A, A @ corner, corner, p, np.zeros((p, 0)))


BUILDERS = {"wedge": wedge, "vertex": vertex}


def corner_normals(geometry: Geometry) -> np.ndarray:
    """The unit normals of the corner rows."""
    rows = geometry.A[:geometry.corner_rows]
    return rows / np.linalg.norm(rows, axis=1, keepdims=True)


def canonical_start(geometry: Geometry, rng,
                    distance: float = START_DISTANCE) -> tuple[np.ndarray, np.ndarray]:
    """A canonical start at distance from the corner, in the units of the coordinates, and its
    projection."""
    normals = corner_normals(geometry)
    direction = rng.dirichlet(np.ones(len(normals))) @ normals
    free = geometry.free_directions.shape[1]
    reference = geometry.corner + geometry.free_directions @ rng.uniform(
        -BOX_HALF_WIDTH / 2, BOX_HALF_WIDTH / 2, size=free)
    return reference + distance * direction / np.linalg.norm(direction), reference


def approaching_starts(geometry: Geometry, count: int, rng) -> tuple:
    """count corner-approaching starts, whether each is captured, and the projections.

    The projection of a start that is not captured is NaN.
    """
    normals = corner_normals(geometry)
    p = geometry.corner.size
    directions = np.empty((0, p))
    while len(directions) < count:
        drawn = rng.standard_normal((count, p))
        drawn /= np.linalg.norm(drawn, axis=1, keepdims=True)
        directions = np.vstack([directions, drawn[(drawn @ normals.T > 0.0).any(axis=1)]])
    starts = geometry.corner + START_DISTANCE * directions[:count]
    # a start is in the normal cone when its offset from the corner, within the span of the
    # corner normals, has positive weights on all of them. The box stays inactive since
    # START_DISTANCE <= BOX_HALF_WIDTH
    weights = np.linalg.lstsq(normals.T, (starts - geometry.corner).T, rcond=None)[0].T
    captured = (weights > 0.0).all(axis=1)
    references = np.where(captured[:, None], starts - weights @ normals, np.nan)
    return starts, captured, references


def censored_median(values: np.ndarray, horizon: int) -> tuple[float, bool]:
    """The median of cycle counts in which -1 marks a run censored at horizon, and whether the
    median is censored, i.e. the middle order statistics include a censored run."""
    ordered = np.sort(np.where(values < 0, np.inf, values))
    middle = ordered[(len(ordered) - 1) // 2: len(ordered) // 2 + 1]
    if np.isinf(middle).any():
        return float(horizon), True
    return float(middle.mean()), False


def contraction(errors: np.ndarray, converged_at: int) -> float:
    """The median ratio of consecutive squared errors from cycle 1 to convergence or the
    horizon, the measured contraction rate of the slow phase."""
    end = converged_at if converged_at > 0 else len(errors) - 1
    window = errors[1:end + 1]
    window = window[window > 0.0]
    return float(np.median(window[1:] / window[:-1])) if len(window) > 2 else np.nan


def stack(geometries: list[Geometry]) -> tuple[np.ndarray, np.ndarray]:
    """The constraints of a batch of geometries of one shape."""
    return np.array([g.A for g in geometries]), np.array([g.b for g in geometries])


def sweep_cell(family: str, num_dimensions: int, angle_index: int, problems: int,
               horizon: int) -> list[dict]:
    """Run the canonical problems of one cell of the angle sweep."""
    quiet_accelerate_warnings()
    theta = ANGLES[angle_index]
    geometries, starts, references, seeds = [], [], [], []
    for index in range(problems):
        seeds.append(seed(0, family, num_dimensions, angle_index, index))
        rng = np.random.RandomState(seeds[-1])
        geometries.append(BUILDERS[family](theta, num_dimensions, rng))
        start, reference = canonical_start(geometries[-1], rng)
        starts.append(start)
        references.append(reference)
    A, b = stack(geometries)
    began = time.perf_counter()
    batch = run_batch(np.array(starts), A, b, np.array(references), horizon)
    seconds = time.perf_counter() - began
    records = summarise_runs(batch, num_dimensions)
    quadprog = ConvexProjectionSolver._find_optimal_solution(starts[0], A[0], b[0])
    for index, record in enumerate(records):
        record.update({"family": family, "p": num_dimensions, "theta": theta, "index": index,
                       "seed": seeds[index], "seconds_per_problem": seconds / problems,
                       "contraction": contraction(batch["errors"][index], record["converged_at"])})
    records[0]["reference_difference"] = float(np.abs(quadprog - references[0]).max())
    return records


def capture_cell(family: str, num_dimensions: int, angle_index: int, samples: int,
                 samples_fit: int, horizon: int) -> dict:
    """Sample corner-approaching starts at one angle and run Dykstra from the captured ones."""
    quiet_accelerate_warnings()
    theta = ANGLES[angle_index]
    rng = np.random.RandomState(seed(1, family, num_dimensions, angle_index))
    geometry = BUILDERS[family](theta, num_dimensions, rng)
    starts, captured, references = approaching_starts(geometry, samples, rng)
    record = {"family": family, "p": num_dimensions, "theta": theta, "samples": samples,
              "captures": int(captured.sum())}
    if captured.any():
        chosen = np.flatnonzero(captured)
        count = len(chosen)
        batch = run_batch(starts[chosen], np.repeat(geometry.A[None], count, axis=0),
                          np.repeat(geometry.b[None], count, axis=0), references[chosen], horizon)
        runs = summarise_runs(batch, num_dimensions)
        for field in ("approximate_cycles_decimals", "approximate_cycles_relative",
                      "stalling_cycles_relative", "stalling_cycles_absolute"):
            record[f"with_{field}"] = sum(1 for r in runs if r[field])
        record["converged"] = sum(1 for r in runs if r["converged_at"] >= 0)
    _, captured_fit, _ = approaching_starts(geometry, samples_fit, rng)
    record.update({"samples_fit": samples_fit, "captures_fit": int(captured_fit.sum())})
    if family == "wedge":
        # the direction within the wedge's plane is uniform, the outside of the corner spans an
        # angle pi + theta of it, and the normal cone theta
        record["capture_exact"] = theta / (np.pi + theta)
    return record


def c_cycles_to_tolerance(function, z: np.ndarray, A: np.ndarray, b: np.ndarray,
                          reference: np.ndarray, cap: int) -> int | None:
    """First budget whose C result is within ACCELERATED_TOLERANCE, or None within cap.

    The budget doubles from one cycle and is then bisected, the protocol of the supplement's
    benchmark section, which takes the error as monotone between the last two budgets.
    """
    def within(budget: int) -> bool:
        point = function(z, A, b, budget).projection
        return float(((point - reference) ** 2).sum()) <= ACCELERATED_TOLERANCE

    low, high = 0, 1
    while not within(high):
        if high >= cap:
            return None
        low, high = high, min(2 * high, cap)
    while high - low > 1:
        middle = (low + high) // 2
        low, high = (low, middle) if within(middle) else (middle, high)
    return high


def accelerated_problem(family: str, angle_index: int, index: int, cap: int, use_c: bool,
                        distance: float = START_DISTANCE) -> dict:
    """Cycles to ACCELERATED_TOLERANCE of Dykstra and LTIVer5Solver on one canonical problem
    whose start lies at distance from the corner."""
    quiet_accelerate_warnings()
    theta, p = ACCELERATED_ANGLES[angle_index], ACCELERATED_DIMENSION
    rng = np.random.RandomState(seed(2, family, p, angle_index, index))
    geometry = BUILDERS[family](theta, p, rng)
    z, _ = canonical_start(geometry, rng, distance)
    record = {"family": family, "theta": theta, "index": index, "distance": distance}
    for name, solver in (("dykstra", DykstraProjectionSolver), ("accelerated", LTIVer5Solver)):
        began = time.perf_counter()
        cycles = performance_timing.cycles_to_tolerance(solver, z, geometry.A, geometry.b, p, cap)
        record[f"{name}_seconds"] = time.perf_counter() - began
        record[f"{name}_cycles"] = cap if cycles is None else cycles
        record[f"{name}_censored"] = int(cycles is None)
    settled = LTIVer5Solver(z, geometry.A, geometry.b, max(record["accelerated_cycles"], 1)).solve()
    record["accelerated_settled_at"] = settled.settled_at if settled.is_settled() else -1
    record["accelerated_certificate"] = settled.certificate or ""
    if use_c:
        reference = ConvexProjectionSolver._find_optimal_solution(z, geometry.A, geometry.b)
        for name, function in (("dykstra", c_solvers.dykstra_c_projection),
                               ("accelerated", c_solvers.accelerated_c_projection)):
            cycles = c_cycles_to_tolerance(function, z, geometry.A, geometry.b, reference, cap)
            record[f"{name}_c_cycles"] = cap if cycles is None else cycles
    return record


def parallel(function, arguments: list[tuple], workers: int) -> list:
    """function applied to every tuple of arguments, in processes when workers > 1."""
    if workers <= 1:
        return [function(*a) for a in arguments]
    with ProcessPoolExecutor(max_workers=workers) as pool:
        return list(pool.map(function, *zip(*arguments)))


def run_sweep(angle_indices: tuple, problems: int, horizon: int, workers: int) -> list[dict]:
    """Every record of the angle sweep."""
    cells = [(f, p, k, problems, horizon) for f in FAMILIES for p in ANGLE_DIMENSIONS
             for k in angle_indices]
    # small angles run longest in the relative test, so they go first
    cells.sort(key=lambda cell: cell[2])
    return [r for batch in parallel(sweep_cell, cells, workers) for r in batch]


def run_capture(angle_indices: tuple, samples: int, samples_fit: int, horizon: int,
                workers: int) -> list[dict]:
    """Every record of the capture study."""
    cells = [(f, p, k, samples, samples_fit, horizon) for f in FAMILIES for p in ANGLE_DIMENSIONS
             for k in angle_indices]
    # large angles capture most starts, so they go first
    cells.sort(key=lambda cell: -cell[2])
    return parallel(capture_cell, cells, workers)


def run_accelerated(angle_indices: tuple, problems: int, cap: int, workers: int,
                    distance: float = START_DISTANCE) -> list[dict]:
    """Every record of the accelerated study."""
    use_c = c_solvers.libraries_built()
    tasks = [(f, k, i, cap, use_c, distance) for f in FAMILIES for k in angle_indices
             for i in range(problems)]
    return parallel(accelerated_problem, tasks, workers)


def cell_summary(records: list[dict], horizon: int) -> dict:
    """Per (family, p, theta): the censored median of cycles to converge, the mean cycles of each
    detector, and the median measured contraction."""
    summary = {}
    for key in sorted({(r["family"], r["p"], r["theta"]) for r in records}):
        cell = [r for r in records if (r["family"], r["p"], r["theta"]) == key]
        median, censored = censored_median(np.array([r["converged_at"] for r in cell]), horizon)
        summary[key] = {
            "median": median, "censored": censored,
            "converged": sum(r["converged_at"] >= 0 for r in cell), "problems": len(cell),
            "contraction": float(np.nanmedian([r["contraction"] for r in cell])),
            **{field: float(np.mean([r[field] for r in cell]))
               for field in ("approximate_cycles_decimals", "approximate_cycles_relative",
                             "stalling_cycles_relative", "stalling_cycles_absolute")}}
    return summary


def power_of_ten(value: float) -> str:
    """Mathtext for a power of ten, and the plain number otherwise."""
    exponent = np.log10(value)
    return f"$10^{{{exponent:.0f}}}$" if exponent == round(exponent) else f"{value:g}"


def legend_handles(entries: list[tuple]) -> list[Line2D]:
    """Legend entries from (label, colour, line style, marker, filled)."""
    return [Line2D([], [], color=colour, ls=style, marker=marker, ms=5, label=label,
                   markerfacecolor=colour if filled else "white")
            for label, colour, style, marker, filled in entries]


FAMILY_ENTRIES = [("wedge", BLUE, "-", None, True), ("vertex", RED, "-", None, True),
                  ("$p = 3$", INK, "-", None, True), ("$p = 8$", INK, "--", None, True)]


def angle_cycles_figure(summary: dict, horizon: int) -> float:
    """Draw the median cycles to converge against theta and return the constant c of the
    reference line c / theta^2."""
    fig, ax = plt.subplots(figsize=(6.4, 4.4))
    for (family, p), (colour, style) in STYLES.items():
        keys = [k for k in summary if k[:2] == (family, p)]
        if not keys:
            continue
        thetas = np.array([k[2] for k in keys])
        medians = np.array([summary[k]["median"] for k in keys])
        censored = np.array([summary[k]["censored"] for k in keys])
        ax.loglog(thetas, medians, color=colour, ls=style, lw=1.5)
        ax.scatter(thetas[~censored], medians[~censored], color=colour, s=16, zorder=3)
        ax.scatter(thetas[censored], medians[censored], facecolor="white", edgecolor=colour,
                   s=22, zorder=3)
    # the reference line has slope -2 and is fitted to the uncensored wedge p = 3 medians
    fit = [(k[2], v["median"]) for k, v in summary.items() if k[:2] == ("wedge", 3)
           and not v["censored"]]
    constant = float(np.exp(np.mean([np.log(m) + 2 * np.log(t) for t, m in fit]))) if fit else np.nan
    thetas = np.array(sorted({k[2] for k in summary}))
    if fit:
        ax.loglog(thetas, constant / thetas ** 2, color=INK, ls=":", lw=1.2)
    ax.axhline(horizon, color=GREY, lw=0.8)
    ax.text(thetas[-1], horizon * 1.12, "horizon", color=GREY, fontsize=8, ha="right")
    ax.set_xlabel(r"corner angle $\theta$ in rad")
    ax.set_ylabel("median cycles to converge")
    ax.set_title("Slow-phase length against corner angle (canonical starts)", fontsize=11)
    ax.legend(handles=legend_handles(FAMILY_ENTRIES + [
        (r"$\propto 1/\theta^2$", INK, ":", None, True),
        ("median censored at the horizon", INK, "", "o", False)]), fontsize=8, frameon=False,
        loc="lower left")
    ax.text(0.99, 0.02, f"converged: squared error below {power_of_ten(CONVERGED_SQUARED_ERROR)}",
            transform=ax.transAxes, ha="right", fontsize=8, color=GREY)
    style_axes(ax)
    fig.tight_layout()
    save_figure(fig, "stall_angle_cycles")
    return constant


def angle_stallcycles_figure(summary: dict) -> None:
    """Draw the mean approximate-stall and stalling-period cycles per problem against theta."""
    fig, ax = plt.subplots(figsize=(6.4, 4.4))
    for (family, p), (colour, style) in STYLES.items():
        keys = [k for k in summary if k[:2] == (family, p)]
        if not keys:
            continue
        thetas = [k[2] for k in keys]
        approximate = [max(summary[k]["approximate_cycles_decimals"], CYCLES_CLIP) for k in keys]
        stalling = [max(summary[k]["stalling_cycles_relative"], CYCLES_CLIP) for k in keys]
        ax.loglog(thetas, approximate, color=colour, ls=style, lw=1.4, marker="o", ms=4)
        ax.loglog(thetas, stalling, color=colour, ls=style, lw=0.8, marker="x", ms=6, alpha=0.8)
    ax.set_xlabel(r"corner angle $\theta$ in rad")
    ax.set_ylabel("mean cycles per problem")
    ax.set_title("Approximate-stall and stalling-period cycles (canonical starts)", fontsize=11)
    ax.legend(handles=legend_handles(FAMILY_ENTRIES + [
        ("approximate stall (10 decimals)", INK, "", "o", True),
        ("stalling period", INK, "", "x", True)]), fontsize=8, frameon=False, loc="upper right")
    ax.text(0.01, 0.02, f"values clipped below at {CYCLES_CLIP:g} for log display",
            transform=ax.transAxes, fontsize=8, color=GREY)
    style_axes(ax)
    fig.tight_layout()
    save_figure(fig, "stall_angle_stallcycles")


def angle_capture_figure(records: list[dict]) -> None:
    """Draw the capture probability and the approximate-stall rate given capture against theta."""
    fig, ax = plt.subplots(figsize=(6.4, 4.4))
    for (family, p), (colour, style) in STYLES.items():
        cell = sorted((r for r in records if (r["family"], r["p"]) == (family, p)),
                      key=lambda r: r["theta"])
        if not cell:
            continue
        thetas = [r["theta"] for r in cell]
        capture = [100.0 * r["captures"] / r["samples"] for r in cell]
        if any(capture):
            ax.loglog(thetas, np.maximum(capture, PERCENT_CLIP), color=colour, ls=style, lw=1.5,
                      marker="o", ms=4)
        rated = [r for r in cell if r["captures"]]
        ax.scatter([r["theta"] for r in rated],
                   [max(100.0 * r["with_approximate_cycles_decimals"] / r["captures"],
                        PERCENT_CLIP) for r in rated],
                   marker="s", facecolor="white", edgecolor=colour, s=26, zorder=3)
    exact = sorted({r["theta"] for r in records})
    ax.loglog(exact, [100.0 * t / (np.pi + t) for t in exact], color=GREY, ls=":", lw=1.0)
    ax.set_xlabel(r"corner angle $\theta$ in rad")
    ax.set_ylabel("% of corner-approaching starts")
    ax.set_title("Corner capture probability and approximate-stall rate given capture",
                 fontsize=11)
    ax.set_ylim(PERCENT_CLIP / 2, 150)
    ax.legend(handles=legend_handles(FAMILY_ENTRIES + [
        ("capture probability", INK, "-", "o", True),
        (r"wedge, exact $\theta / (\pi + \theta)$", GREY, ":", None, True),
        ("approximate-stall rate\ngiven capture", INK, "", "s", False)]),
        fontsize=8, frameon=False, loc="center left", bbox_to_anchor=(1.01, 0.55))
    ax.text(1.03, 0.04, f"values clipped below\nat {power_of_ten(PERCENT_CLIP)} for log display",
            transform=ax.transAxes, fontsize=8, color=GREY)
    style_axes(ax)
    fig.tight_layout()
    save_figure(fig, "stall_angle_probability")


def accelerated_summary(records: list[dict]) -> dict:
    """Per (family, theta): medians and the number of censored problems of every column."""
    summary = {}
    columns = [c[:-len("_cycles")] for c in records[0] if c.endswith("_cycles")]
    for key in sorted({(r["family"], r["theta"]) for r in records}):
        cell = [r for r in records if (r["family"], r["theta"]) == key]
        entry: dict = {"problems": len(cell)}
        for column in columns:
            entry[column] = float(np.median([r[f"{column}_cycles"] for r in cell]))
        for name in ("dykstra", "accelerated"):
            entry[f"{name}_censored"] = sum(r[f"{name}_censored"] for r in cell)
        entry["certified"] = sum(r["accelerated_certificate"] == "kkt" for r in cell)
        entry["settled_median"] = float(np.median([r["accelerated_settled_at"] for r in cell
                                                   if r["accelerated_settled_at"] >= 0] or [-1]))
        summary[key] = entry
    return summary


def accelerated_figure(summary: dict, cap: int, name: str = "accelerated_stall_study") -> None:
    """Draw the median cycles to ACCELERATED_TOLERANCE of both solvers, one panel per family."""
    fig, axes = plt.subplots(1, 2, figsize=(9.6, 3.8), sharey=True)
    for ax, family in zip(axes, FAMILIES):
        keys = sorted((k for k in summary if k[0] == family), key=lambda k: -k[1])
        thetas = [k[1] for k in keys]
        ax.loglog(thetas, [summary[k]["dykstra"] for k in keys], color=GREY, lw=3.2, alpha=0.75,
                  marker="o", ms=5, label="Dykstra")
        ax.loglog(thetas, [summary[k]["accelerated"] for k in keys], color=RED, lw=1.5,
                  marker="o", ms=4, label="accelerated")
        if keys and "dykstra_c" in summary[keys[0]]:
            ax.scatter(thetas, [summary[k]["dykstra_c"] for k in keys], marker="s", s=40,
                       facecolor="none", edgecolor=INK, zorder=3, label="Dykstra (C)")
            ax.scatter(thetas, [summary[k]["accelerated_c"] for k in keys], marker="D", s=26,
                       facecolor="none", edgecolor=RED, zorder=3, label="accelerated (C)")
        ax.axhline(cap, color=GREY, ls="--", lw=0.8)
        ax.text(thetas[0] if thetas else 1, cap * 1.15, "cycle cap", color=GREY, fontsize=8)
        ax.set_xlim(max(thetas) * 1.5, min(thetas) / 1.5)
        ax.set_ylim(0.7, cap * 2)
        ax.set_xlabel(r"corner angle $\theta$ in rad")
        ax.set_title(f"{family} ($p = {ACCELERATED_DIMENSION}$)", fontsize=11)
        style_axes(ax)
    axes[0].set_ylabel(f"median cycles to squared error $<$ {power_of_ten(ACCELERATED_TOLERANCE)}")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, fontsize=8, frameon=False, loc="lower center", ncol=len(labels))
    fig.tight_layout(rect=(0, 0.07, 1, 1))
    save_figure(fig, name)


def fitted_exponent(thetas, fractions) -> float:
    """Least-squares slope of log fraction against log theta over the nonzero fractions."""
    points = [(np.log(t), np.log(f)) for t, f in zip(thetas, fractions) if f > 0]
    if len(points) < 2:
        return np.nan
    x, y = np.array(points).T
    return float(np.polyfit(x, y, 1)[0])


def report_sweep(records: list[dict], horizon: int) -> dict:
    """Write the sweep's CSV and two figures, and print its quoted numbers."""
    write_records(records, "stall_angle_sweep")
    summary = cell_summary(records, horizon)
    constant = angle_cycles_figure(summary, horizon)
    angle_stallcycles_figure(summary)
    print(f"angle sweep: {records[0]['seconds_per_problem']:.3f} s per problem in the first cell, "
          f"largest difference between a constructed projection and quadprog "
          f"{max(r.get('reference_difference', 0.0) for r in records):.1e}")
    print("family p theta | median cycles to converge (c: censored), converged of problems | "
          "mean cycles: approximate stall 10 decimals, relative; stalling period relative, "
          "absolute | measured contraction rho, (1 - rho) / theta^2")
    for (family, p, theta), entry in summary.items():
        print(f"{family} {p} {theta:g} | {entry['median']:.0f}{' c' if entry['censored'] else ''}, "
              f"{entry['converged']}/{entry['problems']} | "
              f"{entry['approximate_cycles_decimals']:.1f}, {entry['approximate_cycles_relative']:.1f}; "
              f"{entry['stalling_cycles_relative']:.1f}, {entry['stalling_cycles_absolute']:.1f} | "
              f"{entry['contraction']:.6f}, {(1 - entry['contraction']) / theta ** 2:.3f}")
    stalled = {d: sum(r[f"stalling_cycles_{d}"] for r in records) for d in ("relative", "absolute")}
    print(f"stalling-period cycles over all {len(records)} canonical problems: relative detector "
          f"{stalled['relative']}, absolute detector {stalled['absolute']}")
    for family in FAMILIES:
        for p in ANGLE_DIMENSIONS:
            points = [(t, v["median"]) for (f, q, t), v in summary.items()
                      if (f, q) == (family, p) and not v["censored"]]
            slope = fitted_exponent(*zip(*points)) if len(points) > 1 else np.nan
            print(f"{family} p = {p}: log-log slope of the uncensored medians {slope:.2f}")
    print(f"reference line: {constant:.3g} / theta^2")
    return summary


def report_capture(records: list[dict]) -> None:
    """Write the capture study's CSV and figure, and print its quoted numbers."""
    write_records(records, "stall_angle_capture")
    angle_capture_figure(records)
    print("family p theta | captures of samples, capture % | approximate stall (10 decimals, "
          "relative), stalling period (relative, absolute) among captured | captures of the fit "
          "samples | exact wedge %")
    for r in sorted(records, key=lambda r: (r["family"], r["p"], r["theta"])):
        rates = (f"{r['with_approximate_cycles_decimals']}, {r['with_approximate_cycles_relative']}"
                 f"; {r['with_stalling_cycles_relative']}, {r['with_stalling_cycles_absolute']}"
                 if r["captures"] else "none captured")
        exact = f"{100 * r['capture_exact']:.3f}" if "capture_exact" in r else ""
        print(f"{r['family']} {r['p']} {r['theta']:g} | {r['captures']}/{r['samples']}, "
              f"{100.0 * r['captures'] / r['samples']:.2f} | {rates} | "
              f"{r['captures_fit']}/{r['samples_fit']} | {exact}")
    for family in FAMILIES:
        for p in ANGLE_DIMENSIONS:
            cell = sorted((r for r in records if (r["family"], r["p"]) == (family, p)),
                          key=lambda r: r["theta"])
            if cell:
                exponent = fitted_exponent([r["theta"] for r in cell],
                                           [r["captures_fit"] / r["samples_fit"] for r in cell])
                print(f"{family} p = {p}: capture exponent fitted to the {cell[0]['samples_fit']} "
                      f"samples per angle {exponent:.2f} (theory {1 if family == 'wedge' else p - 1})")


def report_accelerated(records: list[dict], cap: int,
                       name: str = "accelerated_stall_study") -> None:
    """Write the accelerated study's CSV and figure under name, and print its quoted numbers."""
    write_records(records, name)
    summary = accelerated_summary(records)
    accelerated_figure(summary, cap, name)
    print(f"accelerated study at p = {ACCELERATED_DIMENSION}, start distance "
          f"{records[0]['distance']:g}, cap {cap}, "
          f"C {'run' if 'dykstra_c_cycles' in records[0] else 'not built, skipped'}")
    print("family theta problems | median cycles to squared error 1e-3: Dykstra, accelerated "
          "(Python), Dykstra, accelerated (C) | censored Dykstra, accelerated | certified, "
          "median settle cycle")
    for (family, theta), e in sorted(summary.items(), key=lambda item: (item[0][0], -item[0][1])):
        c = (f", {e['dykstra_c']:.0f}, {e['accelerated_c']:.0f}" if "dykstra_c" in e else "")
        print(f"{family} {theta:g} {e['problems']} | {e['dykstra']:.0f}, {e['accelerated']:.0f}{c} | "
              f"{e['dykstra_censored']}, {e['accelerated_censored']} | {e['certified']}, "
              f"{e['settled_median']:.0f}")
    if "dykstra_c_cycles" in records[0]:
        agree = {name: sum(r[f"{name}_cycles"] == r[f"{name}_c_cycles"] for r in records)
                 for name in ("dykstra", "accelerated")}
        print(f"C and Python cycle counts agree on {agree['dykstra']} (Dykstra) and "
              f"{agree['accelerated']} (accelerated) of {len(records)} problems")
    seconds = {name: np.mean([r[f"{name}_seconds"] for r in records])
               for name in ("dykstra", "accelerated")}
    print(f"Python time per problem: Dykstra {seconds['dykstra']:.2f} s, accelerated "
          f"{seconds['accelerated']:.2f} s, each including its doubling ramp")


def main() -> None:
    """Parse the command line and run."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--quick", action="store_true",
                        help="small sizes written to a temporary directory")
    parser.add_argument("--only", choices=("all", "sweep", "capture", "accelerated"),
                        default="all")
    parser.add_argument("--workers", type=int, default=WORKERS)
    parser.add_argument("--accelerated-distance", type=float, default=START_DISTANCE,
                        help="start distance of the accelerated study; any other value than "
                             "START_DISTANCE writes accelerated_stall_study_distance<value>")
    args = parser.parse_args()
    quiet_accelerate_warnings()
    began = time.perf_counter()
    if args.quick:
        paper_figures.OUTPUT_DIR = tempfile.mkdtemp(prefix="stall_angles_")
        angles, problems, horizon = QUICK_ANGLE_INDICES, QUICK_PROBLEMS, QUICK_HORIZON
        samples, samples_fit = QUICK_SAMPLES, QUICK_SAMPLES_FIT
        accelerated, accelerated_problems, cap = (QUICK_ACCELERATED_INDICES,
                                                  QUICK_ACCELERATED_PROBLEMS, QUICK_CAP)
    else:
        angles, problems, horizon = tuple(range(len(ANGLES))), ANGLE_PROBLEMS, ANGLE_HORIZON
        samples, samples_fit = CAPTURE_SAMPLES, CAPTURE_SAMPLES_FIT
        accelerated, accelerated_problems, cap = (tuple(range(len(ACCELERATED_ANGLES))),
                                                  ACCELERATED_PROBLEMS, ACCELERATED_CAP)
    written = []
    if args.only in ("all", "sweep"):
        report_sweep(run_sweep(angles, problems, horizon, args.workers), horizon)
        written += ["stall_angle_cycles", "stall_angle_stallcycles"]
        print(f"sweep done after {time.perf_counter() - began:.0f} s")
    if args.only in ("all", "capture"):
        report_capture(run_capture(angles, samples, samples_fit, horizon, args.workers))
        written.append("stall_angle_probability")
        print(f"capture done after {time.perf_counter() - began:.0f} s")
    if args.only in ("all", "accelerated"):
        distance = args.accelerated_distance
        name = ("accelerated_stall_study" if distance == START_DISTANCE
                else f"accelerated_stall_study_distance{distance:g}")
        report_accelerated(run_accelerated(accelerated, accelerated_problems, cap, args.workers,
                                           distance), cap, name)
        written.append(name)
    print(f"Finished in {time.perf_counter() - began:.0f} s: {', '.join(written)} as PNG and PDF, "
          f"and their CSV files, in {paper_figures.OUTPUT_DIR}")


if __name__ == "__main__":
    sys.exit(main())
