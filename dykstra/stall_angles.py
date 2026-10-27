"""The angle sweeps and the accelerated stall study of the SIAM paper's supplement, section
Additional stall studies: fig:sm-angle-cycles, fig:sm-angle-stallcycles, fig:sm-angle-capture and
fig:sm-acc-stall.

Run from the repository root with `python -m dykstra.stall_angles`, or with `--quick` for small
sizes written to a temporary directory, or with `--only accelerated` to redo the accelerated
study alone. The figures and CSV files go to results/paper at the repository root, and the
numbers the supplement quotes are printed at the end. The full run also repeats the accelerated
study from SENSITIVITY_DISTANCE and writes every quoted number as tab-angle-numbers.tex and
stall_angle_numbers.csv. Plain Dykstra runs through stall_grid.run_batch, with the stall
detectors and the converged threshold of stall_grid.py.

An approximate stall is the one of def:plateau, a cycle whose squared error repeats the previous
cycle's to within ConvexProjectionSolver.RELATIVE_TOLERANCE times the larger, the rule of
_track_error. The canonical and captured starts do not have one, since a slow phase lowers its
squared error by a share of order theta^2 of itself per cycle. fig:sm-angle-stallcycles draws
the contraction rate 1 - rho of the slow phase, where rho is the median ratio of consecutive
squared errors, beside the rate of the slowest mode of the corner's cycle map. For the wedge
that rate is the cos(theta)^4 of the article's lem:wedge.

The supplement does not define either family. Each is built with its corner at the origin, then
turned by a Haar-distributed rotation Q and moved to a corner v drawn uniformly from
[-CORNER_HALF_WIDTH, CORNER_HALF_WIDTH]^p. The corner rows come first.

- wedge: problem_generators.WedgeTestPlanes, two half-spaces whose unit normals meet at angle
  theta and a box on the remaining p - 2 coordinates, the family the benchmark also runs.
- vertex: defined here, p half-spaces through one point whose unit normals
  cos(alpha) u + sin(alpha) w_i lie around the axis u = (1, ..., 1) / sqrt(p), where the w_i are
  the vertices of a regular simplex orthogonal to u. alpha is set so that every two normals meet
  at angle theta, which is then the smallest angle between them, as for the wedge. The normals
  span R^p, the full-rank branch.

A canonical start is problem_generators.canonical_start's, the corner plus START_DISTANCE times
a unit direction in the corner's normal cone, the normalised combination of the corner normals
with weights uniform on the probability simplex. The projection is then the corner by
construction, and every corner row is active at the limit. A wedge start also moves off the
plane by an offset uniform in half the box, which the projection keeps. The problems of one cell
differ in Q, v, the weights, and the offset.

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
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

from dykstra import c_solvers, paper_figures, performance_timing
from dykstra.paper_figures import (BLUE, GREY, INK, RED, quiet_accelerate_warnings, save_figure,
                                   save_tex, style_axes, tex_number)
from dykstra.convex_projection_solver import ConvexProjectionSolver, DykstraProjectionSolver
from dykstra.lti_solver import LTIVer5Solver
# the geometry shared with the benchmark lives in problem_generators, and the names stay
# importable from here
from dykstra.problem_generators import (BOX_HALF_WIDTH, CORNER_HALF_WIDTH, START_DISTANCE,
                                        Geometry, WedgeTestPlanes, canonical_start,
                                        corner_normals, haar_rotation)
from dykstra.stall_grid import CONVERGED_SQUARED_ERROR, run_batch, summarise_runs, write_records


# The angle grid, read off the old figures, and the sweep of the supplement: p in {3, 8}, 40
# problems per cell, a 5000-cycle horizon and 2000 corner-approaching starts per angle
ANGLES = (0.01, 0.018, 0.03, 0.06, 0.1, 0.18, 0.3, 0.6, 1.0)
ANGLE_DIMENSIONS = (3, 8)
FAMILIES = ("wedge", "vertex")
ANGLE_PROBLEMS = 40
ANGLE_HORIZON = 5000
CAPTURE_SAMPLES = 2000

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

# The accelerated study's sensitivity run, reported beside the canonical START_DISTANCE: from
# this distance Dykstra's slow phase lasts beyond the first cycle at every angle
SENSITIVITY_DISTANCE = 10.0

# Chosen here: the power laws are fitted at the angles up to this one, where the leading term
# 2 theta^2 of the wedge's 1 - cos(theta)^4 lies within 1% of it
FIT_MAX_ANGLE = 0.1

# Chosen here: every seed lies in a block of its own, set by the family, p, the angle's index in
# its grid, and the problem's index
ANGLE_SEED = 8_000_000

# Chosen here: a capture share of zero sits at this value on the log axis of the capture figure
PERCENT_CLIP = 1e-3

WORKERS = 8

QUICK_ANGLE_INDICES, QUICK_PROBLEMS, QUICK_HORIZON = (6, 8), 3, 300
QUICK_SAMPLES, QUICK_SAMPLES_FIT = 200, 2000
QUICK_ACCELERATED_INDICES, QUICK_ACCELERATED_PROBLEMS, QUICK_CAP = (0, 1), 2, 2000

STYLES = {("wedge", 3): (BLUE, "-"), ("wedge", 8): (BLUE, "--"),
          ("vertex", 3): (RED, "-"), ("vertex", 8): (RED, "--")}
# the wedge's two dimensions contract alike, hence the p = 8 marker sits inside the p = 3 one
MARKERS = {3: ("o", 6.5), 8: ("s", 3.5)}


def seed(study: int, family: str, num_dimensions: int, angle_index: int, index: int = 0) -> int:
    """The seed of one problem or sample set: study 0 is the sweep, 1 the capture, 2 the
    accelerated study."""
    return (ANGLE_SEED + 1_000_000 * study + 100_000 * FAMILIES.index(family)
            + 1000 * num_dimensions + 100 * angle_index + index)


def wedge(theta: float, num_dimensions: int, rng) -> Geometry:
    """A wedge of angle theta, in rad, with its box, rotated and moved by rng."""
    return WedgeTestPlanes(theta, num_dimensions).geometry(rng)


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


def cycle_map_contraction(geometry: Geometry) -> float:
    """The squared modulus of the slowest mode of the cycle map with every corner row active,
    restricted to the span of the corner normals, i.e. the ratio of consecutive squared errors
    once that mode dominates. For the wedge it is the cos(theta)^4 of lem:wedge."""
    normals = corner_normals(geometry)
    # each row holds a unit normal in an orthonormal basis of the span. The complement of the
    # span is fixed by every factor, and its unit eigenvalues would hide the wedge's rate
    coordinates = normals @ np.linalg.qr(normals.T)[0]
    identity = np.eye(coordinates.shape[1])
    cycle_map = identity
    for normal in coordinates:
        cycle_map = (identity - np.outer(normal, normal)) @ cycle_map
    return float(np.abs(np.linalg.eigvals(cycle_map)).max() ** 2)


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
                       "contraction": contraction(batch["errors"][index], record["converged_at"]),
                       "cycle_map_contraction": cycle_map_contraction(geometries[index])})
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
    detector, the median measured contraction, and the contraction of the cycle map."""
    summary = {}
    for key in sorted({(r["family"], r["p"], r["theta"]) for r in records}):
        cell = [r for r in records if (r["family"], r["p"], r["theta"]) == key]
        median, censored = censored_median(np.array([r["converged_at"] for r in cell]), horizon)
        summary[key] = {
            "median": median, "censored": censored,
            "converged": sum(r["converged_at"] >= 0 for r in cell), "problems": len(cell),
            "contraction": float(np.nanmedian([r["contraction"] for r in cell])),
            "cycle_map_contraction": float(np.median([r["cycle_map_contraction"] for r in cell])),
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
    ax.text(thetas[-1], horizon * 1.12, "cycle budget", color=GREY, fontsize=8, ha="right")
    ax.set_xlabel(r"corner angle $\theta$ in rad")
    ax.set_ylabel("median cycles to converge")
    ax.set_title("Slow-phase length against corner angle (canonical starts)", fontsize=11)
    ax.legend(handles=legend_handles(FAMILY_ENTRIES + [
        (r"$\propto 1/\theta^2$", INK, ":", None, True),
        ("median censored at the cycle budget", INK, "", "o", False)]), fontsize=8, frameon=False,
        loc="lower left")
    ax.text(0.99, 0.02, f"converged: squared error below {power_of_ten(CONVERGED_SQUARED_ERROR)}",
            transform=ax.transAxes, ha="right", fontsize=8, color=GREY)
    style_axes(ax)
    fig.tight_layout()
    save_figure(fig, "stall_angle_cycles")
    return constant


def power_law(thetas, values, max_angle: float = FIT_MAX_ANGLE) -> tuple[float, float]:
    """The constant C and the exponent k of values = C theta^k, fitted by least squares in
    log-log over the positive values at theta <= max_angle, or NaNs with fewer than two."""
    points = [(np.log(t), np.log(v)) for t, v in zip(thetas, values) if v > 0 and t <= max_angle]
    if len(points) < 2:
        return np.nan, np.nan
    x, y = np.array(points).T
    exponent, intercept = np.polyfit(x, y, 1)
    return float(np.exp(intercept)), float(exponent)


def contraction_fits(summary: dict) -> dict:
    """Per (family, p): the power laws of the measured 1 - rho and of the cycle map's, and the
    constant c of the measured c theta^2, each over the angles up to FIT_MAX_ANGLE."""
    fits = {}
    for family, p in sorted({k[:2] for k in summary}):
        keys = [k for k in summary if k[:2] == (family, p)]
        thetas = np.array([k[2] for k in keys])
        measured = np.array([1.0 - summary[k]["contraction"] for k in keys])
        mapped = np.array([1.0 - summary[k]["cycle_map_contraction"] for k in keys])
        fitted = (thetas <= FIT_MAX_ANGLE) & (measured > 0)
        fits[(family, p)] = {
            "measured": power_law(thetas, measured), "cycle_map": power_law(thetas, mapped),
            "squared": float(np.exp(np.mean(np.log(measured[fitted] / thetas[fitted] ** 2))))
            if fitted.any() else np.nan}
    return fits


def angle_contraction_figure(summary: dict, fits: dict, horizon: int) -> None:
    """Draw the measured contraction 1 - rho of the slow phase against theta, the cycle map's,
    and the power law fitted to the wedge, as stall_angle_stallcycles, the name the supplement
    includes."""
    fig, ax = plt.subplots(figsize=(6.4, 4.4))
    thetas = np.array(sorted({k[2] for k in summary}))
    fine = np.geomspace(thetas[0], thetas[-1], 200)
    ax.loglog(fine, 1.0 - np.cos(fine) ** 4, color=BLUE, lw=5, alpha=0.25)
    for p, style in (3, "-"), (8, "--"):
        keys = [k for k in summary if k[:2] == ("vertex", p)]
        if keys:
            ax.loglog([k[2] for k in keys], [1.0 - summary[k]["cycle_map_contraction"]
                                             for k in keys], color=RED, ls=style, lw=0.9)
    constant, exponent = fits.get(("wedge", 3), {}).get("measured", (np.nan, np.nan))
    if np.isfinite(constant):
        span = np.geomspace(thetas[0], max(t for t in thetas if t <= FIT_MAX_ANGLE), 50)
        ax.loglog(span, constant * span ** exponent, color=INK, ls=":", lw=1.4)
    for (family, p), (colour, _) in STYLES.items():
        keys = [k for k in summary if k[:2] == (family, p)]
        if not keys:
            continue
        marker, size = MARKERS[p]
        for open_marker in (False, True):
            chosen = [k for k in keys if summary[k]["censored"] == open_marker]
            ax.loglog([k[2] for k in chosen], [1.0 - summary[k]["contraction"] for k in chosen],
                      ls="", marker=marker, ms=size, color=colour, zorder=3,
                      markerfacecolor="white" if open_marker else colour)
    handles = [Line2D([], [], ls="", marker=MARKERS[p][0], ms=MARKERS[p][1], color=colour,
                      label=f"{family}, $p = {p}$, measured")
               for (family, p), (colour, _) in STYLES.items()]
    handles += [
        Line2D([], [], ls="", marker="o", ms=5, color=INK, markerfacecolor="white",
               label=f"median run censored at the {horizon}-cycle budget"),
        Line2D([], [], color=BLUE, lw=5, alpha=0.25, label=r"wedge lemma, $1 - \cos^4\theta$"),
        Line2D([], [], color=RED, lw=0.9, label="vertex, cycle map, $p = 3$"),
        Line2D([], [], color=RED, lw=0.9, ls="--", label="vertex, cycle map, $p = 8$")]
    if np.isfinite(constant):
        handles.append(Line2D([], [], color=INK, ls=":", lw=1.4,
                              label=rf"wedge fit, ${constant:.3f}\,\theta^{{{exponent:.3f}}}$"))
    ax.legend(handles=handles, fontsize=8, frameon=False, loc="upper left")
    ax.text(0.99, 0.02, r"$\rho$: median ratio of consecutive squared errors" + "\n"
            + rf"fit over $\theta \leq {FIT_MAX_ANGLE:g}$", transform=ax.transAxes, ha="right",
            fontsize=8, color=GREY)
    ax.set_xlabel(r"corner angle $\theta$ in rad")
    ax.set_ylabel(r"contraction rate $1 - \rho$ per cycle")
    ax.set_title("Contraction rate of the slow phase against corner angle (canonical starts)",
                 fontsize=11)
    style_axes(ax)
    fig.tight_layout()
    save_figure(fig, "stall_angle_stallcycles")


def angle_capture_figure(records: list[dict]) -> None:
    """Draw the capture probability against theta. The approximate-stall rate given capture is
    printed rather than drawn, since def:plateau's rule finds no approximate stall among the
    captured starts of the full run."""
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
    exact = sorted({r["theta"] for r in records})
    ax.loglog(exact, [100.0 * t / (np.pi + t) for t in exact], color=GREY, ls=":", lw=1.0)
    ax.set_xlabel(r"corner angle $\theta$ in rad")
    ax.set_ylabel("% of corner-approaching starts")
    ax.set_title("Corner capture probability against corner angle", fontsize=11)
    ax.set_ylim(PERCENT_CLIP / 2, 150)
    ax.legend(handles=legend_handles(FAMILY_ENTRIES + [
        ("capture probability", INK, "-", "o", True),
        (r"wedge, exact $\theta / (\pi + \theta)$", GREY, ":", None, True)]),
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
        ax.text(thetas[0] if thetas else 1, cap * 1.15, "cycle budget", color=GREY, fontsize=8)
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
    return power_law(thetas, fractions, np.inf)[1]


def median_slopes(summary: dict) -> dict:
    """Per (family, p): the log-log slope of the uncensored medians of cycles to converge."""
    slopes = {}
    for family, p in sorted({k[:2] for k in summary}):
        points = [(k[2], v["median"]) for k, v in summary.items()
                  if k[:2] == (family, p) and not v["censored"]]
        slopes[(family, p)] = fitted_exponent(*zip(*points)) if len(points) > 1 else np.nan
    return slopes


def capture_exponents(records: list[dict]) -> dict:
    """Per (family, p): the capture exponent fitted to the fit samples over every angle and over
    the angles up to FIT_MAX_ANGLE."""
    exponents = {}
    for family, p in sorted({(r["family"], r["p"]) for r in records}):
        cell = [r for r in records if (r["family"], r["p"]) == (family, p)]
        thetas = [r["theta"] for r in cell]
        shares = [r["captures_fit"] / r["samples_fit"] for r in cell]
        exponents[(family, p)] = {"all": fitted_exponent(thetas, shares),
                                  "small": power_law(thetas, shares)[1]}
    return exponents


def report_sweep(records: list[dict], horizon: int) -> dict:
    """Write the sweep's CSV and two figures, and print its quoted numbers."""
    write_records(records, "stall_angle_sweep")
    summary = cell_summary(records, horizon)
    constant = angle_cycles_figure(summary, horizon)
    fits = contraction_fits(summary)
    angle_contraction_figure(summary, fits, horizon)
    print(f"angle sweep: {records[0]['seconds_per_problem']:.3f} s per problem in the first cell, "
          f"largest difference between a constructed projection and quadprog "
          f"{max(r.get('reference_difference', 0.0) for r in records):.1e}")
    print("family p theta | median cycles to converge (c: censored), converged of problems | "
          "mean cycles: approximate stall relative, 10 decimals; stalling period relative, "
          "absolute | measured contraction rho, (1 - rho) / theta^2, cycle map (1 - rho) / theta^2")
    for (family, p, theta), entry in summary.items():
        print(f"{family} {p} {theta:g} | {entry['median']:.0f}{' c' if entry['censored'] else ''}, "
              f"{entry['converged']}/{entry['problems']} | "
              f"{entry['approximate_cycles_relative']:.1f}, "
              f"{entry['approximate_cycles_decimals']:.1f}; "
              f"{entry['stalling_cycles_relative']:.1f}, {entry['stalling_cycles_absolute']:.1f} | "
              f"{entry['contraction']:.6f}, {(1 - entry['contraction']) / theta ** 2:.4f}, "
              f"{(1 - entry['cycle_map_contraction']) / theta ** 2:.4f}")
    stalled = {d: sum(r[f"stalling_cycles_{d}"] for r in records) for d in ("relative", "absolute")}
    approximate = {rule: sum(r[f"approximate_cycles_{rule}"] for r in records)
                   for rule in ("relative", "decimals")}
    print(f"stalling-period cycles over all {len(records)} canonical problems: relative detector "
          f"{stalled['relative']}, absolute detector {stalled['absolute']}")
    print(f"approximate-stall cycles over all {len(records)} canonical problems: relative rule "
          f"with RELATIVE_TOLERANCE = {ConvexProjectionSolver.RELATIVE_TOLERANCE:g} "
          f"{approximate['relative']}, ten-decimal rule {approximate['decimals']}")
    for (family, p), slope in median_slopes(summary).items():
        print(f"{family} p = {p}: log-log slope of the uncensored medians {slope:.2f}")
    print(f"reference line: {constant:.3g} / theta^2")
    for (family, p), fit in fits.items():
        (measured_c, measured_k), (mapped_c, mapped_k) = fit["measured"], fit["cycle_map"]
        print(f"{family} p = {p}: 1 - rho over theta <= {FIT_MAX_ANGLE:g}, measured "
              f"{measured_c:.4f} theta^{measured_k:.4f}, cycle map {mapped_c:.4f} "
              f"theta^{mapped_k:.4f}, measured at exponent 2 {fit['squared']:.4f} theta^2")
    return summary


def report_capture(records: list[dict]) -> None:
    """Write the capture study's CSV and figure, and print its quoted numbers."""
    write_records(records, "stall_angle_capture")
    angle_capture_figure(records)
    print("family p theta | captures of samples, capture % | approximate stall (relative, "
          "10 decimals), stalling period (relative, absolute) among captured | captures of the "
          "fit samples | exact wedge %")
    for r in sorted(records, key=lambda r: (r["family"], r["p"], r["theta"])):
        rates = (f"{r['with_approximate_cycles_relative']}, {r['with_approximate_cycles_decimals']}"
                 f"; {r['with_stalling_cycles_relative']}, {r['with_stalling_cycles_absolute']}"
                 if r["captures"] else "none captured")
        exact = f"{100 * r['capture_exact']:.3f}" if "capture_exact" in r else ""
        print(f"{r['family']} {r['p']} {r['theta']:g} | {r['captures']}/{r['samples']}, "
              f"{100.0 * r['captures'] / r['samples']:.2f} | {rates} | "
              f"{r['captures_fit']}/{r['samples_fit']} | {exact}")
    captured = [r for r in records if r["captures"]]
    print(f"captured starts with an approximate stall under the relative rule: "
          f"{sum(r['with_approximate_cycles_relative'] for r in captured)} of "
          f"{sum(r['captures'] for r in captured)}")
    for (family, p), exponent in capture_exponents(records).items():
        print(f"{family} p = {p}: capture exponent fitted to the {records[0]['samples_fit']} "
              f"samples per angle {exponent['all']:.2f} over every angle, {exponent['small']:.2f} "
              f"over theta <= {FIT_MAX_ANGLE:g} (theory {1 if family == 'wedge' else p - 1})")


def report_accelerated(records: list[dict], cap: int,
                       name: str = "accelerated_stall_study") -> dict:
    """Write the accelerated study's CSV and figure under name, print its quoted numbers, and
    return its summary."""
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
    return summary


def tex_scientific(value: float) -> str:
    """LaTeX for value to three significant digits as a mantissa times a power of ten, or n/a."""
    if not np.isfinite(value):
        return "n/a"
    exponent = int(np.floor(np.log10(abs(value))))
    return f"${value / 10.0 ** exponent:.2f} \\times 10^{{{exponent}}}$"


def tex_power_law(fit: tuple[float, float]) -> str:
    """LaTeX for the power law C theta^k of a (C, k) pair."""
    constant, exponent = fit
    if not np.isfinite(constant):
        return "n/a"
    return f"${constant:.3f}\\,\\theta^{{{exponent:.3f}}}$"


def tex_exponent(value: float) -> str:
    """LaTeX for theta to the power value, or n/a."""
    return f"$\\theta^{{{value:.2f}}}$" if np.isfinite(value) else "n/a"


def write_numbers(summary: dict, capture: list[dict], accelerated: dict,
                  cap: int = ACCELERATED_CAP) -> tuple[str, str]:
    """Write every number the supplement quotes from the angle studies as tab-angle-numbers.tex
    and stall_angle_numbers.csv, and return both paths.

    summary is report_sweep's, capture holds the capture records, and accelerated maps up to two
    start distances to the summaries report_accelerated returns for them, run with cycle cap cap.
    """
    rows = []

    def add(section, family, p, theta, distance, quantity, value):
        rows.append({"section": section, "family": family, "p": p, "theta": theta,
                     "distance": distance, "quantity": quantity, "value": value})

    shares = {(r["family"], r["p"], r["theta"]): r for r in capture}
    fits, slopes = contraction_fits(summary), median_slopes(summary)
    exponents = capture_exponents(capture)
    problems = max(entry["problems"] for entry in summary.values())
    samples = {r["samples"] for r in capture} or {CAPTURE_SAMPLES}
    samples_fit = {r["samples_fit"] for r in capture} or {CAPTURE_SAMPLES_FIT}
    lines = ["\\begin{tabular}{l r l r r l l r r}", "  \\hline",
             f"  family & $p$ & $\\theta$ & median cycles to converge & converged of {problems} & "
             f"$1 - \\rho$, measured & $1 - \\rho$, cycle map & capture \\%, {max(samples)} "
             f"starts & capture \\%, {max(samples_fit)} starts \\\\", "  \\hline"]
    for family, p in sorted({k[:2] for k in summary}, key=lambda k: (FAMILIES.index(k[0]), k[1])):
        for key in sorted(k for k in summary if k[:2] == (family, p)):
            entry, theta = summary[key], key[2]
            share = shares.get(key)
            capture_cells = (f"{100.0 * share['captures'] / share['samples']:.2f} & "
                             f"{100.0 * share['captures_fit'] / share['samples_fit']:.3f}"
                             if share else " & ")
            median = (f"$>{entry['median']:.0f}$" if entry["censored"]
                      else f"{entry['median']:.0f}")
            lines.append(f"  {family} & {p} & {theta:g} & {median} & {entry['converged']} & "
                         f"{tex_scientific(1.0 - entry['contraction'])} & "
                         f"{tex_scientific(1.0 - entry['cycle_map_contraction'])} & "
                         f"{capture_cells} \\\\")
            for quantity, value in (
                    ("median_cycles_to_converge", entry["median"]),
                    ("median_censored", int(entry["censored"])),
                    ("converged", entry["converged"]), ("problems", entry["problems"]),
                    ("contraction_rate", 1.0 - entry["contraction"]),
                    ("cycle_map_contraction_rate", 1.0 - entry["cycle_map_contraction"])):
                add("sweep", family, p, theta, "", quantity, value)
            if share:
                add("capture", family, p, theta, "", "capture_percent",
                    100.0 * share["captures"] / share["samples"])
                add("capture", family, p, theta, "", "capture_percent_fit",
                    100.0 * share["captures_fit"] / share["samples_fit"])
                if "capture_exact" in share:
                    add("capture", family, p, theta, "", "capture_percent_exact",
                        100.0 * share["capture_exact"])
        fit, exponent = fits[(family, p)], exponents.get((family, p), {})
        slope = slopes[(family, p)]
        lines.append(f"  {family} & {p} & fit, every $\\theta$ & "
                     f"{f'slope ${slope:.2f}$' if np.isfinite(slope) else 'n/a'} & & & & & "
                     f"{tex_exponent(exponent.get('all', np.nan))} \\\\")
        lines.append(f"  {family} & {p} & fit, $\\theta \\leq {FIT_MAX_ANGLE:g}$ & & & "
                     f"{tex_power_law(fit['measured'])} & {tex_power_law(fit['cycle_map'])} & & "
                     f"{tex_exponent(exponent.get('small', np.nan))} \\\\")
        for quantity, value in (
                ("median_slope", slopes[(family, p)]),
                ("contraction_constant", fit["measured"][0]),
                ("contraction_exponent", fit["measured"][1]),
                ("contraction_constant_at_exponent_2", fit["squared"]),
                ("cycle_map_constant", fit["cycle_map"][0]),
                ("cycle_map_exponent", fit["cycle_map"][1]),
                ("capture_exponent_every_angle", exponent.get("all", np.nan)),
                ("capture_exponent_small_angles", exponent.get("small", np.nan))):
            add("fit", family, p, "", "", quantity, value)
    distances = sorted(accelerated)[:2]
    listed = ", ".join(f"{d:g}" for d in distances)
    if distances:
        lines += ["  \\hline",
                  "  family & $p$ & $\\theta$ & "
                  + " & ".join(f"{solver}, distance {d:g}" for d in distances
                               for solver in ("Dykstra", "accelerated"))
                  + f" & Dykstra at the cap, distances {listed} & KKT certified, distances "
                  f"{listed} \\\\", "  \\hline"]
        for family in FAMILIES:
            thetas = sorted({k[1] for d in distances for k in accelerated[d] if k[0] == family},
                            reverse=True)
            for theta in thetas:
                cells, capped, certified = [], [], []
                for d in distances:
                    entry = accelerated[d].get((family, theta))
                    if entry is None:
                        cells += ["", ""]
                        continue
                    cells += [f"$\\geq {entry[s]:.0f}$" if entry[s] >= cap
                              else tex_number(entry[s], 1) for s in ("dykstra", "accelerated")]
                    capped.append(str(entry["dykstra_censored"]))
                    certified.append(str(entry["certified"]))
                    for quantity in ("dykstra", "accelerated", "dykstra_c", "accelerated_c"):
                        if quantity in entry:
                            add("accelerated", family, ACCELERATED_DIMENSION, theta, d,
                                f"{quantity}_median", entry[quantity])
                    for quantity in ("dykstra_censored", "accelerated_censored", "certified",
                                     "problems"):
                        add("accelerated", family, ACCELERATED_DIMENSION, theta, d, quantity,
                            entry[quantity])
                cells += [""] * (4 - len(cells))
                lines.append(f"  {family} & {ACCELERATED_DIMENSION} & {theta:g} & "
                             f"{' & '.join(cells)} & {', '.join(capped)} & "
                             f"{', '.join(certified)} \\\\")
    lines += ["  \\hline", "\\end{tabular}", ""]
    sweep_stalls = sum(e["approximate_cycles_relative"] * e["problems"] for e in summary.values())
    captured = [r for r in capture if r["captures"]]
    header = (f"% the angle sweeps and the accelerated study. A median of cycles above the cycle budget "
              f"is censored. rho is the median ratio of consecutive squared errors, and the cycle "
              f"map's is the squared modulus of its slowest mode. Power laws are least-squares "
              f"fits in log-log. The accelerated rows give the median cycles to squared error "
              f"{ACCELERATED_TOLERANCE:g}, the problems where Dykstra reaches the cap, and those "
              f"the KKT test certified, per start distance. Approximate-stall cycles under the "
              f"relative rule (RELATIVE_TOLERANCE = "
              f"{ConvexProjectionSolver.RELATIVE_TOLERANCE:g}): "
              f"{sweep_stalls:.0f} in the sweep's canonical problems, and captured starts with "
              f"one: {sum(r['with_approximate_cycles_relative'] for r in captured)} of "
              f"{sum(r['captures'] for r in captured)}\n")
    tex = save_tex(header + "\n".join(lines), "tab-angle-numbers", "python -m dykstra.stall_angles")
    return tex, write_records(rows, "stall_angle_numbers")


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
    written, tables = [], []
    if args.only in ("all", "sweep"):
        summary = report_sweep(run_sweep(angles, problems, horizon, args.workers), horizon)
        written += ["stall_angle_cycles", "stall_angle_stallcycles"]
        print(f"sweep done after {time.perf_counter() - began:.0f} s")
    if args.only in ("all", "capture"):
        capture = run_capture(angles, samples, samples_fit, horizon, args.workers)
        report_capture(capture)
        written.append("stall_angle_probability")
        print(f"capture done after {time.perf_counter() - began:.0f} s")
    if args.only in ("all", "accelerated"):
        distances = [args.accelerated_distance]
        if args.only == "all" and SENSITIVITY_DISTANCE != args.accelerated_distance:
            distances.append(SENSITIVITY_DISTANCE)
        studies = {}
        for distance in distances:
            name = ("accelerated_stall_study" if distance == START_DISTANCE
                    else f"accelerated_stall_study_distance{distance:g}")
            studies[distance] = report_accelerated(run_accelerated(
                accelerated, accelerated_problems, cap, args.workers, distance), cap, name)
            written.append(name)
            print(f"accelerated study from distance {distance:g} done after "
                  f"{time.perf_counter() - began:.0f} s")
    if args.only == "all":
        write_numbers(summary, capture, studies, cap)
        tables = ["tab-angle-numbers.tex", "stall_angle_numbers.csv"]
    print(f"Finished in {time.perf_counter() - began:.0f} s: {', '.join(written)} as PNG and PDF, "
          f"and their CSV files{', ' + ', '.join(tables) if tables else ''}, in "
          f"{paper_figures.OUTPUT_DIR}")


if __name__ == "__main__":
    sys.exit(main())
