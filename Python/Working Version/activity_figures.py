"""The statistics and the drawings of the nine figures of the activity study in activity_study.py.

Each *_figure function writes its figure to results/paper beside this script, as PNG and PDF,
through paper_figures.save_figure, and keeps the basename the supplement includes. The other
functions compute the numbers the supplement quotes from the recorded runs.
"""

from __future__ import annotations

import numpy as np
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

import paper_figures
from paper_figures import BLUE, GREEN, GREY, GRID, INK, RED, TEAL, save_figure, style_axes
from activity_study import (SINGLE_INDEX, Run, battery_timing, change_counts, change_cycles,
                            draw_problem, exact_entries, sampled, study_seed)
from convex_projection_solver import ConvexProjectionSolver, DykstraProjectionSolver
from lti_numerics import unit_constraints

plt = paper_figures.plt

# One colour and marker per curve size and per half-space of the single run
COLOURS = (RED, BLUE, GREEN, TEAL, INK, GREY)
MARKERS = ("o", "s", "^", "v", "D", "P")
LINE_STYLES = ("-", "--", "-.", ":", (0, (6, 2)), (0, (3, 1, 1, 1, 1, 1)))
# The trend figure clips a zero percentile to this value, since its axis is logarithmic
TREND_CLIP = 0.1


def percentile(values: np.ndarray, q, axis: int | None = None) -> np.ndarray:
    """Percentiles as order statistics, the smallest value with at least q percent at or below,
    since every count is an integer."""
    return np.percentile(values, q, axis=axis, method="inverted_cdf")


def counts_until(run: Run, last_cycle: int, opening: bool = False) -> np.ndarray:
    """change_counts over the cycles up to last_cycle."""
    kept = run.cycles <= last_cycle
    return change_counts(Run(run.seed, run.cycles[kept], run.masks[kept], 0, None, False, None,
                             run.tight), opening)


def single_run_index(runs: list) -> int:
    """The index of the run the single-run figure shows, by the rule stated beside SINGLE_INDEX
    in activity_study.py."""
    for index, run in enumerate(runs):
        if change_counts(run).sum() >= 3 and run.cycles[-1] > 5:
            return index
    return SINGLE_INDEX


def single_run(index: int, cfg: dict) -> dict:
    """Residuals and activity of one 6 x 2 run over its first cycles."""
    n, p = cfg["SINGLE_SIZE"]
    seed = study_seed(n, p, index)
    A, b, z = draw_problem(1, n, p, seed)
    cycles, masks, points = exact_entries(A, b, z, cfg["SINGLE_CYCLES"])
    unit_A, unit_b = unit_constraints(A, b)
    run = Run(seed, cycles, masks, cfg["SINGLE_CYCLES"], None, False, None, np.zeros(n, bool))
    active = np.array([masks[np.searchsorted(cycles, t, side="right") - 1]
                       for t in range(cfg["SINGLE_CYCLES"] + 1)])
    ends = points[:, -1]

    # The core solver walks the same path. Its activity record tests the cycle's end point
    # shifted by each correction, which is not eq:activeset, and the cells it differs in are
    # reported rather than drawn
    core = DykstraProjectionSolver(z, A, b, cfg["SINGLE_CYCLES"],
                                   plot_active_halfspaces=True).solve()
    return {"seed": seed, "index": index, "n": n, "p": p, "z": z,
            "residuals": ends @ unit_A.T - unit_b,
            "active": active, "active_counts": active.sum(axis=1),
            "change_cycles": [int(c) for c in cycles[1:]],
            "changes": int(change_counts(run).sum()),
            "changes_with_cycle_1": int(change_counts(run, opening=True).sum()),
            "core_path_difference": float(np.abs(core.path[:, -1] - ends).max()),
            "core_activity_cells_differing": int((core.active_half_spaces.T != active).sum())}


def sample_cycles(horizon: int) -> np.ndarray:
    """Every cycle up to 100 and about 300 more spread evenly on a log scale up to horizon."""
    spread = np.round(np.geomspace(100, horizon, 300)).astype(int) if horizon > 100 else []
    return np.unique(np.concatenate([np.arange(1, min(100, horizon) + 1), spread]))


def curve_numbers(runs: list, n: int, p: int, horizon: int) -> tuple:
    """Activity curves and the statistics the prose quotes, for one size of the curve battery."""
    cycles = sample_cycles(horizon)
    counts, cumulative = map(np.array, zip(*(sampled(run, cycles) for run in runs)))
    with_cycle_1 = np.array([sampled(run, cycles[-1:], opening=True)[1][0] for run in runs])
    mean = counts.mean(axis=0)
    converged = [run for run in runs if run.converged]
    final = np.array([run.masks[-1] for run in converged]).reshape(-1, n)
    tight = np.array([run.tight for run in converged]).reshape(-1, n)

    def flips(opening: bool) -> tuple:
        up = down = 0
        for run in runs:
            kept = np.ones(len(run.cycles) - 1, bool) if opening else run.cycles[1:] >= 2
            before, after = run.masks[:-1][kept], run.masks[1:][kept]
            up += int((~before & after).sum())
            down += int((before & ~after).sum())
        return up, down

    # Cycles 2 to converged_at of each converged run that repeat the previous cycle's set
    repeated = total = 0
    for run in converged:
        last = run.converged_at
        total += max(last - 1, 0)
        repeated += max(last - 1, 0) - int(((run.cycles >= 2) & (run.cycles <= last)).sum())
    up, down = flips(False)
    up_all, down_all = flips(True)
    numbers = {
        "size": f"{n}x{p}", "problems": len(runs), "converged": len(converged),
        "fixed_point": sum(run.fixed_at is not None for run in runs),
        "mean_active_cycle_1": float(mean[0]), "mean_active_cycle_2": float(mean[1]),
        "peak_cycle": int(cycles[np.argmax(mean)]), "peak_mean_active": float(mean.max()),
        "mean_active_at_horizon": float(mean[-1]),
        "active_band_at_horizon": percentile(counts[:, -1], [5, 95]).tolist(),
        "mean_cumulative_changes_at_horizon": float(cumulative[:, -1].mean()),
        "cumulative_band_at_horizon": percentile(cumulative[:, -1], [5, 95]).tolist(),
        "mean_cumulative_changes_with_cycle_1_at_horizon": float(with_cycle_1.mean()),
        "share_final_above_p": float((final.sum(axis=1) > p).mean()) if len(final) else None,
        "share_final_equals_tight":
            float((final == tight).all(axis=1).mean()) if len(final) else None,
        "activations_after_cycle_1": up, "deactivations_after_cycle_1": down,
        "activations_with_cycle_1": up_all, "deactivations_with_cycle_1": down_all,
        "share_preconvergence_cycles_repeating": repeated / total if total else None,
    }
    curves = {"cycles": cycles, "active_mean": mean,
              "active_band": percentile(counts, [5, 95], axis=0),
              "changes_mean": cumulative.mean(axis=0),
              "changes_band": percentile(cumulative, [5, 95], axis=0)}
    return numbers, curves


def eigenvalue_numbers(batteries: list, sizes: tuple) -> tuple:
    """The spectrum of A_m over every episode of the curve battery."""
    values = np.concatenate([battery["eigenvalues"] for battery in batteries])
    moduli = np.abs(values)
    histogram, _ = np.histogram(moduli, bins=100, range=(0.0, 1.0))
    complex_values = np.abs(values.imag) > 1e-12 * np.maximum(moduli, 1e-300)
    numbers = {
        "sizes": [f"{n}x{p}" for n, p in sizes], "episodes": sum(b["episodes"] for b in batteries),
        "empty_episodes": sum(b["empty_episodes"] for b in batteries),
        "eigenvalues": int(values.size),
        "unit_eigenvalues_left_out": sum(b["unit_eigenvalues"] for b in batteries),
        "complex_eigenvalues": int(complex_values.sum()),
        "largest_modulus": float(moduli.max()) if moduli.size else None,
        "largest_complex_modulus":
            float(moduli[complex_values].max()) if complex_values.any() else None,
        "first_bin": int(histogram[0]), "last_bin": int(histogram[-1]),
        "below_1e-12": int((moduli < 1e-12).sum()),
        "complex_by_size": {f"{n}x{p}": int((np.abs(b["eigenvalues"].imag) > 1e-12 * np.maximum(
            np.abs(b["eigenvalues"]), 1e-300)).sum()) for (n, p), b in zip(sizes, batteries)},
    }
    return numbers, {"values": values, "histogram": histogram}


def longest_gap(run: Run, half_space: int) -> int:
    """The longest run of cycles a half-space stayed inactive before it was active again, from
    cycle 1 on."""
    states = run.masks[:, half_space]
    longest, left = 0, None
    for k in range(1, len(states)):
        if run.cycles[k] < 2:
            continue
        if states[k - 1] and not states[k]:
            left = run.cycles[k]
        elif states[k] and not states[k - 1] and left is not None:
            longest = max(longest, int(run.cycles[k] - left))
    return longest


def longest_wait(run: Run, half_space: int) -> int:
    """The longest run of inactive cycles, from cycle 1 on, that ended in an activation."""
    first = np.searchsorted(run.cycles, 1, side="right") - 1
    state, start, longest = run.masks[first, half_space], 1, 0
    for cycle, mask in zip(run.cycles[first + 1:], run.masks[first + 1:]):
        if mask[half_space] == state:
            continue
        state = mask[half_space]
        if state:
            longest = max(longest, int(cycle - start))
        else:
            start = cycle
    return longest


def example_numbers(runs: list, cfg: dict) -> tuple:
    """The run of the example sweep with the most changes of one half-space, drawn in the plane.

    Ties go to the longer inactive gap before a reactivation, then to the lower index.
    """
    def key(item):
        index, run = item
        counts = change_counts(run)
        return (counts.max(), longest_gap(run, int(np.argmax(counts))), -index)

    index, run = max(enumerate(runs), key=key)
    counts = change_counts(run)
    half_space = int(np.argmax(counts))
    n, p = cfg["EXAMPLE_SIZE"]
    A, b, z = draw_problem(1, n, p, run.seed)
    cycles, masks, points = exact_entries(A, b, z, run.cycles_run)
    states = masks[:, half_space]
    numbers = {
        "seed": run.seed, "index": index, "n": n, "half_space": half_space + 1,
        "changes": int(counts[half_space]),
        "changes_with_cycle_1": int(change_counts(run, opening=True)[half_space]),
        "change_cycles": [int(c) for c in change_cycles(run, half_space)],
        "active_in_cycle_0": bool(states[0]),
        "active_in_cycle_1": bool(states[np.searchsorted(cycles, 1, side="right") - 1]),
        "longest_gap": longest_gap(run, half_space),
        "cycles_run": run.cycles_run, "fixed_point": run.fixed_at is not None,
        "agrees_with_exact_cycle": bool(np.array_equal(cycles, run.cycles)
                                        and np.array_equal(masks, run.masks)),
        "runs_with_a_half_space_changing_4_or_more":
            int(sum(change_counts(r).max() >= 4 for r in runs)),
        "longest_gap_in_sweep": max((longest_gap(r, i), r.seed, int(i) + 1)
                                    for r in runs for i in range(r.masks.shape[1])),
        "longest_wait_in_sweep": max((longest_wait(r, i), r.seed, int(i) + 1)
                                     for r in runs for i in range(r.masks.shape[1])),
        "z": z.tolist(), "x_star": ConvexProjectionSolver._find_optimal_solution(z, A, b).tolist(),
    }
    return numbers, {"A": A, "b": b, "z": z, "points": points, "half_space": half_space,
                     "x_star": np.array(numbers["x_star"]), "changes": numbers["changes"],
                     "change_cycles": numbers["change_cycles"]}


def limit_numbers(battery: dict, n: int, cfg: dict) -> dict:
    """The worst single half-space's change count of each 2-D problem, summarised over problems."""
    worst = np.array([change_counts(run).max() for run in battery["runs"]])
    early = np.array([change_counts(run, opening=True).max() for run in battery["runs"]])
    first = np.array([counts_until(run, cfg["LIMIT_FIRST_HORIZON"]).max()
                      for run in battery["runs"]])
    record = int(np.argmax(worst))
    return {"n": n, "problems": len(worst), "maximum": int(worst.max()),
            "p99": float(percentile(worst, 99)), "median": float(percentile(worst, 50)),
            "log2_n_plus_1": float(np.log2(n) + 1),
            "maximum_with_cycle_1": int(early.max()),
            "p99_with_cycle_1": float(percentile(early, 99)),
            "median_with_cycle_1": float(percentile(early, 50)),
            "maximum_at_first_horizon": int(first.max()),
            "problems_raised_after_first_horizon": int((worst > first).sum()),
            "unconverged": sum(not run.converged for run in battery["runs"]),
            "record_seed": battery["runs"][record].seed,
            **battery_timing(battery)}


def trend_numbers(battery: dict, n: int, p: int) -> dict:
    """Per-half-space and total change counts and the final active-set size, for one size."""
    runs = battery["runs"]
    converged = [run for run in runs if run.converged]
    every = np.concatenate([change_counts(run) for run in runs])
    every_early = np.concatenate([change_counts(run, opening=True) for run in runs])
    totals = np.array([change_counts(run).sum() for run in converged])
    finals = np.array([run.masks[-1].sum() for run in converged])
    return {"n": n, "p": p, "problems": len(runs), "converged": len(converged),
            "fixed_point": sum(run.fixed_at is not None for run in runs),
            "per_half_space_mean": float(every.mean()),
            "per_half_space_p99": float(percentile(every, 99)),
            "per_half_space_max": int(every.max()),
            "share_3_or_more": float((every >= 3).mean()),
            "share_3_or_more_with_cycle_1": float((every_early >= 3).mean()),
            "per_half_space_max_with_cycle_1": int(every_early.max()),
            "per_half_space_mean_with_cycle_1": float(every_early.mean()),
            "per_half_space_p99_with_cycle_1": float(percentile(every_early, 99)),
            "final_mean": float(finals.mean()), "final_band": percentile(finals, [5, 95]).tolist(),
            "share_final_equals_p": float((finals == p).mean()),
            "total_mean": float(totals.mean()), "total_band": percentile(totals, [5, 95]).tolist(),
            "total_mean_with_cycle_1":
                float(np.mean([change_counts(r, opening=True).sum() for r in converged])),
            **battery_timing(battery)}


def power_fit(dimensions: list, means: list) -> float:
    """The least-squares slope of log(mean) against log(p)."""
    return float(np.polyfit(np.log(dimensions), np.log(means), 1)[0])


def single_run_figure(data: dict) -> None:
    """half_space_activity_single: each residual per cycle, filled while active, and the count."""
    residuals, active, counts = data["residuals"], data["active"], data["active_counts"]
    cycles = np.arange(len(counts))
    fig, ax = plt.subplots(figsize=(7.0, 4.2))
    right = ax.twinx()
    right.fill_between(cycles, counts, step="mid", color=GRID, linewidth=0)
    right.step(cycles, counts, where="mid", color=GREY, linewidth=1.0)
    right.set_ylim(0, data["n"] + 0.5)
    right.set_ylabel("number of active half-spaces")
    right.spines["top"].set_visible(False)
    # The residual axis is drawn over the count's area
    ax.set_zorder(right.get_zorder() + 1)
    ax.patch.set_visible(False)
    for j in range(data["n"]):
        colour, marker, mask = COLOURS[j % 6], MARKERS[j % 6], active[:, j]
        ax.plot(cycles, residuals[:, j], color=colour, linewidth=0.8, alpha=0.5)
        ax.scatter(cycles[mask], residuals[mask, j], color=colour, marker=marker, s=24, zorder=3)
        ax.scatter(cycles[~mask], residuals[~mask, j], facecolors="white", edgecolors=colour,
                   marker=marker, s=24, linewidths=1.0, zorder=3)
    ax.axhline(0.0, color=INK, linestyle="--", linewidth=0.8)
    style_axes(ax)
    ax.set_xlim(-0.5, cycles[-1] + 0.5)
    ax.set_xlabel(r"cycle $t$")
    ax.set_ylabel(r"residual $a_j^\top x_{tn} - b_j$, rows scaled to $\|a_j\| = 1$")
    handles = [Line2D([], [], color=INK, marker="o", linestyle="", label=r"active in cycle $t$"),
               Line2D([], [], color=INK, marker="o", markerfacecolor="white", linestyle="",
                      label=r"inactive in cycle $t$"),
               Patch(facecolor=GRID, edgecolor=GREY, label="active count (right axis)")]
    handles += [Line2D([], [], color=COLOURS[j % 6], marker=MARKERS[j % 6], linestyle="",
                       label=f"half-space {j + 1}") for j in range(data["n"])]
    ax.legend(handles=handles, fontsize=8, frameon=False, ncol=3, loc="upper right")
    save_figure(fig, "half_space_activity_single")


def curve_figures(curves: dict) -> None:
    """half_space_activity_study and half_space_activity_changes: means and bands per size."""
    for key, name, label in (
            ("active", "half_space_activity_study", r"active half-spaces $|\mathcal{C}^{(t)}|$"),
            ("changes", "half_space_activity_changes", "cumulative activity changes")):
        fig, ax = plt.subplots(figsize=(6.4, 4.4))
        for ((n, p), curve), colour, style in zip(curves.items(), COLOURS, LINE_STYLES):
            ax.fill_between(curve["cycles"], *curve[f"{key}_band"], color=colour, alpha=0.08,
                            linewidth=0)
            for edge in curve[f"{key}_band"]:
                ax.plot(curve["cycles"], edge, color=colour, linewidth=0.5, alpha=0.6)
            ax.plot(curve["cycles"], curve[f"{key}_mean"], color=colour, linewidth=1.5,
                    linestyle=style, label=rf"${n} \times {p}$")
        ax.set_xscale("log")
        style_axes(ax)
        ax.set_xlabel(r"cycle $t$")
        ax.set_ylabel(label)
        handles, _ = ax.get_legend_handles_labels()
        handles.append(Patch(facecolor=GREY, alpha=0.3, label="5th to 95th percentile"))
        ax.legend(handles=handles, title=r"$n \times p$, mean", fontsize=8, title_fontsize=8,
                  frameon=False, loc="upper left" if key == "changes" else "upper right")
        save_figure(fig, name)


def eigenvalue_figure(spectrum: dict) -> None:
    """am_eigs_study: the moduli of the eigenvalues of A_m, and the spectrum against the circle."""
    values, histogram = spectrum["values"], spectrum["histogram"]
    fig, (left, right) = plt.subplots(1, 2, figsize=(11, 4.4))
    left.bar(np.linspace(0.0, 1.0, len(histogram), endpoint=False), histogram,
             width=1.0 / len(histogram), align="edge", color=BLUE, linewidth=0)
    left.axvline(1.0, color=RED, linestyle="--", linewidth=1.2,
                 label=r"$|\lambda| = 1$, contraction bound")
    left.set_yscale("log")
    left.set_xlim(0.0, 1.05)
    left.set_xlabel(r"modulus $|\lambda|$ of an eigenvalue of $A_m$")
    left.set_ylabel(f"eigenvalues per bin of width {1.0 / len(histogram):g}")
    left.legend(fontsize=8, frameon=False, loc="upper right")
    angles = np.linspace(0.0, 2.0 * np.pi, 400)
    right.plot(np.cos(angles), np.sin(angles), color=RED, linestyle="--", linewidth=1.2,
               label="unit circle")
    right.scatter(values.real, values.imag, s=5, color=BLUE, alpha=0.15, linewidths=0,
                  rasterized=True)
    right.plot([], [], color=BLUE, marker="o", markersize=4, linestyle="",
               label=r"eigenvalues $\lambda$")
    right.set_aspect("equal")
    right.set_xlim(-1.1, 1.1)
    right.set_ylim(-1.1, 1.1)
    right.set_xlabel(r"$\mathrm{Re}\,\lambda$")
    right.set_ylabel(r"$\mathrm{Im}\,\lambda$")
    right.legend(fontsize=8, frameon=False, loc="upper right")
    for ax in (left, right):
        style_axes(ax)
    save_figure(fig, "am_eigs_study")


def feasible_polygon(A: np.ndarray, b: np.ndarray) -> np.ndarray:
    """The vertices of the bounded polygon A x <= b, in counterclockwise order."""
    unit_A, unit_b = unit_constraints(A, b)
    vertices = []
    for i in range(len(b)):
        for j in range(i + 1, len(b)):
            matrix = unit_A[[i, j]]
            if abs(np.linalg.det(matrix)) < 1e-12:
                continue
            vertex = np.linalg.solve(matrix, unit_b[[i, j]])
            if np.all(unit_A @ vertex - unit_b <= 1e-9 * (1.0 + np.abs(vertex).sum())):
                vertices.append(vertex)
    vertices = np.array(vertices)
    centre = vertices.mean(axis=0)
    return vertices[np.argsort(np.arctan2(*(vertices - centre).T[::-1]))]


def _box(points: np.ndarray, margin: float) -> tuple:
    """The lower and upper corners of a square box around points, widened by margin."""
    low, high = points.min(axis=0), points.max(axis=0)
    half = 0.5 * (1.0 + 2.0 * margin) * (high - low).max()
    return 0.5 * (low + high) - half, 0.5 * (low + high) + half


def _draw_plane(ax, example: dict, polygon: np.ndarray, path: np.ndarray, box: tuple) -> None:
    """Every boundary, the intersection, and the iterates of the example, within box."""
    unit_A, unit_b = unit_constraints(example["A"], example["b"])
    low, high = box
    centre, reach = 0.5 * (low + high), 4.0 * (high - low).max()
    for i, (normal, offset) in enumerate(zip(unit_A, unit_b)):
        foot = centre - (normal @ centre - offset) * normal
        along = np.array([-normal[1], normal[0]]) * reach
        line = np.array([foot - along, foot + along]).T
        if i == example["half_space"]:
            ax.plot(*line, color=RED, linewidth=2.2, zorder=4,
                    label=f"half-space {i + 1}: {example['changes']} activity changes,\nin cycles "
                          + ", ".join(str(c) for c in example["change_cycles"]))
        else:
            ax.plot(*line, color=GREY, linewidth=0.8, zorder=2)
    ax.fill(*polygon.T, facecolor=BLUE, alpha=0.3, edgecolor=BLUE, linewidth=1.0, zorder=3,
            label=r"intersection $\mathcal{H}$")
    ax.plot(*path.T, color=INK, linestyle="--", linewidth=0.8, zorder=5,
            label="Dykstra's iterates, every visit")
    ax.plot(*example["points"][1:, -1].T, color=INK, marker=".", markersize=3, linestyle="",
            zorder=5)
    ax.plot(*example["z"], color=INK, marker="s", markersize=6, linestyle="", zorder=6,
            label=r"start $x^0$")
    ax.plot(*example["x_star"], color=GREEN, marker="*", markersize=13, linestyle="", zorder=6,
            markeredgecolor=INK, markeredgewidth=0.5, label=r"projection $x^\star$")
    ax.set_xlim(low[0], high[0])
    ax.set_ylim(low[1], high[1])
    ax.set_aspect("equal")
    style_axes(ax)


def example_figure(example: dict) -> None:
    """multi_toggle_example: every boundary, the intersection, Dykstra's path, and the half-space
    that changes activity most often, in full on the left and enlarged on the right."""
    points = example["points"]
    path = np.vstack([example["z"][None, :], points[1:].reshape(-1, 2)])
    polygon = feasible_polygon(example["A"], example["b"])
    fig, (left, right) = plt.subplots(1, 2, figsize=(11.0, 5.4))
    # The boxes follow the path, since an intersection can reach far beyond it
    x_star = example["x_star"][None, :]
    _draw_plane(left, example, polygon, path, _box(np.vstack([path, x_star]), 0.06))
    low, high = _box(np.vstack([points[2:].reshape(-1, 2), x_star]), 0.1)
    _draw_plane(right, example, polygon, path, (low, high))
    left.indicate_inset([*low, *(high - low)], edgecolor=INK)
    for ax in (left, right):
        ax.set_xlabel(r"$x_1$")
        ax.set_ylabel(r"$x_2$")
    left.legend(fontsize=8, frameon=True, framealpha=0.95, loc="best")
    save_figure(fig, "multi_toggle_example")


def limit_figure(rows: list, cfg: dict) -> None:
    """activity_change_limit_2d: the worst half-space's change count against n, at p = 2."""
    n = np.array([row["n"] for row in rows])
    fig, ax = plt.subplots(figsize=(6.4, 4.4))
    ax.plot(n, [row["maximum"] for row in rows], color=BLUE, marker="o", label="maximum")
    ax.plot(n, [row["p99"] for row in rows], color=BLUE, linestyle="--", label="99th percentile")
    ax.plot(n, [row["median"] for row in rows], color=BLUE, linestyle=":", label="median")
    ax.plot(n, np.log2(n) + 1, color=INK, linestyle="--", linewidth=1.0, label=r"$\log_2 n + 1$")
    ax.set_xscale("log", base=2)
    ax.set_xticks(n, [str(value) for value in n])
    ax.set_ylim(bottom=0)
    style_axes(ax)
    ax.set_xlabel(r"number of half-spaces $n$")
    ax.set_ylabel("changes of the most-changing half-space")
    ax.legend(fontsize=8, frameon=False, loc="upper left")
    ax.text(0.98, 0.03, f"{cfg['LIMIT_PROBLEMS']} problems per $n$ from Generator 2, run to a fixed\n"
            f"point or {cfg['LIMIT_HORIZON']} cycles", transform=ax.transAxes, ha="right",
            va="bottom", fontsize=8, color=GREY)
    save_figure(fig, "activity_change_limit_2d")


def trend_figures(rows: list, fits: dict, cfg: dict) -> None:
    """activity_toggle_tail, activity_trend_final, and activity_trend_changes against p."""
    series = [(ratio, colour, [row for row in rows if row["n"] == ratio * row["p"]])
              for ratio, colour in zip(cfg["TREND_RATIOS"], (BLUE, RED))]

    fig, ax = plt.subplots(figsize=(6.4, 4.4))
    for ratio, colour, chosen in series:
        p = [row["p"] for row in chosen]
        ax.plot(p, [row["per_half_space_max"] for row in chosen], color=colour, marker="o")
        ax.plot(p, [row["per_half_space_p99"] for row in chosen], color=colour, linestyle="--")
        ax.plot(p, [row["per_half_space_mean"] for row in chosen], color=colour, linestyle=":")
    handles = [Line2D([], [], color=colour, label=f"$n = {ratio}p$") for ratio, colour, _ in series]
    handles += [Line2D([], [], color=INK, marker="o", label="maximum"),
                Line2D([], [], color=INK, linestyle="--", label="99th percentile"),
                Line2D([], [], color=INK, linestyle=":", label="mean")]
    ax.legend(handles=handles, fontsize=8, frameon=False, loc="upper left")
    ax.set_ylim(bottom=0)
    ax.set_xticks(cfg["TREND_DIMENSIONS"])
    style_axes(ax)
    ax.set_xlabel(r"dimension $p$")
    ax.set_ylabel("activity changes of one half-space")
    save_figure(fig, "activity_toggle_tail")

    fig, ax = plt.subplots(figsize=(6.4, 4.4))
    dimensions = np.array(cfg["TREND_DIMENSIONS"])
    ax.plot(dimensions, dimensions, color=INK, linestyle="--", linewidth=1.0, label=r"dimension $p$")
    for ratio, colour, chosen in series:
        p = [row["p"] for row in chosen]
        ax.fill_between(p, *np.array([row["final_band"] for row in chosen]).T, color=colour,
                        alpha=0.15, linewidth=0)
        ax.plot(p, [row["final_mean"] for row in chosen], color=colour, marker="o",
                label=f"$n = {ratio}p$, mean")
    ax.set_xticks(dimensions)
    style_axes(ax)
    ax.set_xlabel(r"dimension $p$")
    ax.set_ylabel("final number of active half-spaces")
    ax.legend(fontsize=8, frameon=False, loc="upper left")
    ax.text(0.98, 0.03, "converged runs only, shaded: 5th to 95th percentile",
            transform=ax.transAxes, ha="right", va="bottom", fontsize=8, color=GREY)
    save_figure(fig, "activity_trend_final")

    fig, ax = plt.subplots(figsize=(6.4, 4.4))
    for ratio, colour, chosen in series:
        p = [row["p"] for row in chosen]
        band = np.maximum(np.array([row["total_band"] for row in chosen]).T, TREND_CLIP)
        ax.fill_between(p, *band, color=colour, alpha=0.15, linewidth=0)
        ax.plot(p, [row["total_mean"] for row in chosen], color=colour, marker="o",
                label=rf"$n = {ratio}p$, mean, fitted $\propto p^{{{fits[f'n = {ratio}p']:.2f}}}$")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xticks(dimensions, [str(value) for value in dimensions])
    ax.minorticks_off()
    style_axes(ax)
    ax.set_xlabel(r"dimension $p$")
    ax.set_ylabel("total activity changes until convergence")
    ax.legend(fontsize=8, frameon=False, loc="upper left")
    ax.text(0.98, 0.03, f"converged runs only, shaded: 5th to 95th percentile,\n"
            f"clipped below at {TREND_CLIP:g} for the logarithmic axis", transform=ax.transAxes,
            ha="right", va="bottom", fontsize=8, color=GREY)
    save_figure(fig, "activity_trend_changes")


def print_quoted(numbers: dict) -> None:
    """Print the numbers the supplement's prose quotes, against its claims."""
    single = numbers["single"]
    print(f"Single run: Generator 1 problem {single['n']} x {single['p']}, seed {single['seed']}, "
          f"active counts {[int(c) for c in single['active_counts']]}, changes in cycles "
          f"{single['change_cycles']}", flush=True)
    for row in numbers["curves"]:
        print(f"Curves {row['size']}: mean active {row['mean_active_cycle_1']:.2f} at cycle 1, peak "
              f"{row['peak_mean_active']:.2f} at cycle {row['peak_cycle']}, "
              f"{row['mean_active_at_horizon']:.2f} at the horizon. Mean cumulative changes "
              f"{row['mean_cumulative_changes_at_horizon']:.2f} "
              f"({row['mean_cumulative_changes_with_cycle_1_at_horizon']:.2f} with cycle 1). "
              f"Converged {row['converged']} of {row['problems']}, final set above p in "
              f"{row['share_final_above_p']:.3f}, equal to the tight set in "
              f"{row['share_final_equals_tight']:.3f}. {row['activations_after_cycle_1']} "
              f"activations and {row['deactivations_after_cycle_1']} deactivations. Share of "
              f"preconvergence cycles repeating the previous set "
              f"{row['share_preconvergence_cycles_repeating']:.4f}", flush=True)
    example = numbers["example"]
    gap, wait = example["longest_gap_in_sweep"], example["longest_wait_in_sweep"]
    print(f"Example: seed {example['seed']}, n = {example['n']}, half-space {example['half_space']} "
          f"changes {example['changes']} times ({example['changes_with_cycle_1']} with cycle 1), "
          f"in cycles {example['change_cycles']}, longest inactive gap {example['longest_gap']}. "
          f"Longest gap of the sweep {gap[0]} cycles, seed {gap[1]}, half-space {gap[2]}. "
          f"Longest inactive stretch ending in an activation {wait[0]} cycles, seed {wait[1]}, "
          f"half-space {wait[2]}", flush=True)
    for row in numbers["limit"]:
        print(f"2-D, n = {row['n']}: worst half-space maximum {row['maximum']}, 99th percentile "
              f"{row['p99']:g}, median {row['median']:g} (with cycle 1 {row['maximum_with_cycle_1']}, "
              f"{row['p99_with_cycle_1']:g}, {row['median_with_cycle_1']:g}), "
              f"log2 n + 1 = {row['log2_n_plus_1']:g}", flush=True)
    for row in numbers["trend"]:
        print(f"Trend {row['n']} x {row['p']}: per half-space mean {row['per_half_space_mean']:.3f}, "
              f"99th percentile {row['per_half_space_p99']:g}, maximum {row['per_half_space_max']}, "
              f"share with 3 or more {row['share_3_or_more']:.4f} "
              f"({row['share_3_or_more_with_cycle_1']:.4f} with cycle 1). Converged "
              f"{row['converged']} of {row['problems']}, final size {row['final_mean']:.2f}, total "
              f"changes {row['total_mean']:.2f} ({row['total_mean_with_cycle_1']:.2f} with cycle 1)",
              flush=True)
    for name, label in (("trend_fits", "from cycle 2"), ("trend_fits_with_cycle_1", "with cycle 1")):
        fits = ", ".join(f"{key} as p^{value:.2f}" for key, value in numbers[name].items())
        print(f"Total changes {label} grow with {fits}, against p^1.5 in the prose and p^1.4 for "
              f"n = 2p in the old figure", flush=True)
    eigen = numbers["eigenvalues"]
    print(f"Eigenvalues: {eigen['eigenvalues']} from {eigen['episodes']} episodes, "
          f"{eigen['unit_eigenvalues_left_out']} unit eigenvalues left out, "
          f"{eigen['complex_eigenvalues']} complex, largest modulus {eigen['largest_modulus']}",
          flush=True)
