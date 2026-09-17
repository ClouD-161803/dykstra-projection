"""Rebuild the figures of the LTI acceleration write-up from this repository."""

import os

import numpy as np
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from matplotlib.figure import Figure

from convex_projection_solver import DykstraProjectionSolver
from lti_ver3 import rigorous_deactivation_horizon
from lti_ver5 import LTIVer5Solver


OUTPUT_DIR = "results/paper"

RED, GREEN, BLUE, TEAL = "#d62728", "#2ca02c", "#1f77b4", "#2a9d8f"
INK, GREY, GRID = "#52514e", "#666666", "#e1e0d9"


def save_figure(fig: Figure, name: str) -> None:
    """Write a figure as PNG and PDF."""
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    path = os.path.join(OUTPUT_DIR, name)
    fig.savefig(path + ".png", dpi=130, bbox_inches="tight")
    fig.savefig(path + ".pdf", bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {path}.png and {path}.pdf")


def style_axes(ax: Axes) -> None:
    """Apply the shared axis styling."""
    ax.grid(True, color=GRID, linewidth=0.6, which="both")
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)


def three_plane_arrangement() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Build the three-plane arrangement."""
    # Boundary 1 is the vertical line x = 1; boundaries 2 and 3 pass through
    # v_12 and v_13 on it with slopes -0.16 and 0.585
    v_12, v_13 = np.array([1.0, 0.85]), np.array([1.0, 0.35])
    slope_2, slope_3 = -0.16, 0.585

    # Outward unit normals a_k and offsets b_k
    a_1 = np.array([1.0, 0.0])
    a_2 = np.array([-slope_2, 1.0]) / np.hypot(slope_2, 1.0)
    a_3 = np.array([slope_3, -1.0]) / np.hypot(slope_3, 1.0)
    A = np.stack([a_1, a_2, a_3])
    b = np.array([1.0, a_2 @ v_12, a_3 @ v_13])

    # Unique dependence w_1 a_1 + w_2 a_2 + w_3 a_3 = 0, normalised by w_3 = 1
    w = np.append(np.linalg.solve(A[:2].T, -a_3), 1.0)

    # Vertex floors Gamma_k and the per-cycle loop residuals r_k they induce
    floors = -(w @ b) / w
    residuals = 2.0 * w ** 2 * floors / (w @ w)
    return A, b, floors, residuals


def corner(A: np.ndarray, b: np.ndarray, i: int, j: int) -> np.ndarray:
    """Intersection of two boundaries."""
    return np.linalg.solve(A[[i, j]], b[[i, j]])


def phase_scalars(A: np.ndarray, b: np.ndarray, x: np.ndarray,
                  scalars: np.ndarray, cycles: int) -> np.ndarray:
    """Stored scalars over one phase."""
    x, scalars = x.astype(float).copy(), scalars.astype(float).copy()
    history = [scalars.copy()]
    for _ in range(cycles):
        for k in range(len(scalars)):
            # Shift by the stored scalar d_k, then project back if that leaves H_k
            shifted = x + scalars[k] * A[k]
            excess = A[k] @ shifted - b[k]
            if excess > 0.0:
                x, scalars[k] = shifted - excess * A[k], excess
            else:
                x, scalars[k] = shifted, 0.0
        history.append(scalars.copy())
    return np.array(history)


def vertex_gamma_corner_figure() -> None:
    """Draw the vertex floors."""
    A, b, floors, _ = three_plane_arrangement()
    v_12, v_13, v_23 = corner(A, b, 0, 1), corner(A, b, 0, 2), corner(A, b, 1, 2)
    slope_2, slope_3 = -A[1, 0] / A[1, 1], -A[2, 0] / A[2, 1]
    low, high = -1.1, 2.6

    fig, ax = plt.subplots(figsize=(7.0, 5.4))
    span = np.array([low, high])
    ax.axvline(1.0, color=GREY, lw=1.2)
    ax.plot(span, v_12[1] + slope_2 * (span - 1.0), color=GREY, lw=1.2)
    ax.plot(span, v_13[1] + slope_3 * (span - 1.0), color=GREY, lw=1.2)

    # The feasible region, bounded by the three boundaries to the left
    region = np.array([v_12, [low, v_12[1] + slope_2 * (low - 1.0)],
                       [low, v_13[1] + slope_3 * (low - 1.0)], v_13])
    ax.fill(region[:, 0], region[:, 1], color="#cfe0f0", alpha=0.75, zorder=0)
    ax.text(-0.35, -0.15, r"$\mathcal{H}$", fontsize=15, color="#3a5a80")

    for vertex, name, dx, dy in ((v_12, r"$v_{12}$", -0.14, 0.06),
                                 (v_13, r"$v_{13}$", -0.16, -0.10),
                                 (v_23, r"$v_{23}$", 0.05, 0.07)):
        ax.plot(*vertex, "o", color="black", ms=5, zorder=5)
        ax.text(vertex[0] + dx, vertex[1] + dy, name, fontsize=12)

    for base, normal, name in ((np.array([1.0, -0.42]), A[0], r"$a_1$"),
                               (np.array([-0.55, v_12[1] + slope_2 * -1.55]), A[1], r"$a_2$"),
                               (np.array([2.05, v_13[1] + slope_3 * 1.05]), A[2], r"$a_3$")):
        tip = base + 0.28 * normal
        ax.annotate("", xy=tip, xytext=base,
                    arrowprops=dict(arrowstyle="->", color="#444444", lw=1.3))
        ax.text(*(tip + 0.07 * normal + 0.02), name, fontsize=12)

    ax.text(1.03, -0.9, r"$\partial\mathcal{H}_1$", fontsize=11, color="#444444")
    ax.text(-1.05, 1.27, r"$\partial\mathcal{H}_2$", fontsize=11, color="#444444")
    ax.text(2.25, v_13[1] + slope_3 * 1.25 + 0.16, r"$\partial\mathcal{H}_3$",
            fontsize=11, color="#444444")

    # Gamma_1 > 0: the opposite vertex v_23 lies outside boundary 1
    ax.annotate("", xy=v_23, xytext=(1.0, v_23[1]),
                arrowprops=dict(arrowstyle="-|>", color=RED, lw=1.6, linestyle="--"))
    ax.text(1.22, v_23[1] - 0.52, r"$\Gamma_1 = a_1^\top v_{23} - b_1 > 0$",
            color=RED, fontsize=12)

    # Gamma_3 < 0: v_12 lies inside boundary 3, at the foot of its projection
    foot = v_12 - (A[2] @ v_12 - b[2]) * A[2]
    ax.annotate("", xy=foot, xytext=v_12,
                arrowprops=dict(arrowstyle="-|>", color=BLUE, lw=1.6, linestyle="--"))
    ax.text(1.02, 1.55, r"$\Gamma_3 = a_3^\top v_{12} - b_3 < 0$",
            color=BLUE, fontsize=12)

    ax.set_xlim(low, high)
    ax.set_ylim(-1.0, 1.75)
    ax.set_xlabel("X coordinate", fontsize=12)
    ax.set_ylabel("Y coordinate", fontsize=12)
    fig.tight_layout()
    save_figure(fig, "vertex_gamma_corner")
    print(f"  floors = {np.round(floors, 3).tolist()}")


def triple_loop_drift_figure() -> None:
    """Draw the stored scalars."""
    A, b, _, residuals = three_plane_arrangement()
    entry_scalars, cycles = np.array([4.0, 6.0, 5.0]), 20
    history = phase_scalars(A, b, np.array([0.9, 0.6]), entry_scalars, cycles)

    # The phase ends when the draining scalar of half-space 3 reaches zero
    shed = int(np.argmax(history[:, 2] <= 1e-12))
    t = np.arange(history.shape[0])
    drift = np.arange(0, shed + 3)

    fig, ax = plt.subplots(figsize=(7.6, 4.8))
    for k, colour in enumerate((RED, GREEN, BLUE)):
        ax.plot(t, history[:, k], "-o", color=colour, ms=4, lw=1.6,
                label=rf"$d_{k + 1}^t$")
        ax.plot(drift, history[1, k] + (drift - 1) * residuals[k], "--",
                color=colour, lw=1.0, alpha=0.7)
    ax.axhline(0.0, color="black", lw=0.8)
    ax.axvline(shed, color=GREY, lw=1.0, ls=":")
    ax.text(shed + 0.35, 2.0, "half-space 3 sheds", rotation=90, color=GREY,
            fontsize=11, va="bottom")

    for k, colour, position in ((0, RED, (8.6, 7.7)), (1, GREEN, (9.0, 3.4)),
                                (2, BLUE, (2.8, 1.7))):
        ax.text(*position, rf"slope $r_{k + 1} = {residuals[k]:+.2f}$",
                color=colour, fontsize=12)

    ax.set_xlim(0, cycles)
    ax.set_ylim(-0.5, 10.5)
    ax.set_xlabel(r"cycle $t$ of the phase", fontsize=12)
    ax.set_ylabel(r"$d_k^t$", fontsize=13)
    ax.legend(loc="upper left", fontsize=11)
    fig.tight_layout()
    save_figure(fig, "triple_loop_drift")
    print(f"  residuals = {np.round(residuals, 3).tolist()}, shed at cycle {shed}")


def envelope_bounds_figure() -> None:
    """Draw the two envelope families."""
    # One representative episode: a contracting map, a draining active
    # half-space and an inactive one whose slack sits below a negative floor
    rho = 0.85
    level, drift, amplitude = 1.0, -0.06, 0.9
    floor, transient = -0.5, 0.85

    fig, (left, right) = plt.subplots(1, 2, figsize=(11, 4.2))

    # Active half-space: the drift line inside a decaying envelope, and the
    # bracket between the crossings of the lower and upper bounds
    k = np.linspace(0, 21, 600)
    line, envelope = level + drift * k, amplitude * rho ** k
    y = line - envelope * np.cos(0.7 * k + 0.4)
    lower_crossing = rigorous_deactivation_horizon(level, drift, amplitude, rho)
    upper_crossing = rigorous_deactivation_horizon(level, drift, -amplitude, rho)
    switch = int(np.argmax(y <= 0))
    y[switch:] = np.nan

    left.axhline(0, color=GREY, lw=0.8)
    left.fill_between(k, line - envelope, line + envelope, color=BLUE, alpha=0.12, lw=0)
    left.plot(k, line + envelope, color=BLUE, lw=2)
    left.plot(k, line - envelope, color=BLUE, lw=2)
    left.plot(k, line, color=INK, lw=1.4, ls="--")
    left.plot(k, y, color=TEAL, lw=1.6, label=r"$y_m^{t+k}$ (one trajectory)")
    left.plot(k[switch], 0, "o", color=RED, ms=5, zorder=5)
    left.axvspan(lower_crossing, upper_crossing, color=RED, alpha=0.08, lw=0)
    for crossing, name in ((lower_crossing, r"$k_m^\star$"),
                           (upper_crossing, r"$\bar{k}_m$")):
        left.axvline(crossing, color=RED, lw=0.9, ls=":")
        left.text(crossing, -0.62, name, color=RED, ha="center", fontsize=10)
    left.annotate(r"upper bound $U_m(k)$",
                  (2.0, level + 2 * drift + amplitude * rho ** 2), (4.5, 1.55),
                  color=BLUE, fontsize=9,
                  arrowprops=dict(arrowstyle="-", color=BLUE, lw=0.7))
    left.annotate(r"lower bound $L_m(k)$",
                  (3.0, level + 3 * drift - amplitude * rho ** 3), (6.0, 0.12),
                  color=BLUE, fontsize=9,
                  arrowprops=dict(arrowstyle="-", color=BLUE, lw=0.7))
    left.text(9.0, level + 9.0 * drift + 0.07, r"$G_m + \beta_m k$",
              color=INK, fontsize=9, rotation=-8)
    left.legend(fontsize=8, frameon=False, loc="upper right")
    left.set_xlabel("cycles ahead $k$")
    left.set_ylabel(r"$y_m^{t+k}$")
    left.set_title(r"Active, deactivating ($\beta_m < 0$): "
                   r"switch confined to $[k_m^\star, \bar{k}_m]$",
                   fontsize=10, fontweight="bold")
    left.set_xlim(0, 21)
    left.set_ylim(-0.75, 2.05)
    style_axes(left)

    # Inactive half-space: the slack decays onto its floor, and only the window
    # before the upper bound drops below zero admits a reactivation
    k = np.linspace(0, 14, 500)
    envelope = transient * rho ** k
    slack = floor + envelope * np.cos(0.9 * k + 2.6)
    reactivation = np.log(-floor / transient) / np.log(rho)

    right.axhline(0, color=GREY, lw=0.8)
    right.fill_between(k, floor - envelope, floor + envelope, color=BLUE, alpha=0.12, lw=0)
    right.plot(k, floor + envelope, color=BLUE, lw=2)
    right.plot(k, floor - envelope, color=BLUE, lw=2)
    right.axhline(floor, color=INK, lw=1.4, ls="--")
    right.plot(k, slack, color=TEAL, lw=1.6, label=r"$g_j^{t+k}$ (one slack)")
    right.axvspan(0, reactivation, color=RED, alpha=0.08, lw=0)
    right.axvline(reactivation, color=RED, lw=0.9, ls=":")
    right.text(reactivation, -1.52, r"$k_j^\star$", color=RED, ha="center", fontsize=10)
    right.text(reactivation / 2, 0.28, "reactivation\npossible", color=RED,
               ha="center", fontsize=8)
    right.annotate(r"upper bound $U_j(k)$", (1.5, floor + transient * rho ** 1.5),
                   (4.0, 0.28), color=BLUE, fontsize=9,
                   arrowprops=dict(arrowstyle="-", color=BLUE, lw=0.7))
    right.annotate(r"lower bound $\Gamma_j - \|z_j^t\|_2\,\rho^k$",
                   (2.5, floor - transient * rho ** 2.5), (5.5, -1.25),
                   color=BLUE, fontsize=9,
                   arrowprops=dict(arrowstyle="-", color=BLUE, lw=0.7))
    right.text(10.5, floor + 0.05, r"$\Gamma_j$", color=INK, fontsize=10)
    right.legend(fontsize=8, frameon=False, loc="lower right")
    right.set_xlabel("cycles ahead $k$")
    right.set_ylabel(r"$g_j^{t+k}$")
    right.set_title(r"Inactive ($\beta = 0$): skip refused while $U_j > 0$",
                    fontsize=10, fontweight="bold")
    right.set_xlim(0, 14)
    right.set_ylim(-1.6, 0.55)
    style_axes(right)

    fig.tight_layout()
    save_figure(fig, "envelope_bounds")
    print(f"  crossings = {lower_crossing:.2f}, {upper_crossing:.2f}, "
          f"{reactivation:.2f}")


def bounded_problem(num_planes: int, num_dimensions: int) -> tuple[np.ndarray, np.ndarray]:
    """Random bounded polyhedron."""
    # A positively spanning set of normals bounds the region in every
    # direction, and a random rotation keeps the orientation generic
    spanning = np.vstack([np.eye(num_dimensions), -np.ones((1, num_dimensions))])
    basis, upper = np.linalg.qr(np.random.standard_normal((num_dimensions, num_dimensions)))
    A = spanning @ (basis * np.sign(np.diag(upper))).T

    # Any further normals only shrink the region, so they may be arbitrary
    extra = num_planes - (num_dimensions + 1)
    if extra > 0:
        angles = np.random.uniform(0.0, np.pi / 2, size=(extra, num_dimensions))
        signs = np.random.choice([-1, 1], size=(extra, num_dimensions))
        A = np.vstack([A, signs * np.tan(angles)])

    # Offsets set from a strictly interior point keep the region non-empty
    interior = np.random.uniform(-2.0, 2.0, size=num_dimensions)
    return A, A @ interior + np.random.uniform(0.2, 1.5, size=num_planes)


def infeasible_start(A: np.ndarray, b: np.ndarray, num_dimensions: int) -> np.ndarray:
    """Draw a point outside the region."""
    scale = 4.0
    for _ in range(200):
        candidate = np.random.standard_normal(num_dimensions) * scale
        if np.any(A @ candidate > b):
            return candidate
        scale *= 1.5
    raise RuntimeError("No infeasible start found.")


def performance_error_figure() -> None:
    """Draw squared error against cycle."""
    num_planes, num_dimensions, max_iter = 96, 32, 6000
    tolerance, floor = 1e-3, 1e-10

    # Fixed seed, then generate the region and the start off the same stream
    np.random.seed(100000 + 1000 * num_planes + 3)
    A, b = bounded_problem(num_planes, num_dimensions)
    z = infeasible_start(A, b, num_dimensions)

    curves = {}
    for name, solver_type in (("Dykstra", DykstraProjectionSolver),
                              ("accelerated", LTIVer5Solver)):
        solver = solver_type(z, A, b, max_iter, track_error=True,
                             min_error=tolerance, dimensions=num_dimensions)
        result = solver.solve()
        curves[name] = (np.asarray(result.squared_errors, dtype=float), solver)

    fig, ax = plt.subplots(figsize=(6.2, 4.4))
    for name, colour, width in (("Dykstra", INK, 2.4), ("accelerated", RED, 1.8)):
        errors, _ = curves[name]
        # Tracked errors are rounded, so settled cycles collapse to zero
        ax.semilogy(np.arange(len(errors)), np.maximum(errors, floor),
                    color=colour, lw=width, label=name)
        reached = np.where(errors <= tolerance)[0]
        print(f"  {name}: cycles to squared error {tolerance:g}: "
              f"{int(reached[0]) if reached.size else 'not reached'}")
    ax.axhline(tolerance, color=GREY, lw=1.0, ls="--", label=r"tolerance $10^{-3}$")

    # The accelerated solver certifies the projection and holds it from there
    errors, solver = curves["accelerated"]
    certified = np.where(errors <= tolerance)[0]
    if solver.settled and certified.size:
        settle = int(certified[0])
        ax.axvline(settle, color=RED, lw=0.9, ls=":")
        ax.text(settle, 2.0 * floor, f"  settle ({settle})", color=RED,
                fontsize=9, ha="left")
        print(f"  accelerated settles at cycle {settle}")
    else:
        print("  accelerated did not certify within the cycle budget")

    ax.set_xlabel("cycle $t$")
    ax.set_ylabel(r"squared error $\|x^t - x^\star\|_2^2$")
    ax.set_xlim(0, max_iter)
    ax.set_ylim(bottom=0.5 * floor)
    ax.legend(fontsize=9, frameon=False, loc="upper right")
    style_axes(ax)
    fig.tight_layout()
    save_figure(fig, "performance_error")


def run() -> None:
    """Rebuild every figure."""
    vertex_gamma_corner_figure()
    triple_loop_drift_figure()
    envelope_bounds_figure()

    # Two long solves on a 96-plane problem, a couple of minutes in total
    performance_error_figure()


if __name__ == "__main__":
    run()
