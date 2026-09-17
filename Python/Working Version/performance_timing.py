"""Time the solvers to a fixed accuracy on the current machine and plot the scaling."""

import statistics
import time

import numpy as np
import matplotlib.pyplot as plt

from convex_projection_solver import DykstraProjectionSolver
from lti_ver5 import LTIVer5Solver
from paper_figures import (
    INK,
    RED,
    bounded_problem,
    infeasible_start,
    save_figure,
    style_axes,
)


# Problem sizes to time, as (half-spaces, dimensions, seeds); larger sizes are
# slower and need roughly cycles * half-spaces * dimensions * 8 bytes of memory
TIERS = ((12, 4, 5), (24, 8, 5), (48, 16, 5), (72, 24, 3))

TOLERANCE = 1e-3
MAX_CYCLES = 20000

SOLVERS = (("Dykstra", DykstraProjectionSolver, INK, "o"),
           ("accelerated", LTIVer5Solver, RED, "s"))


def cycles_to_tolerance(solver_type: type, z: np.ndarray, A: np.ndarray,
                        b: np.ndarray, dimensions: int) -> int | None:
    """First cycle within the tolerance."""
    budget = 64
    while True:
        solver = solver_type(z, A, b, budget, track_error=True,
                             min_error=TOLERANCE, dimensions=dimensions)
        errors = np.asarray(solver.solve().squared_errors, dtype=float)
        reached = np.where(errors <= TOLERANCE)[0]
        if reached.size:
            return int(reached[0])
        if budget >= MAX_CYCLES:
            return None
        budget = min(2 * budget, MAX_CYCLES)


def median_seconds(solver_type: type, z: np.ndarray, A: np.ndarray, b: np.ndarray,
                   dimensions: int, cycles: int) -> float:
    """Median time of a run."""
    timings = []
    # Repeat short runs so the median is not dominated by clock resolution
    while not timings or (sum(timings) < 0.5 and len(timings) < 5):
        solver = solver_type(z, A, b, cycles, dimensions=dimensions)
        start = time.perf_counter()
        solver.solve()
        timings.append(time.perf_counter() - start)
    return statistics.median(timings)


def time_tier(num_planes: int, num_dimensions: int, seeds: int) -> dict:
    """Medians for one size."""
    measured = {name: {"cycles": [], "seconds": []} for name, _, _, _ in SOLVERS}
    for seed in range(seeds):
        np.random.seed(100000 + 1000 * num_planes + seed)
        A, b = bounded_problem(num_planes, num_dimensions)
        z = infeasible_start(A, b, num_dimensions)
        for name, solver_type, _, _ in SOLVERS:
            # Find the budget the solver needs, then time a fresh run over it
            cycles = cycles_to_tolerance(solver_type, z, A, b, num_dimensions)
            if cycles is None:
                continue
            seconds = median_seconds(solver_type, z, A, b, num_dimensions, cycles)
            measured[name]["cycles"].append(cycles)
            measured[name]["seconds"].append(seconds)
    return {name: {key: statistics.median(values) if values else None
                   for key, values in runs.items()}
            for name, runs in measured.items()}


def performance_scaling_figure(results: dict) -> None:
    """Draw median time against size."""
    fig, ax = plt.subplots(figsize=(6.2, 4.4))
    for name, _, colour, marker in SOLVERS:
        sizes = [n for n, runs in results.items() if runs[name]["seconds"]]
        seconds = [results[n][name]["seconds"] for n in sizes]
        if len(sizes) < 2:
            continue
        # A straight line on log-log axes, so its gradient is the growth order
        gradient = np.polyfit(np.log(sizes), np.log(seconds), 1)[0]
        ax.loglog(sizes, seconds, marker=marker, color=colour, lw=1.6, ms=5,
                  label=rf"{name}  ($\sim n^{{{gradient:.2f}}}$)")

    ax.set_xlabel("number of half-spaces $n$")
    ax.set_ylabel(rf"median time to squared error ${TOLERANCE:g}$ [s]")
    ax.legend(fontsize=9, frameon=False, loc="upper left")
    style_axes(ax)
    fig.tight_layout()
    save_figure(fig, "performance_scaling")


def run() -> None:
    """Time every size and plot."""
    results = {}
    print(f"{'size':>10} | {'Dykstra':>22} | {'accelerated':>22}")
    print("-" * 60)
    for num_planes, num_dimensions, seeds in TIERS:
        runs = time_tier(num_planes, num_dimensions, seeds)
        results[num_planes] = runs

        columns = []
        for name, _, _, _ in SOLVERS:
            cycles, seconds = runs[name]["cycles"], runs[name]["seconds"]
            columns.append(f"{cycles:>6.0f} cyc {seconds * 1000:>8.2f} ms"
                           if cycles is not None else f"{'not reached':>22}")
        print(f"{num_planes:>5} x {num_dimensions:<3} | " + " | ".join(columns), flush=True)

    performance_scaling_figure(results)


if __name__ == "__main__":
    run()
