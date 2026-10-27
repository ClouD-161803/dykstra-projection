"""Measure how closely the C solvers reproduce the Python solvers, the numbers behind the
C-implementation paragraph of the SIAM paper's supplement.

Run from this folder with `python c_validation.py` after building the libraries in the
repository's C folder. On problems 0 to SEEDS - 1 of each size tier in TIERS, drawn by
problem_generators.problem, it prints and writes to results/paper/tab-c-validation.tex:

- the largest infinity-norm distance between the C and Python Dykstra iterates at each
  budget in BUDGETS, and the largest stationarity residual z - x - A^T lambda of the C
  multipliers,
- the same distance for the accelerated solvers, over the runs in which both settle or
  neither does, with the count of runs in which only one settles,
- the largest distance between a certified C point and the quadprog projection over the
  same problems under a budget of CERTIFIED_BUDGET cycles,
- and the same checks on the degenerate problems of edge_cases().
"""

import numpy as np

import c_solvers
from convex_projection_solver import ConvexProjectionSolver, DykstraProjectionSolver
from lti_solver import LTIVer5Solver
from paper_figures import quiet_accelerate_warnings, save_tex
from problem_generators import problem, problem_seed


TIERS = ((12, 4), (20, 6), (28, 8), (36, 10))
SEEDS = 10
BUDGETS = (200, 2000)
CERTIFIED_BUDGET = 10**6


def edge_cases() -> dict:
    """Degenerate problems: a feasible start, n = p + 1, an interval in one dimension,
    duplicated and rescaled rows, a start of magnitude 1e6, and no constraints."""
    A, b, z = problem(12, 4, problem_seed(12, 0))
    A_min, b_min, z_min = problem(5, 4, problem_seed(5, 0))
    return {
        # The projection of an infeasible start is feasible, on the boundary
        "feasible start": (ConvexProjectionSolver._find_optimal_solution(z, A, b), A, b),
        "n = p + 1": (z_min, A_min, b_min),
        "interval": (np.array([3.0]), np.array([[1.0], [-2.0]]), np.array([1.0, 1.0])),
        "duplicated and rescaled rows": (z, np.vstack([A, 3.0 * A[:4]]),
                                         np.hstack([b, 3.0 * b[:4]])),
        "start of magnitude 1e6": (1e6 * z / np.linalg.norm(z), A, b),
        "no constraints": (z, np.zeros((0, 4)), np.zeros(0)),
    }


def stationarity(z: np.ndarray, A: np.ndarray, x: np.ndarray, multipliers: np.ndarray) -> float:
    """The infinity norm of z - x - A^T lambda."""
    return float(np.max(np.abs(z - x - A.T @ multipliers))) if A.size else 0.0


def compare(z: np.ndarray, A: np.ndarray, b: np.ndarray, budget: int) -> dict:
    """C against Python, for both solvers, at one budget."""
    python_dykstra = DykstraProjectionSolver(z, A, b, budget).solve().projection
    c_dykstra = c_solvers.dykstra_c_projection(z, A, b, budget)
    python_accelerated = LTIVer5Solver(z, A, b, budget)
    python_point = python_accelerated.solve().projection
    c_accelerated = c_solvers.accelerated_c_projection(z, A, b, budget)
    return {
        "dykstra distance": float(np.max(np.abs(c_dykstra.projection - python_dykstra))),
        "dykstra stationarity": stationarity(z, A, c_dykstra.projection, c_dykstra.multipliers),
        "same status": python_accelerated.settled == c_accelerated.settled,
        "accelerated distance": float(np.max(np.abs(c_accelerated.projection - python_point))),
        "settled cycles agree": (not c_accelerated.settled
                                 or c_accelerated.cycles + 1 == python_accelerated.settled_at),
    }


def certified_distance(z: np.ndarray, A: np.ndarray, b: np.ndarray) -> tuple:
    """Whether the C accelerated solver settles, its distance from quadprog and its residual."""
    result = c_solvers.accelerated_c_projection(z, A, b, CERTIFIED_BUDGET)
    reference = ConvexProjectionSolver._find_optimal_solution(z, A, b)
    return (result.settled, float(np.max(np.abs(result.projection - reference))),
            stationarity(z, A, result.projection, result.multipliers))


def run() -> dict:
    """Measure, print and write every number."""
    quiet_accelerate_warnings()
    if not c_solvers.libraries_built():
        raise SystemExit("Build the C solvers first, with make -C C from the repository root.")
    worst = {"dykstra distance": 0.0, "dykstra stationarity": 0.0, "accelerated distance": 0.0}
    mismatched, cycle_disagreements, compared = 0, 0, 0
    certified, certified_worst, certified_residual = 0, 0.0, 0.0
    instances = 0
    for num_planes, num_dimensions in TIERS:
        for index in range(SEEDS):
            A, b, z = problem(num_planes, num_dimensions, problem_seed(num_planes, index))
            instances += 1
            for budget in BUDGETS:
                result = compare(z, A, b, budget)
                compared += 1
                worst["dykstra distance"] = max(worst["dykstra distance"],
                                                result["dykstra distance"])
                worst["dykstra stationarity"] = max(worst["dykstra stationarity"],
                                                    result["dykstra stationarity"])
                cycle_disagreements += not result["settled cycles agree"]
                if result["same status"]:
                    worst["accelerated distance"] = max(worst["accelerated distance"],
                                                        result["accelerated distance"])
                else:
                    mismatched += 1
            settled, distance, residual = certified_distance(z, A, b)
            if settled:
                certified += 1
                certified_worst = max(certified_worst, distance)
                certified_residual = max(certified_residual, residual)

    edge_worst = {"dykstra distance": 0.0, "accelerated distance": 0.0}
    edge_mismatched = 0
    for name, (z, A, b) in edge_cases().items():
        for budget in BUDGETS:
            result = compare(z, A, b, budget)
            edge_worst["dykstra distance"] = max(edge_worst["dykstra distance"],
                                                 result["dykstra distance"])
            if result["same status"]:
                edge_worst["accelerated distance"] = max(edge_worst["accelerated distance"],
                                                         result["accelerated distance"])
            else:
                edge_mismatched += 1

    sizes = ", ".join(f"{n} x {p}" for n, p in TIERS)
    lines = [
        f"instances: {instances} ({SEEDS} per tier, tiers {sizes}, seeds "
        f"problem_seed(n, 0) to problem_seed(n, {SEEDS - 1})), budgets {BUDGETS}",
        f"Dykstra, C against Python: largest distance {worst['dykstra distance']:.2g}",
        f"Dykstra, C multipliers: largest stationarity residual "
        f"{worst['dykstra stationarity']:.2g}",
        f"accelerated, C against Python: largest distance {worst['accelerated distance']:.2g} "
        f"over {compared - mismatched} of {compared} runs, {mismatched} with only one settled, "
        f"{cycle_disagreements} whose settled cycles disagree",
        f"accelerated, C certified under {CERTIFIED_BUDGET} cycles: {certified} of {instances}, "
        f"largest distance from quadprog {certified_worst:.2g}, largest stationarity residual "
        f"{certified_residual:.2g}",
        f"edge cases ({', '.join(edge_cases())}): largest Dykstra distance "
        f"{edge_worst['dykstra distance']:.2g}, largest accelerated distance "
        f"{edge_worst['accelerated distance']:.2g}, {edge_mismatched} runs with only one "
        "settled",
    ]
    for line in lines:
        print(line)
    tex_sizes = sizes.replace(" x ", r" $\times$ ")
    rows = [
        ("instances, tiers", f"{instances}, {tex_sizes}"),
        ("Dykstra: largest $\\|x_\\mathrm{C} - x_\\mathrm{py}\\|_\\infty$",
         f"{worst['dykstra distance']:.2g}"),
        ("Dykstra: largest $\\|z - x - A^\\top \\lambda\\|_\\infty$",
         f"{worst['dykstra stationarity']:.2g}"),
        ("accelerated: largest $\\|x_\\mathrm{C} - x_\\mathrm{py}\\|_\\infty$",
         f"{worst['accelerated distance']:.2g} ({compared - mismatched} of {compared} runs)"),
        ("accelerated, certified: count, largest distance from quadprog",
         f"{certified} of {instances}, {certified_worst:.2g}"),
        ("edge cases: largest Dykstra and accelerated distances",
         f"{edge_worst['dykstra distance']:.2g}, {edge_worst['accelerated distance']:.2g}"),
    ]
    body = "\n".join(f"  {name} & {value} \\\\" for name, value in rows)
    save_tex("% " + "\n% ".join(lines) + "\n\\begin{tabular}{l l}\n  \\hline\n" + body
             + "\n  \\hline\n\\end{tabular}\n", "tab-c-validation", "python c_validation.py")
    return {"worst": worst, "mismatched": mismatched, "certified": certified,
            "certified_worst": certified_worst, "edge_worst": edge_worst,
            "cycle_disagreements": cycle_disagreements}


if __name__ == "__main__":
    run()
