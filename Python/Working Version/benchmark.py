"""Benchmark the projection methods against general-purpose QP solvers, the protocol behind
the dense columns of the SIAM paper's tab:benchmark and the supplement's tab:sm-bench-tiers
and tab:sm-bench-profiles.

Install the solvers with `uv sync --group benchmark` and build the C solvers with
`make -C C`, both from the repository root, then run from this folder with
`python benchmark.py`, or with `--tiers 2 --seeds 1` for a short run.

The test set is problems 0 to seeds - 1 of each tier in TIERS, n = 4p - 4 half-spaces in
dimension p, drawn by problem_generators.problem and posed as the QP of minimising
||x - z||^2 / 2 subject to A x <= b. The QP solvers are called through qpsolvers under the
four accuracy profiles of PROFILES, which set each solver's options as qpbenchmark 2.7.1
does: its own defaults, or absolute tolerances of 1e-3, 1e-6 and 1e-9 with relative ones
zero, all with a time limit of TIME_LIMIT seconds where the solver takes one. A QP run is
solved when the solver reports a solution. The C accelerated solver and C Dykstra run under
a budget of BUDGET cycles and are solved when they settle, which certifies the projection
for the accelerated solver and is a cycle that leaves the state unchanged for Dykstra. No
profile changes their runs, hence each is measured once per problem. The Python oracle
replays the schedule of active sets of one Dykstra run, recorded untimed, and is solved when
that run reached a fixed point of the iteration within BUDGET cycles. Only problems on which
C Dykstra settles are recorded, because recording is plain Python Dykstra.

Every solver is called once, untimed, under every profile before the timed runs start,
because the first call pays a one-time initialisation. Every time is the median of runs
repeated until they total REPEAT_SECONDS or number REPEATS, and excludes the conversion of a problem to sparse form for solvers that take only sparse
input, as qpbenchmark does. The primal residual, dual residual and duality gap of each
returned point follow qpbenchmark's definitions. Results go to results/paper beside this
script.
"""

import argparse
import csv
import importlib.metadata as metadata
import math
import os
import statistics
import sys
import time

import numpy as np

import c_solvers
import paper_figures
from convex_projection_solver import ConvexProjectionSolver
from oracle import oracle_lti_projection, record_schedule
from paper_figures import quiet_accelerate_warnings, save_tex
from performance_timing import machine_description
from problem_generators import problem, problem_seed


TIERS = tuple((4 * p - 4, p) for p in (4, 6, 8, 10, 14, 18, 26, 38, 64))
SEEDS = 5
BUDGET = 10**6
TIME_LIMIT = 10.0
REPEAT_SECONDS, REPEATS = 0.5, 5

QP_SOLVERS = ("clarabel", "cvxopt", "daqp", "highs", "osqp", "piqp", "proxqp", "qpalm",
              "quadprog", "scs")
PACKAGES = {"clarabel": "clarabel", "cvxopt": "cvxopt", "daqp": "daqp", "highs": "highspy",
            "osqp": "osqp", "piqp": "piqp", "proxqp": "proxsuite", "qpalm": "qpalm",
            "quadprog": "quadprog", "scs": "scs"}
PROJECTION_METHODS = ("accelerated", "Dykstra", "oracle")
# The rows of tab:benchmark, grouped as the paper groups them
GROUPS = (("projection based, this paper", ("Dykstra", "oracle", "accelerated")),
          ("first order", ("scs", "osqp")),
          ("second order", ("daqp", "quadprog", "highs", "clarabel", "piqp", "cvxopt", "qpalm",
                            "proxqp")))
PROFILES = {"default": None, "low accuracy": 1e-3, "mid accuracy": 1e-6, "high accuracy": 1e-9}

# The option names qpbenchmark 2.7.1 sets for an absolute tolerance, a relative tolerance and
# a time limit, in qpbenchmark/solver_settings.py, for the solvers installed here
ABSOLUTE_OPTIONS = {
    "clarabel": ("tol_feas", "tol_gap_abs"), "cvxopt": ("feastol",),
    "daqp": ("dual_tol", "primal_tol"),
    "highs": ("dual_feasibility_tolerance", "primal_feasibility_tolerance"),
    "osqp": ("eps_abs",), "piqp": ("eps_abs", "eps_duality_gap_abs"),
    "proxqp": ("eps_abs", "eps_duality_gap_abs"), "qpalm": ("eps_abs",), "scs": ("eps_abs",),
}
RELATIVE_OPTIONS = {
    "clarabel": ("tol_gap_rel",), "osqp": ("eps_rel",),
    "piqp": ("eps_duality_gap_rel", "eps_rel"), "proxqp": ("eps_duality_gap_rel", "eps_rel"),
    "qpalm": ("eps_rel",), "scs": ("eps_rel",),
}
TIME_LIMIT_OPTIONS = {"highs": "time_limit", "osqp": "time_limit", "qpalm": "time_limit",
                      "scs": "time_limit_secs"}


def solver_options(solver: str, profile: str) -> dict:
    """The options qpbenchmark passes to a solver under an accuracy profile."""
    options = {}
    tolerance = PROFILES[profile]
    if tolerance is not None:
        options.update({name: tolerance for name in ABSOLUTE_OPTIONS.get(solver, ())})
        options.update({name: 0.0 for name in RELATIVE_OPTIONS.get(solver, ())})
        if solver in ("piqp", "proxqp"):
            options["check_duality_gap"] = True
    if solver in TIME_LIMIT_OPTIONS:
        options[TIME_LIMIT_OPTIONS[solver]] = TIME_LIMIT
    return options


def residuals(z: np.ndarray, A: np.ndarray, b: np.ndarray, x: np.ndarray,
              multipliers: np.ndarray) -> tuple[float, float, float]:
    """Primal residual, dual residual and duality gap of a point and its multipliers.

    For the QP with P = I, q = -z, G = A and h = b these are max([A x - b]^+),
    ||x - z + A^T lambda||_inf and |x^T x - z^T x + b^T lambda|, as qpbenchmark defines them.
    """
    primal = float(max(0.0, np.max(A @ x - b))) if len(b) else 0.0
    dual = float(np.max(np.abs(x - z + A.T @ multipliers)))
    gap = float(abs(x @ x - z @ x + b @ multipliers))
    return primal, dual, gap


def repeated(call) -> tuple[float, int, object]:
    """Median seconds of repeated calls, their count, and the last call's result."""
    timings, result = [], None
    while not timings or (sum(timings) < REPEAT_SECONDS and len(timings) < REPEATS):
        start = time.perf_counter()
        result = call()
        timings.append(time.perf_counter() - start)
    return statistics.median(timings), len(timings), result


def run_qp_solver(solver: str, profile: str, z: np.ndarray, A: np.ndarray,
                  b: np.ndarray) -> dict:
    """One QP solver under one profile, through qpsolvers."""
    import qpsolvers
    from scipy import sparse

    qp = qpsolvers.Problem(np.eye(len(z)), -z, A, b)
    if solver in qpsolvers.sparse_solvers and solver not in qpsolvers.dense_solvers:
        qp = qpsolvers.Problem(sparse.csc_matrix(qp.P), qp.q, sparse.csc_matrix(qp.G), qp.h)
    options = solver_options(solver, profile)

    def call():
        try:
            return qpsolvers.solve_problem(qp, solver=solver, **options)
        except Exception as error:  # a solver's failure counts as unsolved, as in qpbenchmark
            print(f"    {solver} ({profile}) raised {type(error).__name__}: {error}", flush=True)
            return qpsolvers.Solution(qp)

    seconds, runs, solution = repeated(call)
    found = bool(solution.found) and solution.x is not None
    record = {"solver": solver, "profile": profile, "solved": found, "seconds": seconds,
              "runs": runs, "primal residual": math.nan, "dual residual": math.nan,
              "duality gap": math.nan, "projection error": math.nan}
    if found:
        x = np.asarray(solution.x, dtype=float)
        multipliers = (np.asarray(solution.z, dtype=float) if solution.z is not None
                       else np.full(len(b), math.nan))
        record.update(zip(("primal residual", "dual residual", "duality gap"),
                          residuals(z, A, b, x, multipliers)))
        record["projection x"] = x
    return record


def run_projection_method(method: str, z: np.ndarray, A: np.ndarray, b: np.ndarray,
                          dykstra: dict | None) -> dict:
    """One projection method on one problem.

    dykstra is C Dykstra's record on the same problem, which the oracle needs: its schedule
    is recorded over the cycles C Dykstra took to settle, and it is solved when its replay
    lands on C Dykstra's settled point to within 1e-9 of that point's largest coordinate.
    """
    record = {"solver": method, "profile": "all", "primal residual": math.nan,
              "dual residual": math.nan, "duality gap": math.nan, "cycles": None}
    if method == "oracle":
        if dykstra is None or not dykstra["solved"]:
            record.update(solved=False, seconds=math.nan, runs=0)
            return record
        # Python rounds differently from C and need not reach a state that a cycle leaves
        # unchanged bit for bit, hence the recording stops at C Dykstra's cycle count
        schedule, _ = record_schedule(z, A, b, dykstra["cycles"])
        seconds, runs, result = repeated(lambda: oracle_lti_projection(z, A, b, schedule))
        difference = float(np.max(np.abs(result.projection - dykstra["point"])))
        record.update(solved=difference <= 1e-9 * (1.0 + np.max(np.abs(dykstra["point"]))),
                      seconds=seconds, runs=runs, cycles=sum(k for _, k in schedule),
                      **{"projection x": result.projection, "replay difference": difference})
        record["primal residual"] = float(max(0.0, np.max(A @ result.projection - b)))
        return record
    function = (c_solvers.accelerated_c_projection if method == "accelerated"
                else c_solvers.dykstra_c_projection)
    seconds, runs, result = repeated(lambda: function(z, A, b, BUDGET))
    record.update(solved=bool(result.settled), seconds=seconds, runs=runs, cycles=result.cycles,
                  point=result.projection, **{"projection x": result.projection})
    record.update(zip(("primal residual", "dual residual", "duality gap"),
                      residuals(z, A, b, result.projection, result.multipliers)))
    return record


def benchmark(tiers: tuple = TIERS, seeds: int = SEEDS, solvers: tuple = QP_SOLVERS,
              profiles: tuple = tuple(PROFILES), oracle: bool = True) -> list[dict]:
    """Run every solver on every problem, returning one record per run."""
    A, b, z = problem(*tiers[0], problem_seed(tiers[0][0], 0))
    for solver in solvers:
        for profile in profiles:
            run_qp_solver(solver, profile, z, A, b)
    for method in ("accelerated", "Dykstra"):
        run_projection_method(method, z, A, b, None)
    records = []
    for num_planes, num_dimensions in tiers:
        for index in range(seeds):
            started = time.perf_counter()
            A, b, z = problem(num_planes, num_dimensions, problem_seed(num_planes, index))
            reference = ConvexProjectionSolver._find_optimal_solution(z, A, b)
            problem_records = []
            dykstra = run_projection_method("Dykstra", z, A, b, None)
            problem_records.append(dykstra)
            problem_records.append(run_projection_method("accelerated", z, A, b, None))
            if oracle:
                problem_records.append(run_projection_method("oracle", z, A, b, dykstra))
            for solver in solvers:
                for profile in profiles:
                    problem_records.append(run_qp_solver(solver, profile, z, A, b))
            for record in problem_records:
                record.pop("point", None)
                x = record.pop("projection x", None)
                record["projection error"] = (float(np.max(np.abs(x - reference)))
                                              if x is not None else math.nan)
                record.update(tier=f"{num_planes}x{num_dimensions}", num_planes=num_planes,
                              num_dimensions=num_dimensions, index=index,
                              seed=problem_seed(num_planes, index))
            records.extend(problem_records)
            solved = {r["solver"] for r in problem_records if r["solved"]}
            print(f"{num_planes} x {num_dimensions} index {index}: Dykstra "
                  f"{'settled at ' + str(dykstra['cycles']) if dykstra['solved'] else 'unsettled'}"
                  f", unsolved by {sorted(set(r['solver'] for r in problem_records) - solved)} "
                  f"in some profile ({time.perf_counter() - started:.1f} s)", flush=True)
    return records


def runs_of(records: list[dict], solver: str, profile: str, tier: str | None = None) -> list:
    """The records of one solver under one profile, the projection methods' in every profile."""
    return [r for r in records if r["solver"] == solver
            and r["profile"] in (profile, "all") and (tier is None or r["tier"] == tier)]


def format_ms(seconds: float) -> str:
    """Milliseconds to three significant figures, whole above 100."""
    ms = 1000.0 * seconds
    return f"{ms:.0f}" if ms >= 100 else f"{ms:.3g}"


def tiers_table(records: list[dict], tiers: tuple, seeds: int) -> str:
    """Rows of tab:sm-bench-tiers: per tier, solved and the median time over the solved."""
    lines = []
    for num_planes, num_dimensions in tiers:
        cells = []
        for method in ("accelerated", "Dykstra"):
            runs = runs_of(records, method, "low accuracy", f"{num_planes}x{num_dimensions}")
            solved = [r["seconds"] for r in runs if r["solved"]]
            median = f"${format_ms(statistics.median(solved))}$ ms" if solved else "---"
            cells.append(f"${len(solved)}/{len(runs)}$ & {median}")
        size = f"${num_planes} \\times {num_dimensions}$"
        lines.append(f"  {size:<16} & " + " & ".join(cells) + r" \\")
    return "\n".join(lines) + "\n"


def geometric_mean_ms(records: list[dict], solver: str, profile: str) -> float:
    """Geometric-mean milliseconds over a solver's solved runs, infinite when none is solved."""
    solved = [r["seconds"] for r in runs_of(records, solver, profile) if r["solved"]]
    return 1000.0 * math.exp(statistics.fmean(math.log(s) for s in solved)) if solved else math.inf


def profiles_table(records: list[dict], solvers: tuple) -> str:
    """Rows of tab:sm-bench-profiles: solved and geometric-mean time over the solved.

    The projection methods come first, then the QP solvers by low-accuracy mean, as in the
    paper.
    """
    lines = []
    projection = [s for s in solvers if s in PROJECTION_METHODS]
    others = sorted((s for s in solvers if s not in PROJECTION_METHODS),
                    key=lambda s: geometric_mean_ms(records, s, "low accuracy"))
    for solver in projection + ["---"] + others:
        if solver == "---":
            lines.append(r"  \hline")
            continue
        cells = []
        for profile in PROFILES:
            runs = runs_of(records, solver, profile)
            solved = [r["seconds"] for r in runs if r["solved"]]
            mean = f"${geometric_mean_ms(records, solver, profile):.2f}$" if solved else "---"
            cells.append(f"${len(solved)}/{len(runs)}$ & {mean}")
        lines.append(f"  {solver:<11} & " + " & ".join(cells) + r" \\")
    return "\n".join(lines) + "\n"


def headline_table(records: list[dict], solvers: tuple, shown: tuple) -> str:
    """Dense rows of tab:benchmark at 1e-3, grouped as in GROUPS: solved, and per shown tier
    the median time with an unsolved run counted at its own cost, which for the oracle is not
    measured."""
    lines = []
    for group, members in GROUPS:
        present = [s for s in members if s in solvers]
        if not present:
            continue
        if lines:
            lines.append(r"  \hline")
        lines.append(f"  \\multicolumn{{{len(shown) + 2}}}{{l}}{{\\textit{{{group}}}}} \\\\")
        lines.extend(headline_row(records, solver, shown) for solver in present)
    return "\n".join(lines) + "\n"


def headline_tabular(records: list[dict], solvers: tuple, shown: tuple, problems: int) -> str:
    """The whole tabular of tab:benchmark, which keeps only the dense columns."""
    columns = " & ".join(f"{{\\footnotesize ${n}{{\\times}}{p}$}}" for n, p in shown)
    return (f"\\begin{{tabular}}{{l {'c' * (len(shown) + 1)}}}\n  \\hline\n"
            f"  & \\multicolumn{{{len(shown) + 1}}}{{c}}{{dense (${problems}$)}} \\\\\n"
            f"  solver & solved & {columns} \\\\\n  \\hline\n"
            + headline_table(records, solvers, shown) + "  \\hline\n\\end{tabular}\n")


def headline_row(records: list[dict], solver: str, shown: tuple) -> str:
    """One row of tab:benchmark."""
    runs = runs_of(records, solver, "low accuracy")
    cells = [f"${sum(r['solved'] for r in runs)}$"]
    for num_planes, num_dimensions in shown:
        tier_runs = runs_of(records, solver, "low accuracy", f"{num_planes}x{num_dimensions}")
        costs = [r["seconds"] for r in tier_runs]
        cells.append("---" if not costs or any(math.isnan(c) for c in costs)
                     else f"${format_ms(statistics.median(costs))}$")
    name = {"accelerated": "accelerated (C)", "Dykstra": "Dykstra (C)",
            "oracle": "oracle (Python)"}.get(solver, solver)
    return f"  {name:<16} & " + " & ".join(cells) + r" \\"


def write_records(records: list[dict]) -> str:
    """Write every run as CSV and return the path."""
    os.makedirs(paper_figures.OUTPUT_DIR, exist_ok=True)
    path = os.path.join(paper_figures.OUTPUT_DIR, "benchmark.csv")
    fields = sorted({key for record in records for key in record})
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(records)
    print(f"Wrote {path}")
    return path


def describe() -> list[str]:
    """The machine, the package versions and the method, as lines."""
    lines = [f"{key}: {value}" for key, value in machine_description().items()]
    versions = ", ".join(f"{package} {metadata.version(package)}"
                         for package in ["qpsolvers", "scipy"] + sorted(set(PACKAGES.values())))
    lines.append(f"packages: {versions}")
    lines.append(f"method: each time the median of runs repeated until they total "
                 f"{REPEAT_SECONDS} s or number {REPEATS}; time limit {TIME_LIMIT} s; projection "
                 f"budget {BUDGET} cycles; profiles as qpbenchmark 2.7.1 sets them")
    return lines


def run(tiers: tuple = TIERS, seeds: int = SEEDS, oracle: bool = True,
        command: str = "python benchmark.py") -> list[dict]:
    """Benchmark, print the method, and write the CSV and the three tables."""
    header = describe()
    for line in header:
        print(line)
    records = benchmark(tiers, seeds, oracle=oracle)
    write_records(records)

    comment = "".join(f"% {line}\n" for line in header)
    methods = ("accelerated", "Dykstra") + (("oracle",) if oracle else ())
    shown = tuple(t for t in tiers if t in ((36, 10), (100, 26), (252, 64))) or tiers[-1:]
    for name, text in (
            ("tab-sm-bench-tiers", tiers_table(records, tiers, seeds)),
            ("tab-sm-bench-profiles", profiles_table(records, ("accelerated", "Dykstra") + QP_SOLVERS)),
            ("tab-benchmark-dense", headline_tabular(records, methods + QP_SOLVERS, shown,
                                                     len(tiers) * seeds))):
        print(text)
        save_tex(comment + text, name, command)

    projection = [r for r in records if r["solver"] in ("accelerated", "Dykstra") and r["solved"]]
    if projection:
        print(f"largest primal residual over the settled projection runs: "
              f"{max(r['primal residual'] for r in projection):.2g}; accelerated runs with a "
              f"dual residual above 1e-3: "
              f"{sum(r['dual residual'] > 1e-3 for r in projection if r['solver'] == 'accelerated')}")
    return records


def main() -> None:
    """Parse the command line and run."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--tiers", type=int, default=len(TIERS),
                        help="number of tiers to run, from the smallest")
    parser.add_argument("--seeds", type=int, default=SEEDS)
    parser.add_argument("--no-oracle", action="store_true")
    args = parser.parse_args()
    quiet_accelerate_warnings()
    run(TIERS[:args.tiers], args.seeds, not args.no_oracle,
        " ".join(["python benchmark.py", *sys.argv[1:]]))


if __name__ == "__main__":
    main()
