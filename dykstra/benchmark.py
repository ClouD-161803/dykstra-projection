"""Benchmark the projection methods against general-purpose QP solvers on three families of
problems, the protocol behind the SIAM paper's tab:benchmark and the supplement's
tab:sm-bench-tiers and tab:sm-bench-profiles.

Install the solvers with `uv sync --group benchmark` and build the C solvers with
`make -C C`, then run with `python -m dykstra.benchmark --family all`, all from the repository
root. `--family` picks dense, the default, wedge, stall, or all three. `--tiers 2 --seeds 1`
gives a short run, `--quick` one problem of each chosen family written to a temporary
directory, and `--from-csv` rebuilds the tables from the CSV files of an earlier run on the same
machine without timing again.

Every problem is posed as the QP of minimising ||x - z||^2 / 2 subject to A x <= b, and the
families are the following, problems 0 to seeds - 1 of each tier.

- dense: the tiers of TIERS, n = 4p - 4 half-spaces in dimension p, drawn by
  problem_generators.problem from Generator 2 with an infeasible start.
- wedge: problem_generators.WedgeTestPlanes at each angle of WEDGE_ANGLES in p = 3 and 8,
  from its canonical start, on which Dykstra's algorithm has a slow phase.
- stall: problem_generators.BoxSlabTestPlanes in p = 2, 4, and 8, the square-and-line example
  lifted, on which Dykstra's algorithm has a stalling period.

The QP solvers are called through qpsolvers under the four accuracy profiles of PROFILES, which
set each solver's options as qpbenchmark 2.7.1 does: its own defaults, or absolute tolerances of
1e-3, 1e-6 and 1e-9 with relative ones zero, all with a time limit of TIME_LIMIT seconds where
the solver takes one. A QP run is solved when the solver reports a solution. The C accelerated
solver and C Dykstra run under a budget of BUDGET cycles and are solved when they settle, which
certifies the projection for the accelerated solver and is a cycle that leaves the state
unchanged for Dykstra. No profile changes their runs, hence each is measured once per problem.
The Python oracle replays the schedule of active sets of one Dykstra run, recorded untimed, and
is solved when that run reached a fixed point of the iteration within BUDGET cycles. Only
problems on which C Dykstra settles are recorded, because recording is plain Python Dykstra.

Every solver is called once, untimed, under every profile before the timed runs of a family
start, because the first call pays a one-time initialisation. Every time is the median of runs
repeated until they total REPEAT_SECONDS or number REPEATS, and excludes the conversion of a
problem to sparse form for solvers that take only sparse input, as qpbenchmark does. The primal
residual, dual residual and duality gap of each returned point follow qpbenchmark's definitions.

Results go to results/paper at the repository root. Each family gets the CSV of every run, the
rows of its per-tier and per-profile tables, and its headline tabular. The dense outputs keep
the names the paper includes, benchmark.csv, tab-sm-bench-tiers.tex, tab-sm-bench-profiles.tex,
and tab-benchmark-dense.tex. Those of the wedge and stall families are benchmark-<family>.csv,
tab-sm-bench-tiers-<family>.tex, tab-sm-bench-profiles-<family>.tex, and
tab-benchmark-<family>.tex. A run of all three also writes tab-benchmark.tex, the headline with
the families as column groups.
"""

import argparse
import csv
import importlib.metadata as metadata
import math
import os
import statistics
import sys
import tempfile
import time

import numpy as np

from dykstra import c_solvers, paper_figures
from dykstra.convex_projection_solver import ConvexProjectionSolver
from dykstra.oracle import oracle_lti_projection, record_schedule
from dykstra.paper_figures import quiet_accelerate_warnings, save_tex
from dykstra.performance_timing import machine_description, read_records
from dykstra.problem_generators import BoxSlabTestPlanes, WedgeTestPlanes, problem, problem_seed


TIERS = tuple((4 * p - 4, p) for p in (4, 6, 8, 10, 14, 18, 26, 38, 64))
SEEDS = 5
# The tiers of the dense columns of tab:benchmark
SHOWN_TIERS = ((36, 10), (100, 26), (252, 64))

# The tiers of the two structured families: wedges at three angles, in rad, in p = 3 and 8, and
# boxes and slabs in p = 2, 4, and 8
WEDGE_ANGLES = (0.1, 0.03, 0.01)
WEDGE_TIERS = tuple((theta, p) for p in (3, 8) for theta in WEDGE_ANGLES)
STALL_TIERS = ((2,), (4,), (8,))
FAMILY_TIERS = {"dense": TIERS, "wedge": WEDGE_TIERS, "stall": STALL_TIERS}

# Chosen here: the seed blocks of the two structured families, for index < 100, apart from those
# of problem_seed, stall_grid.py, stall_angles.py, and activity_study.py
WEDGE_SEED, STALL_SEED = 5_000_000, 6_000_000
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


def family_problem(family: str, tier: tuple, index: int) -> tuple:
    """Problem number index of a tier of a family, as (A, b, z, seed)."""
    if family == "dense":
        num_planes, num_dimensions = tier
        seed = problem_seed(num_planes, index)
        return (*problem(num_planes, num_dimensions, seed), seed)
    if family == "wedge":
        theta, num_dimensions = tier
        seed = WEDGE_SEED + 1000 * num_dimensions + 100 * WEDGE_ANGLES.index(theta) + index
        return (*WedgeTestPlanes(theta, num_dimensions).problem(seed), seed)
    (num_dimensions,) = tier
    seed = STALL_SEED + 1000 * num_dimensions + index
    return (*BoxSlabTestPlanes(num_dimensions).problem(seed), seed)


def tier_fields(family: str, tier: tuple) -> dict:
    """The fields that name a tier in a record: its name, n, p, and a wedge's theta."""
    if family == "dense":
        num_planes, num_dimensions = tier
        return {"tier": f"{num_planes}x{num_dimensions}", "num_planes": num_planes,
                "num_dimensions": num_dimensions}
    if family == "wedge":
        theta, num_dimensions = tier
        num_planes = WedgeTestPlanes(theta, num_dimensions).num_planes
        return {"tier": f"{num_planes}x{num_dimensions} theta {theta:g}",
                "num_planes": num_planes, "num_dimensions": num_dimensions, "theta": theta}
    (num_dimensions,) = tier
    num_planes = BoxSlabTestPlanes(num_dimensions).num_planes
    return {"tier": f"{num_planes}x{num_dimensions}", "num_planes": num_planes,
            "num_dimensions": num_dimensions}


def benchmark(tiers: tuple = TIERS, seeds: int = SEEDS, solvers: tuple = QP_SOLVERS,
              profiles: tuple = tuple(PROFILES), oracle: bool = True,
              family: str = "dense") -> list[dict]:
    """Run every solver on every problem of the given tiers of a family, returning one record
    per run."""
    A, b, z, _ = family_problem(family, tiers[0], 0)
    for solver in solvers:
        for profile in profiles:
            run_qp_solver(solver, profile, z, A, b)
    for method in ("accelerated", "Dykstra"):
        run_projection_method(method, z, A, b, None)
    records = []
    for tier in tiers:
        fields = tier_fields(family, tier)
        for index in range(seeds):
            started = time.perf_counter()
            A, b, z, seed = family_problem(family, tier, index)
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
                record.update(family=family, index=index, seed=seed, **fields)
            records.extend(problem_records)
            solved = {r["solver"] for r in problem_records if r["solved"]}
            print(f"{family} {fields['tier']} index {index}: Dykstra "
                  f"{'settled at ' + str(dykstra['cycles']) if dykstra['solved'] else 'unsettled'}"
                  f", unsolved by {sorted(set(r['solver'] for r in problem_records) - solved)} "
                  f"in some profile ({time.perf_counter() - started:.1f} s)", flush=True)
    return records


def runs_of(records: list[dict], solver: str, profile: str, where: dict | None = None) -> list:
    """The records of one solver under one profile, the projection methods' in every profile,
    restricted to those whose fields take the values in where."""
    return [r for r in records if r["solver"] == solver and r["profile"] in (profile, "all")
            and all(r.get(key) == value for key, value in (where or {}).items())]


def format_ms(seconds: float) -> str:
    """Milliseconds to three significant figures, whole above 100."""
    ms = 1000.0 * seconds
    return f"{ms:.0f}" if ms >= 100 else f"{ms:.3g}"


def tier_label(record: dict) -> str:
    """The LaTeX name of a record's tier: its size, and a wedge's angle."""
    size = f"${record['num_planes']} \\times {record['num_dimensions']}$"
    theta = record.get("theta")
    return size if theta is None else f"{size}, $\\theta = {theta:g}$"


def tiers_table(records: list[dict]) -> str:
    """Rows of tab:sm-bench-tiers: per tier, solved and the median time over the solved."""
    first = {}
    for record in records:
        first.setdefault(record["tier"], record)
    lines = []
    for tier, record in first.items():
        cells = []
        for method in ("accelerated", "Dykstra"):
            runs = runs_of(records, method, "low accuracy", {"tier": tier})
            solved = [r["seconds"] for r in runs if r["solved"]]
            median = f"${format_ms(statistics.median(solved))}$ ms" if solved else "---"
            cells.append(f"${len(solved)}/{len(runs)}$ & {median}")
        lines.append(f"  {tier_label(record):<16} & " + " & ".join(cells) + r" \\")
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


def headline_columns(family: str, records: list[dict], combined: bool = False) -> list[tuple]:
    """The time columns of a family in tab:benchmark, each a header and the record fields it
    selects.

    Dense shows the tiers of SHOWN_TIERS that ran, or else the largest. Beside dense, a family
    shows one median over all its problems, and alone, one median per angle or per p.
    """
    if family == "dense":
        present = {(r["num_planes"], r["num_dimensions"]) for r in records}
        shown = [t for t in SHOWN_TIERS if t in present] or [max(present)]
        return [(f"{{\\footnotesize ${n}{{\\times}}{p}$}}", {"tier": f"{n}x{p}"})
                for n, p in shown]
    if combined:
        return [("med", {})]
    if family == "wedge":
        return [(f"{{\\footnotesize $\\theta{{=}}{theta:g}$}}", {"theta": theta})
                for theta in dict.fromkeys(r["theta"] for r in records)]
    return [(f"{{\\footnotesize $p{{=}}{p}$}}", {"num_dimensions": p})
            for p in dict.fromkeys(r["num_dimensions"] for r in records)]


def headline_cells(records: list[dict], solver: str, columns: list[tuple]) -> list[str]:
    """One family's cells of a row of tab:benchmark at 1e-3: solved, and per column the median
    time with an unsolved run counted at its own cost, which for the oracle is not measured."""
    runs = runs_of(records, solver, "low accuracy")
    cells = [f"${sum(r['solved'] for r in runs)}$"]
    for _, where in columns:
        costs = [r["seconds"] for r in runs_of(records, solver, "low accuracy", where)]
        cells.append("---" if not costs or any(math.isnan(c) for c in costs)
                     else f"${format_ms(statistics.median(costs))}$")
    return cells


def headline_tabular(groups: list[tuple], solvers: tuple) -> str:
    """The whole tabular of tab:benchmark, one column group per family.

    groups holds (family, records, columns) with columns as headline_columns returns them, and
    the rows are grouped as in GROUPS.
    """
    spec = "|".join("c" * (len(columns) + 1) for _, _, columns in groups)
    width = 1 + sum(len(columns) + 1 for _, _, columns in groups)
    titles, heads = [], []
    for position, (family, records, columns) in enumerate(groups):
        problems = len({(r["tier"], r["index"]) for r in records})
        rule = "|" if position < len(groups) - 1 else ""
        titles.append(f"\\multicolumn{{{len(columns) + 1}}}{{c{rule}}}{{{family} (${problems}$)}}")
        heads.append(" & ".join(["solved"] + [header for header, _ in columns]))
    lines = []
    for group, members in GROUPS:
        present = [s for s in members if s in solvers]
        if not present:
            continue
        if lines:
            lines.append(r"  \hline")
        lines.append(f"  \\multicolumn{{{width}}}{{l}}{{\\textit{{{group}}}}} \\\\")
        for solver in present:
            cells = [c for _, records, columns in groups
                     for c in headline_cells(records, solver, columns)]
            name = {"accelerated": "accelerated (C)", "Dykstra": "Dykstra (C)",
                    "oracle": "oracle (Python)"}.get(solver, solver)
            lines.append(f"  {name:<16} & " + " & ".join(cells) + r" \\")
    return (f"\\begin{{tabular}}{{l {spec}}}\n  \\hline\n  & {' & '.join(titles)} \\\\\n"
            f"  solver & {' & '.join(heads)} \\\\\n  \\hline\n" + "\n".join(lines)
            + "\n  \\hline\n\\end{tabular}\n")


def output_names(family: str) -> dict:
    """The basenames of a family's CSV and tables, dense keeping the names the paper includes."""
    if family == "dense":
        return {"csv": "benchmark", "tiers": "tab-sm-bench-tiers",
                "profiles": "tab-sm-bench-profiles", "headline": "tab-benchmark-dense"}
    return {"csv": f"benchmark-{family}", "tiers": f"tab-sm-bench-tiers-{family}",
            "profiles": f"tab-sm-bench-profiles-{family}", "headline": f"tab-benchmark-{family}"}


def write_records(records: list[dict], name: str = "benchmark") -> str:
    """Write every run as CSV under name and return the path."""
    os.makedirs(paper_figures.OUTPUT_DIR, exist_ok=True)
    path = os.path.join(paper_figures.OUTPUT_DIR, f"{name}.csv")
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


def methods_shown(records: list[dict]) -> tuple:
    """The projection methods that ran, then the QP solvers."""
    oracle = ("oracle",) if any(r["solver"] == "oracle" for r in records) else ()
    return ("accelerated", "Dykstra") + oracle + QP_SOLVERS


def write_tables(records: list[dict], family: str, header: list[str], command: str) -> None:
    """Write a family's three tables, each headed by the machine and the method."""
    comment = "".join(f"% {line}\n" for line in header)
    names = output_names(family)
    for name, text in (
            (names["tiers"], tiers_table(records)),
            (names["profiles"], profiles_table(records, ("accelerated", "Dykstra") + QP_SOLVERS)),
            (names["headline"], headline_tabular(
                [(family, records, headline_columns(family, records))], methods_shown(records)))):
        print(text)
        save_tex(comment + text, name, command)


def median_cycles(runs: list[dict]) -> float:
    """The median cycle count of projection runs, NaN for runs that do not count cycles."""
    cycles = [r.get("cycles") for r in runs if isinstance(r.get("cycles"), (int, float))]
    return float(statistics.median(cycles)) if cycles else math.nan


def standing(records: list[dict], family: str) -> dict:
    """Where the accelerated solver stands on a family at 1e-3 against C Dykstra and the QP
    solver of smallest median time among those that solve every problem, over all the family's
    problems and per column of its own headline.

    The entries are keyed by "all" and by the fields a column selects, e.g. "theta 0.1", and
    hold per solver the problems solved, the median time with unsolved runs at their own cost,
    and for the projection methods the median cycles.
    """
    columns = [{}] + [where for _, where in headline_columns(family, records)]
    summary = {}
    for where in columns:
        entry = {}
        for solver in ("accelerated", "Dykstra") + QP_SOLVERS:
            runs = runs_of(records, solver, "low accuracy", where)
            entry[solver] = {"solved": sum(r["solved"] for r in runs), "problems": len(runs),
                             "ms": 1000.0 * statistics.median(r["seconds"] for r in runs),
                             "cycles": median_cycles(runs)}
        # a solver that fails fast is not the fastest, hence those that solve every problem
        # of the column come first
        complete = [s for s in QP_SOLVERS if entry[s]["solved"] == entry[s]["problems"]]
        best = min(complete or QP_SOLVERS, key=lambda s: entry[s]["ms"])
        entry["best QP"] = {"solver": best, **entry[best]}
        summary[" ".join(f"{key} {value}" for key, value in where.items()) or "all"] = entry
    return summary


def print_standing(records: list[dict], family: str) -> dict:
    """Print standing() as one line per column and return it."""
    summary = standing(records, family)
    for header, entry in summary.items():
        accelerated, dykstra, best = entry["accelerated"], entry["Dykstra"], entry["best QP"]
        print(f"{family} {header}: accelerated {accelerated['solved']}/{accelerated['problems']} "
              f"solved, median {accelerated['ms']:.3g} ms at {accelerated['cycles']:.0f} cycles; "
              f"Dykstra {dykstra['solved']}/{dykstra['problems']}, {dykstra['ms']:.3g} ms at "
              f"{dykstra['cycles']:.0f} cycles; fastest QP solver {best['solver']} "
              f"{best['solved']}/{best['problems']}, {best['ms']:.3g} ms")
    return summary


def run(tiers: tuple = TIERS, seeds: int = SEEDS, oracle: bool = True,
        command: str = "python -m dykstra.benchmark", family: str = "dense") -> list[dict]:
    """Benchmark one family, print the method, and write its CSV and its three tables."""
    header = describe()
    for line in header:
        print(line)
    records = benchmark(tiers, seeds, oracle=oracle, family=family)
    write_records(records, output_names(family)["csv"])
    write_tables(records, family, header, command)
    report_residuals(records)
    print_standing(records, family)
    return records


def report_residuals(records: list[dict]) -> None:
    """Print the largest primal residual of the settled projection runs and the accelerated runs
    with a large dual residual."""
    projection = [r for r in records if r["solver"] in ("accelerated", "Dykstra") and r["solved"]]
    if projection:
        print(f"largest primal residual over the settled projection runs: "
              f"{max(r['primal residual'] for r in projection):.2g}; accelerated runs with a "
              f"dual residual above 1e-3: "
              f"{sum(r['dual residual'] > 1e-3 for r in projection if r['solver'] == 'accelerated')}")


def write_combined(by_family: dict, header: list[str], command: str) -> str:
    """Write tab-benchmark.tex, the headline with the families as column groups."""
    groups = [(family, records, headline_columns(family, records, combined=True))
              for family, records in by_family.items()]
    solvers = tuple(dict.fromkeys(s for records in by_family.values()
                                  for s in methods_shown(records)))
    text = headline_tabular(groups, solvers)
    print(text)
    return save_tex("".join(f"% {line}\n" for line in header) + text, "tab-benchmark", command)


def main() -> None:
    """Parse the command line and run."""
    global REPEATS
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--family", choices=tuple(FAMILY_TIERS) + ("all",), default="dense")
    parser.add_argument("--tiers", type=int, default=None,
                        help="number of tiers to run per family, from the first")
    parser.add_argument("--seeds", type=int, default=SEEDS)
    parser.add_argument("--no-oracle", action="store_true")
    parser.add_argument("--from-csv", action="store_true",
                        help="rebuild the tables from the CSV files in results/paper")
    parser.add_argument("--quick", action="store_true",
                        help="one problem of the first tier of each family and one timed run "
                             "per solver, written to a temporary directory")
    args = parser.parse_args()
    quiet_accelerate_warnings()
    began = time.perf_counter()
    families = tuple(FAMILY_TIERS) if args.family == "all" else (args.family,)
    command = " ".join(["python -m dykstra.benchmark", *sys.argv[1:]])
    seeds = args.seeds
    if args.quick:
        REPEATS = 1
        paper_figures.OUTPUT_DIR = tempfile.mkdtemp(prefix="benchmark_")
        seeds = 1
    header = describe()
    for line in header:
        print(line)
    by_family = {}
    for family in families:
        names = output_names(family)
        if args.from_csv:
            records = read_records(os.path.join(paper_figures.OUTPUT_DIR, names["csv"] + ".csv"))
        else:
            tiers = FAMILY_TIERS[family][:1 if args.quick else args.tiers]
            records = benchmark(tiers, seeds, oracle=not args.no_oracle, family=family)
            write_records(records, names["csv"])
        write_tables(records, family, header, command)
        report_residuals(records)
        by_family[family] = records
    for family, records in by_family.items():
        print_standing(records, family)
    written = [f"{name}.{'csv' if kind == 'csv' else 'tex'}" for family in families
               for kind, name in output_names(family).items()
               if not (args.from_csv and kind == "csv")]
    if len(families) == len(FAMILY_TIERS):
        write_combined(by_family, header, command)
        written.append("tab-benchmark.tex")
    print(f"Finished in {time.perf_counter() - began:.0f} s: {', '.join(written)} in "
          f"{paper_figures.OUTPUT_DIR}")


if __name__ == "__main__":
    main()
