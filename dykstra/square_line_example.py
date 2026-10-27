"""Reproduce the square-and-line stalling example of the SIAM paper's Stalling section.

Run from the repository root with `python -m dykstra.square_line_example`, or with `--quick`
to write into a temporary directory, since the example takes well under a second. The square [-1, 1]^2 is four
half-spaces, right, left, top and bottom, and the line through (0, 1) and (2, 0) is two
opposite half-spaces, below and above it. The script prints, and writes to
results/paper/tab-square-line.tex, the stalling state after the first cycle from
(-4.1, 1.4), its points xi_1 to xi_7 and signed distances beta_m, the stall length N_stall,
the dual bound on it, the stall lengths of further starts, the ratios from (-4, 1.4), the
stall in the order that visits the line first, and what the accelerated solver does with
and without its KKT test. Every count from the theorem is checked against plain Dykstra.
"""

import argparse
import math
import tempfile
import time
from contextlib import contextmanager
from fractions import Fraction

import numpy as np

from dykstra import lti_solver, paper_figures
from dykstra.convex_projection_solver import ConvexProjectionSolver
from dykstra.lti_numerics import exact_cycle, unit_constraints
from dykstra.lti_solver import LTIVer5Solver
from dykstra.paper_figures import quiet_accelerate_warnings, save_tex, tex_number


START = np.array([-4.1, 1.4])
SQUARE = (np.array([[1.0, 0.0], [-1.0, 0.0], [0.0, 1.0], [0.0, -1.0]]), np.ones(4))
LINE = (np.array([[0.5, 1.0], [-0.5, -1.0]]), np.array([1.0, -1.0]))


def square_line_problem(line_first: bool = False) -> tuple[np.ndarray, np.ndarray]:
    """The six half-spaces, square first as in the paper, or line first as Bauschke et al."""
    parts = (LINE, SQUARE) if line_first else (SQUARE, LINE)
    return np.vstack([A for A, _ in parts]), np.hstack([b for _, b in parts])


def dykstra_cycles(z: np.ndarray, A: np.ndarray, b: np.ndarray, cycles: int) -> list[tuple]:
    """Plain Dykstra's state and scalars after each cycle, cycle 0 being the start."""
    unit_A, unit_b = unit_constraints(A, b)
    x, y = z.astype(float), np.zeros(len(b))
    states = [(x.copy(), y.copy(), None)]
    for _ in range(cycles):
        x, y, points = exact_cycle(x, y, unit_A, unit_b)
        states.append((x.copy(), y.copy(), points))
    return states


def stalling_state(A: np.ndarray, b: np.ndarray, x: np.ndarray, y: np.ndarray) -> dict:
    """The points and signed distances of the stationary-state definition.

    Takes the state x after cycle t and the scalars y stored in it, and returns the active
    set, the points xi_1 to xi_{n+1}, the signed distances beta, whether the state is
    stationary, the set D of active half-spaces with negative beta, and N_stall and
    m_switch of the stall-length theorem, all with half-spaces counted from 1.
    """
    unit_A, unit_b = unit_constraints(A, b)
    active = y > 0.0
    points, beta = [x.copy()], np.zeros(len(b))
    for m, (normal, offset) in enumerate(zip(unit_A, unit_b)):
        beta[m] = normal @ points[-1] - offset
        points.append(points[-1] - beta[m] * normal if active[m] else points[-1].copy())
    stationary = (np.allclose(points[-1], points[0], atol=1e-12)
                  and bool(np.all(beta[~active] <= 1e-12)))
    draining = np.flatnonzero(active & (beta < -1e-12))
    ratios = {int(m) + 1: float(y[m] / -beta[m]) for m in draining}
    n_stall = min(math.ceil(r - 1e-9) for r in ratios.values()) if ratios else None
    m_switch = (min(m for m, r in ratios.items() if math.ceil(r - 1e-9) == n_stall)
                if ratios else None)
    return {"active": [int(m) + 1 for m in np.flatnonzero(active)], "points": points,
            "beta": beta, "stationary": stationary, "ratios": ratios, "n_stall": n_stall,
            "m_switch": m_switch, "unit_A": unit_A, "unit_b": unit_b}


def rational_dykstra_cycles(z: np.ndarray, A: np.ndarray, b: np.ndarray,
                            cycles: int) -> list[tuple]:
    """Plain Dykstra in exact rational arithmetic, cycle 0 being the start.

    The data are read as the decimals they print as, and each correction is stored as a
    multiple of its row as given, which keeps every projection rational even for the line,
    whose unit normal is irrational. Returns the state, the multiples and the intermediate
    points after each cycle.
    """
    def rational(values) -> list:
        return [Fraction(repr(float(v))) for v in values]

    rows, offsets = [rational(row) for row in A], rational(b)
    x, multiples = rational(z), [Fraction(0)] * len(offsets)
    states = [(x, list(multiples), None)]
    for _ in range(cycles):
        points = []
        for m, (row, offset) in enumerate(zip(rows, offsets)):
            shifted = [xi + multiples[m] * ai for xi, ai in zip(x, row)]
            slack = sum(ai * si for ai, si in zip(row, shifted)) - offset
            multiples[m] = max(slack, Fraction(0)) / sum(ai * ai for ai in row)
            x = [si - multiples[m] * ai for si, ai in zip(shifted, row)]
            points.append(x)
        states.append((x, list(multiples), points))
    return states


def first_inactive_cycle(states: list[tuple], half_space: int, after: int) -> int:
    """The first cycle after the given one in which a half-space, counted from 1, is inactive."""
    for cycle in range(after + 1, len(states)):
        if states[cycle][1][half_space - 1] <= 0.0:
            return cycle
    raise RuntimeError("The half-space stays active through every recorded cycle.")


def first_moved_cycle(states: list[tuple], reference, after: int) -> int:
    """The first cycle after the given one whose intermediate points differ from reference.

    Floating-point points are compared to within 1e-12, and rational points, passed as a
    list, exactly.
    """
    for cycle in range(after + 1, len(states)):
        points = states[cycle][2]
        if isinstance(reference, list) and points != reference:
            return cycle
        if not isinstance(reference, list) and not np.allclose(points, reference, atol=1e-12):
            return cycle
    raise RuntimeError("The points repeat through every recorded cycle.")


def dual(z: np.ndarray, unit_A: np.ndarray, unit_b: np.ndarray, y: np.ndarray) -> float:
    """The dual function Phi of the paper at the scalars y."""
    return float(y @ (unit_A @ z - unit_b) - 0.5 * np.sum((unit_A.T @ y) ** 2))


class TracedSolver(LTIVer5Solver):
    """LTIVer5Solver recording how it reaches each cycle: exactly, evaluated, or jumped."""

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.trace = []
        self._jumping = False

    def _exact_cycle(self, cycle: int) -> tuple:
        self.trace.append((cycle, "run exactly"))
        return super()._exact_cycle(cycle)

    def _record_cycle(self, cycle: int) -> None:
        self.trace.append((cycle, "jumped over" if self._jumping else "evaluated"))
        super()._record_cycle(cycle)

    def _jump(self, *args) -> tuple:
        self._jumping = True
        try:
            return super()._jump(*args)
        finally:
            self._jumping = False


@contextmanager
def kkt_test_disabled():
    """Make every KKT test of the LTI solver fail while the context is open."""
    original = lti_solver.kkt_certificate
    lti_solver.kkt_certificate = lambda *args, **kwargs: None
    try:
        yield
    finally:
        lti_solver.kkt_certificate = original


def compress_trace(trace: list[tuple]) -> list[str]:
    """Runs of consecutive cycles reached the same way, as 'evaluated 2' or 'jumped over 3-15'."""
    runs = []
    for cycle, kind in trace:
        if runs and runs[-1][2] == kind and runs[-1][1] == cycle - 1:
            runs[-1][1] = cycle
        else:
            runs.append([cycle, cycle, kind])
    return [f"{kind} {first}" if first == last else f"{kind} {first}-{last}"
            for first, last, kind in runs]


def vector(v: np.ndarray) -> str:
    """A point as (x, y) with up to four decimals."""
    return "(" + ", ".join(tex_number(c) for c in v) + ")"


def analyse() -> dict:
    """Every number of the example."""
    A, b = square_line_problem()
    states = dykstra_cycles(START, A, b, 30)
    x_after_1, y_after_1, points_1 = states[1]
    state = stalling_state(A, b, x_after_1, y_after_1)
    xi = state["points"]

    # The stall-length theorem against plain Dykstra: the points of every cycle up to
    # t_switch - 1 repeat, and half-space m_switch is inactive in cycle t_switch
    t_switch = 1 + state["n_stall"]
    repeated_through = first_moved_cycle(states, np.array(xi[1:]), 1) - 1
    leaves = first_inactive_cycle(states, state["m_switch"], 1)

    projection = ConvexProjectionSolver._find_optimal_solution(START, A, b)
    active = np.array([m - 1 for m in state["active"]])
    beta_norm = float(np.sum(state["beta"][active] ** 2))
    distance = float(np.sum((START - projection) ** 2))
    phi = dual(START, state["unit_A"], state["unit_b"], y_after_1)
    bound_first = 1 + (distance - 2 * phi) / beta_norm
    bound_second = 1 + distance / beta_norm

    omegas = {}
    for omega in (1.1, 3.1, 9.1, 99.1):
        start = np.array([-1.0 - omega, 1.4])
        run = dykstra_cycles(start, A, b, int(5 * omega) + 5)
        theorem = stalling_state(A, b, run[1][0], run[1][1])
        omegas[omega] = {"n_stall": theorem["n_stall"], "y2": float(run[1][1][1]),
                         "leaves": first_inactive_cycle(run, 2, 1)}

    start_4 = np.array([-4.0, 1.4])
    run_4 = dykstra_cycles(start_4, A, b, 30)
    ratios_4 = [stalling_state(A, b, run_4[t][0], run_4[t][1])["ratios"].get(2)
                for t in (1, 2)]
    xi_4 = stalling_state(A, b, run_4[1][0], run_4[1][1])["points"]
    leaves_4 = first_inactive_cycle(run_4, 2, 1)
    moved_4 = first_moved_cycle(run_4, np.array(xi_4[1:]), 1)
    # The ratio 15 is an integer only in exact arithmetic, where the left side's test value
    # in cycle 16 is zero
    exact_4 = rational_dykstra_cycles(start_4, A, b, 20)
    exact_ratio_4 = exact_4[1][1][1] / Fraction(1, 5)
    exact_leaves_4 = first_inactive_cycle(exact_4, 2, 1)
    exact_moved_4 = first_moved_cycle(exact_4, exact_4[2][2], 2)

    # Bauschke's order visits the line first. its stall starts once a cycle ends stationary
    A_line, b_line = square_line_problem(line_first=True)
    run_b = dykstra_cycles(start_4, A_line, b_line, 30)
    for t in range(1, 30):
        bauschke = stalling_state(A_line, b_line, run_b[t][0], run_b[t][1])
        if bauschke["stationary"] and bauschke["ratios"]:
            break
    line_normal = LINE[0][0] / np.linalg.norm(LINE[0][0])
    line_offset = LINE[1][0] / np.linalg.norm(LINE[0][0])
    line_foot = start_4 - (line_normal @ start_4 - line_offset) * line_normal
    bauschke_count = 1 + math.floor(min(bauschke["ratios"].values()) + 1e-9)

    certified = LTIVer5Solver(START, A, b, 30)
    certified_result = certified.solve()
    traced = TracedSolver(START, A, b, 20)
    with kkt_test_disabled():
        traced.solve()

    return {
        "A": A, "b": b, "states": states, "state": state, "xi": xi, "t_switch": t_switch,
        "repeated_through": repeated_through, "leaves": leaves, "projection": projection,
        "bound_first": bound_first, "bound_second": bound_second, "omegas": omegas,
        "ratios_4": ratios_4, "leaves_4": leaves_4, "moved_4": moved_4,
        "exact_ratio_4": exact_ratio_4, "exact_leaves_4": exact_leaves_4,
        "exact_moved_4": exact_moved_4,
        "bauschke_cycle": t, "bauschke": bauschke, "bauschke_corner": run_b[t][0],
        "line_foot": line_foot, "bauschke_count": bauschke_count,
        "certified_at": certified_result.settled_at,
        "certificate": certified_result.certificate,
        "certified_point": certified_result.projection,
        "trace_without_kkt": compress_trace(traced.trace),
    }


def table(results: dict) -> str:
    """A two-column tabular of every number, in the order the paper states them."""
    s, xi = results["state"], results["xi"]
    beta = s["beta"]
    rows = [
        ("state after cycle 1, $x_1^{(2)}$", vector(results["states"][1][0])),
        ("active set $\\mathcal{C}^{(1)}$", "\\{" + ", ".join(map(str, s["active"])) + "\\}"),
        ("stationary", "yes" if s["stationary"] else "no"),
    ]
    rows += [(f"$\\xi_{m + 1}$", vector(point)) for m, point in enumerate(xi)]
    rows += [
        ("$\\beta_2$, $\\beta_3$, $\\beta_6$",
         f"{tex_number(beta[1])}, {tex_number(beta[2])}, {tex_number(beta[5])}"),
        ("$y_2^{(1)}$", tex_number(results["states"][1][1][1])),
        ("$N_\\mathrm{stall}$, $m_\\mathrm{switch}$, $t_\\mathrm{switch}$",
         f"{s['n_stall']}, {s['m_switch']}, {results['t_switch']}"),
        ("Dykstra: points repeat through cycle, left side inactive in cycle",
         f"{results['repeated_through']}, {results['leaves']}"),
        ("dual bounds on $N_\\mathrm{stall}$",
         f"{results['bound_first']:.2f}, {results['bound_second']:.2f}"),
    ]
    rows += [(f"$N_\\mathrm{{stall}}$ at $\\omega = {omega:g}$ (Dykstra: inactive in cycle)",
              f"{o['n_stall']} ({o['leaves']})") for omega, o in results["omegas"].items()]
    rows += [
        ("from $(-4, 1.4)$: ratio after cycles 1 and 2",
         ", ".join(tex_number(r) for r in results["ratios_4"])),
        ("from $(-4, 1.4)$: left side inactive in cycle, first cycle whose points differ",
         f"{results['leaves_4']}, {results['moved_4']}"),
        ("the same in exact arithmetic, with the ratio after cycle 1",
         f"{results['exact_leaves_4']}, {results['exact_moved_4']}, ratio "
         f"{results['exact_ratio_4']}"),
        ("line first: stationary after cycle, corner",
         f"{results['bauschke_cycle']}, {vector(results['bauschke_corner'])}"),
        ("line first: ratio, $N_\\mathrm{stall}$, $1 + \\lfloor \\text{ratio} \\rfloor$",
         f"{tex_number(min(results['bauschke']['ratios'].values()))}, "
         f"{results['bauschke']['n_stall']}, {results['bauschke_count']}"),
        ("projection of $(-4, 1.4)$ onto the line", vector(results["line_foot"])),
        ("accelerated solver: settles at cycle, point",
         f"{results['certified_at']} ({results['certificate']}), "
         f"{vector(results['certified_point'])}"),
        ("accelerated solver without the KKT test",
         "; ".join(results["trace_without_kkt"]).replace("-", "--")),
    ]
    body = "\n".join(f"  {name} & {value} \\\\" for name, value in rows)
    return ("\\begin{tabular}{l l}\n  \\hline\n  quantity & value \\\\\n  \\hline\n"
            f"{body}\n  \\hline\n\\end{{tabular}}\n")


def run() -> dict:
    """Print and write every number of the example."""
    quiet_accelerate_warnings()
    results = analyse()
    s = results["state"]
    print(f"state after cycle 1: {vector(results['states'][1][0])}, active set {s['active']}, "
          f"stationary {s['stationary']}")
    print("points xi: " + ", ".join(vector(p) for p in results["xi"]))
    print(f"beta = {np.round(s['beta'], 6).tolist()} (1/sqrt(5) = {1 / math.sqrt(5):.6f}), "
          f"y2 = {results['states'][1][1][1]:.4f}, ratios {s['ratios']}")
    print(f"N_stall = {s['n_stall']}, m_switch = {s['m_switch']}, t_switch = "
          f"{results['t_switch']}; Dykstra repeats its points through cycle "
          f"{results['repeated_through']} and the left side is inactive in cycle "
          f"{results['leaves']}")
    print(f"dual bounds: {results['bound_first']:.4f} and {results['bound_second']:.4f}")
    for omega, o in results["omegas"].items():
        print(f"omega {omega}: y2 = {o['y2']:.4f}, N_stall = {o['n_stall']}, Dykstra's left side "
              f"inactive in cycle {o['leaves']}")
    print(f"from (-4, 1.4): ratios after cycles 1 and 2 {results['ratios_4']}, left side "
          f"inactive in cycle {results['leaves_4']}, first cycle whose points differ "
          f"{results['moved_4']}")
    print(f"the same in exact arithmetic: ratio {results['exact_ratio_4']}, left side inactive "
          f"in cycle {results['exact_leaves_4']}, first cycle whose points differ "
          f"{results['exact_moved_4']}")
    print(f"line first: stationary after cycle {results['bauschke_cycle']} at "
          f"{vector(results['bauschke_corner'])}, ratios {results['bauschke']['ratios']}, "
          f"N_stall {results['bauschke']['n_stall']}, 1 + floor = {results['bauschke_count']}, "
          f"projection onto the line {vector(results['line_foot'])}")
    print(f"accelerated solver: settles at cycle {results['certified_at']} "
          f"({results['certificate']}) on {vector(results['certified_point'])}, quadprog "
          f"{vector(results['projection'])}")
    print("without the KKT test: " + ", ".join(results["trace_without_kkt"]))
    save_tex(table(results), "tab-square-line", "python -m dykstra.square_line_example")
    return results


def main() -> None:
    """Parse the command line and run."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--quick", action="store_true",
                        help="the same example written to a temporary directory")
    args = parser.parse_args()
    began = time.perf_counter()
    if args.quick:
        paper_figures.OUTPUT_DIR = tempfile.mkdtemp(prefix="square_line_example_")
    run()
    print(f"Finished in {time.perf_counter() - began:.1f} s: tab-square-line.tex in "
          f"{paper_figures.OUTPUT_DIR}")


if __name__ == "__main__":
    main()
