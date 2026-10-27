"""Seeded stress battery for the LTI solvers, the KKT certificate, the oracle, and the C
solvers against the Python, on random bounded polyhedra in dimensions 2 to 64 and on
degenerate arrangements. Nothing here imports pyplot, so it runs without a display."""

from functools import lru_cache
from pathlib import Path
import sys
import time
import unittest

import numpy as np


WORKING_VERSION = Path(__file__).resolve().parents[1] / "Python" / "Working Version"
sys.path.insert(0, str(WORKING_VERSION))

from c_solvers import accelerated_c_projection, dykstra_c_projection, libraries_built
from convex_projection_solver import ConvexProjectionSolver, DykstraProjectionSolver
from lti_examples import box_line_problem
from lti_numerics import kkt_certificate, nonnegative_quadratic, unit_constraints
from lti_solver import PRESETS, LTISolver
from oracle import oracle_lti_projection, record_schedule


# Seeded random problems as (rows, dimension, seed), spanning dimensions 2 to 64 and 3 to
# 200 half-spaces
RANDOM_SIZES = ((3, 2, 0), (8, 2, 1), (12, 3, 2), (25, 5, 3), (40, 8, 4), (60, 16, 5),
                (100, 32, 6), (200, 64, 7))

# The KKT certificate's residual allowance, relative to the step from z
KKT_ACCURACY = 1e-9


def bounded_polyhedron(rng: np.random.Generator, rows: int, dimension: int) -> tuple:
    """Random bounded, nonempty polyhedron and a start outside it."""
    # Generator 2 of the paper's supplement: a positively spanning set of normals, the
    # axes and their negated sum turned by a random rotation, bounds the region, the
    # other rows have entries tan(theta) of random sign, and offsets taken from an
    # interior anchor keep it nonempty
    basis, upper = np.linalg.qr(rng.standard_normal((dimension, dimension)))
    spanning = np.vstack([np.eye(dimension), -np.ones((1, dimension))])
    A = spanning @ (basis * np.sign(np.diag(upper))).T
    extra = rows - (dimension + 1)
    if extra > 0:
        angles = rng.uniform(0.0, np.pi / 2, size=(extra, dimension))
        A = np.vstack([A, rng.choice([-1.0, 1.0], size=(extra, dimension)) * np.tan(angles)])
    b = A @ rng.uniform(-2.0, 2.0, size=dimension) + rng.uniform(0.2, 1.5, size=len(A))
    scale = 4.0
    while True:
        z = rng.standard_normal(dimension) * scale
        if np.any(A @ z > b):
            return z, A, b
        scale *= 1.5


@lru_cache(maxsize=None)
def random_problem(rows: int, dimension: int, seed: int) -> tuple:
    """One seeded problem of the battery."""
    return bounded_polyhedron(np.random.default_rng(1000 * rows + dimension + 7919 * seed), rows, dimension)


@lru_cache(maxsize=None)
def dykstra_reference(key: tuple, max_iter: int):
    """Dykstra's run on a battery problem, with its corrections, and quadprog's projection."""
    z, A, b = random_problem(*key)
    solver = DykstraProjectionSolver(z, A, b, max_iter=max_iter, plot_errors=True)
    return solver.solve(), solver.actual_projection


def degenerate_problems() -> dict:
    """Arrangements with dependent or nearly dependent active normals, and redundancy."""
    def unit(degrees: float) -> np.ndarray:
        return np.array([np.cos(np.radians(degrees)), np.sin(np.radians(degrees))])

    a, c = unit(70.0), unit(160.0)
    vertex = np.array([0.4, -0.2])
    fan = np.array([unit(80.0), unit(90.0), unit(100.0)])
    box_A, box_b = np.vstack([np.eye(3), -np.eye(3)]), np.ones(6)
    diagonal = np.ones(3) / np.sqrt(3.0)
    problems = {
        "duplicate rows": (np.array([2.0, 3.0]), np.array([a, a, c]), np.array([1.0, 1.0, 0.5])),
        "rows scaled copies": (np.array([2.0, 3.0]), np.array([a, 3.0 * a, c]), np.array([1.0, 3.0, 0.5])),
        "looser parallel row": (np.array([2.0, 3.0]), np.array([a, a, c]), np.array([1.0, 2.0, 0.5])),
        "three lines through a vertex": (vertex + np.array([0.1, 2.0]), fan, fan @ vertex),
        "three lines through a vertex, z beside": (vertex + np.array([3.0, 1.0]), fan, fan @ vertex),
        "three lines through a vertex, z far": (vertex + 1e3 * np.array([0.05, 1.0]), fan, fan @ vertex),
        "box vertex with a fourth plane": (np.array([3.0, 2.5, 2.0]), np.vstack([box_A, diagonal]),
                                           np.append(box_b, np.sqrt(3.0))),
        "box vertex with the fourth plane first": (np.array([3.0, 2.5, 2.0]), np.vstack([diagonal, box_A]),
                                                   np.append(np.sqrt(3.0), box_b)),
        "box edge with a fourth plane": (np.array([3.0, 3.0, 0.2]), np.vstack([box_A, diagonal]),
                                         np.append(box_b, np.sqrt(3.0))),
        "implied row": (np.array([3.0, 3.0]), np.array([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]]),
                        np.array([1.0, 1.0, 5.0])),
        "row containing the start binds": (np.array([-2.0, 1.4]),
                                           np.array([[-1.0, 0.0], [0.0, 1.0], [-0.5, -1.0]]),
                                           np.array([1.0, 1.0, -1.0])),
    }
    # Rows turned by theta about a point both boundaries pass through, with z beyond it
    for theta in (1e-8, 1e-10, 1e-12):
        turned = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]]) @ a
        point = np.array([0.3, 0.7])
        problems[f"rows {theta:g} rad apart"] = (point + 3.0 * (a + turned), np.array([a, turned, c]),
                                                 np.array([a @ point, turned @ point, c @ point + 0.5]))
    # Five planes through one vertex of R^3
    normals = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1], [1, 1, 0], [0, 1, 1]], dtype=float)
    normals /= np.linalg.norm(normals, axis=1)[:, None]
    apex = np.array([0.5, -0.5, 1.0])
    problems["five planes through a vertex"] = (apex + np.array([2.0, 1.5, 1.0]), normals, normals @ apex)
    return problems


def small_systems(count: int, seed: int):
    """Seeded small systems with structure: half-integer rows, zero coordinates, and
    vertices where more rows are tight than the dimension."""
    # Each yields z, the unit rows and offsets, and quadprog's projection. quadprog
    # rejects as inconsistent some feasible systems whose region is a single point, so
    # those are left out
    rng = np.random.default_rng(seed)
    produced = 0
    while produced < count:
        dimension = int(rng.integers(2, 5))
        rows = np.round(rng.normal(size=(int(rng.integers(dimension + 1, dimension + 5)), dimension)) * 2) / 2
        if np.any(np.all(rows == 0.0, axis=1)):
            continue
        anchor = np.round(rng.normal(size=dimension) * 2) / 2
        b = rows @ anchor + np.round(rng.uniform(0.0, 1.0, size=len(rows)) * 4) / 4
        z = np.round((anchor + 3.0 * rng.normal(size=dimension)) * 2) / 2
        unit_A, unit_b = unit_constraints(rows, b)
        try:
            projection = ConvexProjectionSolver._find_optimal_solution(z, unit_A, unit_b)
        except ValueError:
            continue
        produced += 1
        yield z, unit_A, unit_b, projection


def three_plane_arrangement() -> tuple:
    """The paper's three-plane arrangement, as paper_figures.three_plane_arrangement builds it."""
    # Boundary 1 is the line x = 1, boundaries 2 and 3 pass through (1, 0.85) and
    # (1, 0.35) with slopes -0.16 and 0.585, and the region lies to their left
    a_2 = np.array([0.16, 1.0]) / np.hypot(0.16, 1.0)
    a_3 = np.array([0.585, -1.0]) / np.hypot(0.585, 1.0)
    A = np.array([[1.0, 0.0], a_2, a_3])
    return A, np.array([1.0, a_2 @ np.array([1.0, 0.85]), a_3 @ np.array([1.0, 0.35])])


def performance_problem() -> tuple:
    """The 96 by 32 problem of the paper's performance figure."""
    # paper_figures.performance_error_figure draws it from the legacy generator, seeded
    # with 100000 + 1000 * 96 + 3, through bounded_problem and infeasible_start. The
    # draws are repeated here in the same order so that this file never imports pyplot
    planes, dimension = 96, 32
    np.random.seed(100000 + 1000 * planes + 3)
    spanning = np.vstack([np.eye(dimension), -np.ones((1, dimension))])
    basis, upper = np.linalg.qr(np.random.standard_normal((dimension, dimension)))
    A = spanning @ (basis * np.sign(np.diag(upper))).T
    angles = np.random.uniform(0.0, np.pi / 2, size=(planes - dimension - 1, dimension))
    signs = np.random.choice([-1, 1], size=(planes - dimension - 1, dimension))
    A = np.vstack([A, signs * np.tan(angles)])
    b = A @ np.random.uniform(-2.0, 2.0, size=dimension) + np.random.uniform(0.2, 1.5, size=planes)
    scale = 4.0
    while True:
        z = np.random.standard_normal(dimension) * scale
        if np.any(A @ z > b):
            return z, A, b
        scale *= 1.5


def spy_exact_cycles(solver) -> list:
    """Cycles the solver runs as exact Dykstra cycles."""
    cycles = []
    run_exact_cycle = solver._exact_cycle
    solver._exact_cycle = lambda cycle: (cycles.append(cycle), run_exact_cycle(cycle))[1]
    return cycles


class StressTests(unittest.TestCase):
    def setUp(self) -> None:
        # The battery forms products of matrices up to 64 wide itself, whose spurious
        # flags from Apple's Accelerate lti_numerics.ignore_blas_flags describes
        state = np.errstate(divide="ignore", over="ignore", invalid="ignore")
        state.__enter__()
        self.addCleanup(state.__exit__, None, None, None)

    def assert_certified(self, z: np.ndarray, A: np.ndarray, b: np.ndarray, x: np.ndarray,
                         multipliers: np.ndarray, projection: np.ndarray, label: str) -> None:
        """Checks a point the KKT test certified, and its multipliers on the unit rows."""
        unit_A, unit_b = unit_constraints(A, b)
        step = z - x
        scale = max(float(np.abs(z).max()), float(np.abs(step).max()))
        # The allowance the certificate grants the stationarity residual, which bounds the
        # distance to the projection
        eps = np.finfo(float).eps
        allowance = float(np.linalg.norm(KKT_ACCURACY * np.abs(step) + 64 * eps * (
            np.abs(z) + np.abs(x) + np.abs(unit_A).T @ multipliers)))
        slacks = unit_A @ x - unit_b
        self.assertTrue(np.all(multipliers >= 0.0), label)
        self.assertLessEqual(float(slacks.max()), 1e-12 * scale, label)
        if np.any(multipliers > 0.0):
            self.assertLessEqual(float(np.abs(slacks[multipliers > 0.0]).max()), 1e-12 * scale, label)
        self.assertLessEqual(float(np.linalg.norm(step - unit_A.T @ multipliers)), allowance, label)
        self.assertLessEqual(float(np.linalg.norm(x - projection)), allowance, label)

    def assert_on_contract(self, solver: LTISolver, result, dykstra, projection: np.ndarray,
                           budget: int, label: str) -> None:
        """Checks a result is Dykstra's iterate and history at the budget or, once settled, the limit."""
        z, A, b = solver.z, solver.A, solver.b
        scale = float(np.abs(z).max())
        end = result.settled_at if result.is_settled() else budget + 1
        np.testing.assert_allclose(result.path[:end], dykstra.path[:end], rtol=0.0, atol=1e-9 * scale,
                                   err_msg=label)
        np.testing.assert_allclose(result.errors_for_plotting[:end - 1], dykstra.errors_for_plotting[:end - 1],
                                   rtol=0.0, atol=1e-9 * scale, err_msg=label)
        if result.certificate == "kkt":
            self.assert_certified(z, A, b, result.projection, solver.y, projection, label)
        elif result.certificate == "finality":
            # A limit proven only final carries the cycle map's rounding, up to about
            # cond(I - A_m) * eps with the condition number below its cutoff of 1e8
            np.testing.assert_allclose(result.projection, projection, rtol=0.0, atol=1e-7 * scale,
                                       err_msg=label)
        else:
            self.assertIsNone(result.settled_at, label)
            np.testing.assert_allclose(result.projection, dykstra.path[budget, -1], rtol=0.0,
                                       atol=1e-9 * scale, err_msg=label)

    def test_every_preset_follows_dykstra_or_certifies_on_random_polyhedra(self) -> None:
        for key in RANDOM_SIZES:
            budgets = (3, 300) if key[1] >= 32 else (3, 40, 300)
            dykstra, projection = dykstra_reference(key, max(budgets))
            z, A, b = random_problem(*key)
            for preset in PRESETS:
                for budget in budgets:
                    label = f"{key[0]} rows in {key[1]} dimensions, {preset}, budget {budget}"
                    with self.subTest(label):
                        solver = LTISolver(z, A, b, max_iter=budget, preset=preset, plot_errors=True)
                        result = solver.solve()
                        self.assert_on_contract(solver, result, dykstra, projection, budget, label)

    def test_degenerate_arrangements_follow_dykstra_or_certify_promptly(self) -> None:
        # Dependent active normals leave the multipliers of the projection undetermined
        # and nearly dependent ones amplify rounding through them, which the certificate's
        # accuracy allows for. It must still never certify a point beyond its allowance,
        # and its nonnegative least squares must not stall
        for case_name, (z, A, b) in degenerate_problems().items():
            dykstra = DykstraProjectionSolver(z, A, b, max_iter=400, plot_errors=True).solve()
            for preset in PRESETS:
                for budget in (1, 2, 5, 40, 400):
                    label = f"{case_name}, {preset}, budget {budget}"
                    with self.subTest(label):
                        started = time.perf_counter()
                        solver = LTISolver(z, A, b, max_iter=budget, preset=preset, plot_errors=True)
                        result = solver.solve()
                        self.assertLess(time.perf_counter() - started, 5.0, label)
                        self.assert_on_contract(solver, result, dykstra, solver.actual_projection, budget, label)

    def test_the_projection_dual_reaches_its_optimum_on_small_systems(self) -> None:
        # The certificate's first candidate support comes from the dual of projecting z
        # onto the candidate rows, whose Gram matrix is singular whenever there are more
        # rows than dimensions, and its multipliers must give the projection itself
        for index, (z, N, b, projection) in enumerate(small_systems(3000, 20261023)):
            with self.subTest(system=index):
                multipliers = nonnegative_quadratic(N @ N.T, N @ z - b)
                self.assertTrue(np.all(multipliers >= 0.0))
                np.testing.assert_allclose(z - N.T @ multipliers, projection, rtol=0.0,
                                           atol=1e-9 * max(1.0, float(np.abs(z).max())))

    def test_the_certificate_accepts_the_projection_given_its_support(self) -> None:
        # With the rows tight at the projection as its active set and the projection as
        # its hint, the KKT test has the true support among its candidates, and its
        # rounding floors must let it pass, also where a tight row touches only
        # coordinates the step does not move
        for index, (z, N, b, projection) in enumerate(small_systems(6000, 20261024)):
            with self.subTest(system=index):
                scale = max(1.0, float(np.abs(z).max()))
                tight = np.abs(N @ projection - b) <= 1e-9 * scale
                certified = kkt_certificate(z, N, b, projection, tight)
                self.assertIsNotNone(certified)
                bound = 1e-9 * float(np.linalg.norm(z - projection)) + 1e-13 * scale
                self.assertLessEqual(float(np.linalg.norm(certified[0] - projection)), bound)


if __name__ == "__main__":
    unittest.main()
