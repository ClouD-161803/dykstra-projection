"""Oracle experiment: records the sequence of active sets of one Dykstra run, then
replays it with every activity change known in advance, entering each episode with
one exact cycle and jumping the rest of the episode in closed form."""

import numpy as np
from convex_projection_solver import ConvexProjectionSolver
from lti_numerics import RESOLVENT_COND_CAP, active_auxiliaries, advance, closed_form, cycle_map, exact_cycle
from projection_result import ProjectionResult


def _unit_problem(z: np.ndarray, A: np.ndarray, b: np.ndarray, max_iter: int = 0) -> tuple:
    """Validated problem and unit half-spaces."""
    z, A, b, max_iter = ConvexProjectionSolver._validate_problem(z, A, b, max_iter, None)
    unit_A, unit_b = np.empty_like(A), np.empty_like(b)
    for index, (row, offset) in enumerate(zip(A, b)):
        unit_A[index], unit_b[index] = ConvexProjectionSolver._normalise(row, offset)
    return z, unit_A, unit_b, max_iter


def record_schedule(z: np.ndarray, A: np.ndarray, b: np.ndarray, max_iter: int) -> tuple:
    """Record an active-set schedule for the constraints ``A @ x <= b``."""
    z, unit_A, unit_b, max_iter = _unit_problem(z, A, b, max_iter)
    x, y = z, np.zeros(len(unit_b))
    schedule = []
    for _ in range(max_iter):
        # Extend the current episode while the active set repeats
        x_next, y_next, _ = exact_cycle(x, y, unit_A, unit_b)
        active = tuple(y_next > 0.0)
        if schedule and schedule[-1][0] == active:
            schedule[-1] = (active, schedule[-1][1] + 1)
        else:
            schedule.append((active, 1))

        # Stop at an exact fixed point of the iteration
        stop = np.array_equal(x_next, x) and np.array_equal(y_next, y)
        x, y = x_next, y_next
        if stop:
            break
    return schedule, ProjectionResult(projection=x.copy())


def oracle_lti_projection(z: np.ndarray, A: np.ndarray, b: np.ndarray,
                          schedule: list) -> ProjectionResult:
    """Replay with the schedule known."""
    z, unit_A, unit_b, _ = _unit_problem(z, A, b)
    x, y = z, np.zeros(len(unit_b))
    for active, k in schedule:
        # One exact cycle enters the episode
        x, y, _ = exact_cycle(x, y, unit_A, unit_b)
        if k < 2:
            continue
        cmap = cycle_map(unit_A, unit_b, active)
        IA = np.eye(len(x)) - cmap.A_m
        if np.linalg.cond(IA) < RESOLVENT_COND_CAP:
            # Jump the remaining k - 1 cycles in closed form
            cf = closed_form(cmap, x, y, IA)
            z_end = np.linalg.matrix_power(cmap.A_m, k - 1) @ (x - cf.x_inf)
            x, y = cf.x_inf + z_end, np.where(cf.active, active_auxiliaries(cf, k - 1, z_end), 0.0)
        else:
            # Step the remaining cycles through the map
            for _ in range(k - 1):
                y = np.where(cmap.active, y + cmap.R @ x + cmap.s, 0.0)
                x = advance(cmap, x)
    return ProjectionResult(projection=x.copy())
