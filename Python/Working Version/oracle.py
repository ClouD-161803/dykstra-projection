"""Oracle experiment: records the sequence of active sets of one Dykstra run, then
replays it with every activity change known in advance, entering each episode
with one exact cycle and jumping the rest of the episode in closed form."""

import numpy as np
from lti_ver2 import _RESOLVENT_COND_CAP
from lti_ver5 import LTIVer5Solver
from projection_result import ProjectionResult


def record_schedule(z: np.ndarray, A: np.ndarray, b: np.ndarray, max_iter: int) -> tuple:
    """Record an active-set schedule for the constraints ``A @ x <= b``."""
    solver = LTIVer5Solver(z, A, b, max_iter, dimensions=len(z))
    solver._prepare()
    schedule = []
    x_prev, e_prev = solver.x.copy(), [e_m.copy() for e_m in solver.e]
    for i in range(max_iter):
        # Extend the current episode while the active set repeats
        active = solver._dykstra_cycle(i + 1)
        if schedule and schedule[-1][0] == active:
            schedule[-1] = (active, schedule[-1][1] + 1)
        else:
            schedule.append((active, 1))

        # Stop at an exact fixed point of the iteration
        if np.array_equal(solver.x, x_prev) and all(
                np.array_equal(u, v) for u, v in zip(solver.e, e_prev)):
            break
        x_prev, e_prev = solver.x.copy(), [e_m.copy() for e_m in solver.e]
    return schedule, ProjectionResult(projection=solver.x.copy())


def oracle_lti_projection(z: np.ndarray, A: np.ndarray, b: np.ndarray,
                          schedule: list) -> ProjectionResult:
    """Replay with the schedule known."""
    budget = max(sum(k for _, k in schedule), 1)
    solver = LTIVer5Solver(z, A, b, budget, dimensions=len(z))
    solver._prepare()
    p = len(solver.x)
    cycle = 0
    for active, k in schedule:
        # One exact cycle enters the episode
        cycle += 1
        solver._dykstra_cycle(cycle)
        if k > 1:
            solver._build_cycle_map(active)
            IA = np.eye(p) - solver.A_m
            active = np.array(active)
            if np.linalg.cond(IA) < _RESOLVENT_COND_CAP:
                # Jump the remaining k - 1 cycles in closed form
                x_inf = np.linalg.solve(IA, solver.B_m)
                RIA = np.linalg.solve(IA.T, solver.R.T).T
                z_0 = solver.x - x_inf
                G = solver.y + RIA @ z_0
                beta = solver.R @ x_inf + solver.s
                z_end = np.linalg.matrix_power(solver.A_m, k - 1) @ z_0
                solver._set_state(x_inf + z_end, G + (k - 1) * beta - RIA @ z_end, active)
            else:
                # Step the remaining cycles through the map
                for _ in range(k - 1):
                    delta = solver.R @ solver.x + solver.s
                    solver.y = np.where(active, solver.y + delta, 0.0)
                    solver._advance_cycle()
                solver._sync_e_from_y()
            cycle += k - 1
    return ProjectionResult(projection=solver.x.copy())
