"""Shared setup and runner utilities for the LTI solver experiments."""

from __future__ import annotations

import numpy as np

from convex_projection_solver import ConvexProjectionSolver
from visualiser import VerticalVisualiser


HalfSpaceGroup = tuple[str, str, np.ndarray, np.ndarray]


def box_line_problem() -> tuple[
    np.ndarray, np.ndarray, np.ndarray, list[HalfSpaceGroup]
]:
    """Return the box-and-line projection problem used by the LTI demos."""
    A_box = np.array([
        [1.0, 0.0],
        [-1.0, 0.0],
        [0.0, 1.0],
        [0.0, -1.0],
    ])
    b_box = np.ones(4)
    A_line = np.array([[0.5, 1.0], [-0.5, -1.0]])
    b_line = np.array([1.0, -1.0])

    z = np.array([-2.0, 1.4])
    A = np.vstack([A_box, A_line])
    b = np.hstack([b_box, b_line])
    ab_pairs = [
        ("Box", "Greys", A_box, b_box),
        ("Line", "Greys", A_line, b_line),
    ]
    return z, A, b, ab_pairs


def run_lti_example(solver_type: type[ConvexProjectionSolver], *, max_iter: int = 30) -> None:
    """Run an LTI solver on the shared 2-D example and display its result."""
    z, A, b, ab_pairs = box_line_problem()
    x_range = [-2.05, 0.5]
    y_range = [0.8, 1.5]

    solver = solver_type(
        z,
        A,
        b,
        max_iter,
        track_error=True,
        plot_active_halfspaces=True,
    )
    result = solver.solve()

    distance = solver.actual_projection - result.projection
    print(
        f"\nThe finite time projection over {max_iter} iteration(s) is: "
        f"{result.projection}"
    )
    print(f"The distance to the optimal solution is: {distance}")
    print(f"The squared-error is {np.dot(distance, distance)}")
    if result.is_settled():
        print(f"Settled from cycle {result.settled_at} by the {result.certificate} certificate")
    print()

    visualiser = VerticalVisualiser(
        result,
        ab_pairs,
        max_iter,
        x_range,
        y_range,
        solver.__class__.__name__,
    )
    visualiser.visualise(
        plot_original_point=z,
        plot_optimal_point=solver.actual_projection,
    )
