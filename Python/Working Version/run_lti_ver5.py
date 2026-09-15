"""Runs LTI Ver5 on the box-and-line example of main.py."""

import numpy as np
from lti_ver5 import LTIVer5Solver
from visualiser import VerticalVisualiser


def run() -> None:
    """Box-and-line example."""
    # Box: -1 <= x <= 1, -1 <= y <= 1
    A_box = np.array([[1., 0.], [-1., 0.], [0., 1.], [0., -1.]])
    b_box = np.array([1., 1., 1., 1.])
    # Line x/2 + y = 1 as two opposite half-spaces
    A_line = np.array([[0.5, 1.], [-0.5, -1.]])
    b_line = np.array([1., -1.])

    z = np.array([-2., 1.4])
    x_range = [-2.05, 0.5]
    y_range = [0.8, 1.5]
    max_iter: int = 30

    A: np.ndarray = np.vstack([A_box, A_line])
    b: np.ndarray = np.hstack([b_box, b_line])

    solver = LTIVer5Solver(
        z, A, b, max_iter,
        track_error=True,
        plot_active_halfspaces=True
    )
    result = solver.solve()

    actual_projection = solver.actual_projection
    distance = actual_projection - result.projection
    print(f"\nThe finite time projection over {max_iter} iteration(s) is: "
          f"{result.projection}")
    print(f"The distance to the optimal solution is: {distance}")
    print(f"The squared-error is {np.dot(distance, distance)}\n")

    ab_pairs = [("Box", "Greys", A_box, b_box), ("Line", "Greys", A_line, b_line)]
    visualiser = VerticalVisualiser(result, ab_pairs, max_iter, x_range, y_range,
                                    solver.__class__.__name__)
    visualiser.visualise(plot_original_point=z, plot_optimal_point=actual_projection)


if __name__ == "__main__":
    run()
