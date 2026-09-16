"""
Main script for running Dykstra's projection algorithm.
"""

import numpy as np
from convex_projection_solver import (DykstraProjectionSolver,
                                       DykstraMapHybridSolver,
                                       DykstraStallDetectionSolver)
from visualiser import Visualiser, VerticalVisualiser


def run() -> None:
    """Tests Dykstra's algorithm on the intersection of a box at the origin
    and a line passing through (2, 0) and (0, 1)"""
    # --- Define Problem ---

    # # * Without rounding
    # Define the box constraints (half-spaces) (make sure these are floats)
    N_box = np.array([
        [1., 0.],  # Right side of the box: x <= 1
        [-1., 0.], # Left side of the box: x >= -1
        [0., 1.],  # Top side of the box: y <= 1
        [0., -1.]  # Bottom side of the box: y >= -1
    ])
    b_box = np.array([1., 1., 1., 1.])

    # # * With Rounding (uncomment to use)
    # from edge_rounder import rounded_box_constraints
    # center = (0, 0)
    # width = 2
    # height = 2
    # corner_count = 3
    # N_box, c_box = rounded_box_constraints(center, width, height, corner_count)

    corner_count = 1

    # Define the line constraints
    # The line equation is y = 1 - x/2
    # Rearranging to get it in the form N*x <= b:
    # x/2 + y <= 1 & x/2 + y >= 1
    N_line = np.array([[1/2, 1], [-1/2, -1]])
    b_line = np.array([1, -1])

    # Point to project and x-y range (uncomment wanted example)

    # * Simple top left - stalling - y y y
    z = np.array([-2., 1.4])
    x_range = [-2.05, 0.5]
    y_range = [0.8, 1.5]

    # # * Simple top left - no stalling - y y y
    # z = np.array([-0.75, 1.2])
    # x_range = [-1.5, 0.75]
    # y_range = [0.7, 1.4]

    # # * Intersection - no stalling - y y n
    # z = np.array([0.5, 1.75])
    # x_range = [-2., 2.]
    # y_range = [0., 2.]

    # # * Very far to the top left - y n y
    # z = np.array([-10, 5.])
    # x_range = [-10, 0.5]
    # y_range = [0.5, 6]

    # # * Very far to bottom left - n y y
    # z = np.array([-5, -5])
    # x_range = [-6, 0.5]
    # y_range = [-6, 4.]

    # # * Very far to the top right y y n
    # z = np.array([3.5, 3.5])
    # x_range = [-1, 4.]
    # y_range = [-1., 4.]

    # # * Very far to the bottom right - y n y
    # z = np.array([10, -5])
    # x_range = [0, 11.]
    # y_range = [-6, 1.]

    # --- Configuration ---
    
    max_iter: int = 10
    plot_activity: bool = True
    plot_quivers: bool = True
    
    # Combine constraints
    # Project onto box, then line
    A: np.ndarray = np.vstack([N_box, N_line])
    b: np.ndarray = np.hstack([b_box, b_line])

    # # Project onto line, then box
    # A: np.ndarray = np.vstack([N_line, N_box])
    # b: np.ndarray = np.hstack([b_line, b_box])

    # --- Solver Selection ---
    
    # * Standard Dykstra's Algorithm
    solver = DykstraProjectionSolver(
        z, A, b, max_iter,
        track_error=True,
        plot_errors=plot_quivers,
        plot_active_halfspaces=plot_activity
    )
    
    # # * Hybrid MAP-Dykstra Algorithm
    # solver = DykstraMapHybridSolver(
    #     z, A, b, max_iter,
    #     track_error=True,
    #     plot_errors=plot_quivers,
    #     plot_active_halfspaces=plot_activity
    # )

    # # * Dykstra with Stalling Detection
    # solver = DykstraStallDetectionSolver(
    #     z, A, b, max_iter,
    #     track_error=True,
    #     plot_errors=plot_quivers,
    #     plot_active_halfspaces=plot_activity
    # )
    

    solver_name = solver.__class__.__name__
    
    # --- Run Solver ---
    
    result = solver.solve()
    
    actual_projection = solver.actual_projection
    distance = actual_projection - result.projection

    print(f"\nThe finite time projection over {max_iter} iteration(s) is: "
          f"{result.projection}")
    print(f"The distance to the optimal solution is: {distance}")
    print(f"The squared-error is {np.dot(distance, distance)}\n")

    Nc_pairs = [
        (f"'Box'\n(rounded by {corner_count} corner(s))" if corner_count > 1 else "Box", 
         "Greys", N_box, b_box),
        ("Line", "Greys", N_line, b_line)
    ]

    # # * Vertical Layout
    visualiser = VerticalVisualiser(result, Nc_pairs, max_iter, x_range, y_range, solver_name)
    # # * Horizontal Layout
    # visualiser = Visualiser(result, Nc_pairs, max_iter, x_range, y_range, solver_name)

    visualiser.visualise(plot_original_point=z, plot_optimal_point=actual_projection)


if __name__ == "__main__":
    run()