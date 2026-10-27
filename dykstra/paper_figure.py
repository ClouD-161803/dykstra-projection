"""The comparison figure of the earlier report, plain Dykstra against the stall detection solver
on a box with a line through it. Run from the repository root with
`python -m dykstra.paper_figure`. It writes both results as CSV files to results/ at the
repository root and opens the figure in a window.
"""

import os

import numpy as np
from dykstra import REPOSITORY_ROOT
from dykstra.convex_projection_solver import (DykstraProjectionSolver as Dykstra,
                                              DykstraStallDetectionSolver as FastForward)
from dykstra.visualiser import ResultExporter, ComparisonVisualiser


OUTPUT_DIR = os.path.join(REPOSITORY_ROOT, "results")


def run() -> None:
    A_box = np.array([
        [1., 0.],
        [-1., 0.],
        [0., 1.],
        [0., -1.]
    ])
    b_box = np.array([1., 1., 1., 1.])
    
    corner_count = 1
    
    A_line = np.array([[0.5, 1.], [-0.5, -1.]])
    b_line = np.array([1., -1.])
    
    z = np.array([-2., 1.4])
    x_range = [-2.05, 0.5]
    y_range = [0.8, 1.5]
    
    max_iter: int = 30
    plot_activity: bool = True
    plot_quivers: bool = False
    
    A: np.ndarray = np.vstack([A_box, A_line])
    b: np.ndarray = np.hstack([b_box, b_line])
    
    solver1 = Dykstra(
        z, A, b, max_iter,
        track_error=True,
        plot_errors=plot_quivers,
        plot_active_halfspaces=plot_activity
    )
    
    solver2 = FastForward(
        z, A, b, max_iter,
        track_error=True,
        plot_errors=plot_quivers,
        plot_active_halfspaces=plot_activity
    )
    
    result1 = solver1.solve()
    result2 = solver2.solve()
    
    solver1_name = "Dykstra"
    solver2_name = "Algorithm (1)"
    
    ResultExporter.export(
        result=result1,
        output_path=OUTPUT_DIR,
        solver_name=solver1_name,
        initial_point=z,
        A=A,
        b=b,
        max_iter=max_iter,
        track_error=True,
        plot_errors=plot_quivers,
        plot_active_halfspaces=plot_activity
    )
    
    ResultExporter.export(
        result=result2,
        output_path=OUTPUT_DIR,
        solver_name=solver2_name,
        initial_point=z,
        A=A,
        b=b,
        max_iter=max_iter,
        track_error=True,
        plot_errors=plot_quivers,
        plot_active_halfspaces=plot_activity
    )
    
    print(f"\nSolver 1 ({solver1_name}) projection: {result1.projection}")
    print(f"Solver 2 ({solver2_name}) projection: {result2.projection}")
    
    ab_pairs1 = [
        ("Box", "Greys", A_box, b_box),
        ("Line", "Greys", A_line, b_line)
    ]
    ab_pairs2 = [
        ("Box", "Greys", A_box, b_box),
        ("Line", "Greys", A_line, b_line)
    ]
    
    visualiser = ComparisonVisualiser(
        result1, result2, ab_pairs1, ab_pairs2,
        max_iter, x_range, y_range,
        solver1_name, solver2_name,
        z,
        display_result_index=1,
        display_halfspace_index=0
    )
    visualiser.visualise()


if __name__ == "__main__":
    run()
