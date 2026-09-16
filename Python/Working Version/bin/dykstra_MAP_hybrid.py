"""
This module implements a hybrid of MAP and Dykstra's algorithm for projecting
a point onto the intersection of convex sets (specifically, half-spaces).

Functions:
- dykstra_projection(z, A, b, max_iter, track_error=False, min_error=1e-3,
                dimensions=2, plot_errors=False plot_active_halfspaces=False):
Projects a point 'z' onto the intersection of multiple half-spaces
defined by the matrix A and vector b using a hybrid version of dykstra and MAP.

Additional Features:
- Error tracking: Option to track and plot errors at each iteration.
- Convergence and stalling detection.
- Beta parameter for selecting between MAP and Dykstra's method.
- Generalised for any number of dimensions.
"""


import numpy as np
from dykstra_functions import (is_in_half_space,
                               project_onto_half_space,
                               delete_inactive_half_spaces,
                               find_optimal_solution,
                               beta_check)


def dykstra_projection(z: np.ndarray, A: np.ndarray, b: np.ndarray,
                       max_iter: int, track_error: bool=False,
                       min_error: float=1e-3, dimensions: int=2,
                       plot_errors: bool=False,
                       plot_active_halfspaces: bool=False,
                       delete_spaces: bool=False) -> tuple:
    """
    Projects a point 'z' onto the intersection of convex sets (half spaces)
    via a modified version of dykstra's algorithm: if the current point to be
    projected lies outside the feasible region, we use MAP; otherwise, we
    use dykstra's method.

    Args:
        z: Initial point.
        A: Matrix of normal vectors.
        b: Vector of constant offsets.
        max_iter: Maximum number of iterations.
        track_error (bool, optional): Track the squared error at each iteration.
        min_error (float, optional): Minimum error threshold for convergence.
        dimensions (int, optional): Number of dimensions.
        plot_errors (bool, optional): Plot errors at each iteration.
        plot_active_halfspaces (optional): Whether to plot active half spaces.
        delete_spaces (optional): Whether to delete inactive halfspaces

    Returns:
        tuple: Final projected point, path taken, error metrics (if tracking),
                and errors for plotting (if selected).
    """

    # Eliminate inactive halfspaces (V9)
    if delete_spaces:
        A, b = delete_inactive_half_spaces(z, A, b)

    # Initialise variables
    # print(f"\nThere are {A.shape[0]} halfspaces or A is {A}") # for debug
    n = A.shape[0]  # Number of half-spaces
    x = z.copy()  # create a deep copy of the original point
    errors = np.zeros_like(z) # individual error vectors
    e_dykstra = [errors] * n  # list of a number of error vectors equal to n
    e_MAP = [errors] * n # same but for MAP (V6)

    # Vector for storing all errors (V8)
    # if plot_errors:
    # errors_for_plotting = np.array([np.zeros_like(e_dykstra)
    #                                 for _ in range(max_iter)])
    # print(f"Errors for plotting {errors_for_plotting}") for debugging

    # BUGFIX: Previous version
    errors_for_plotting = [e_MAP.copy()] # initialise with all zeros
    path = [z.copy()]  # Initialize the path with the original point
    

    # Matrix of successive projections
    x_historical = np.array([[np.zeros_like(z) for _ in range(n)] 
                             for _ in range(max_iter)])

    # Active halfspaces vector (V9)
    # if plot_active_halfspaces:
    active_half_spaces = np.array([[np.zeros_like(n) for _ in range(max_iter)]
                            for _ in range(n)])

    # Optimal solution (V4)
    actual_projection = find_optimal_solution(z, A, b, dimensions)
    # Initialise errors vector
    squared_errors = np.zeros(max_iter)
    # Initialise vectors for tracking stalling and convergence
    stalled_errors = np.zeros(max_iter)
    converged_errors = np.zeros(max_iter)

    # Main body of Dykstra's algorithm
    for i in range(max_iter):

        # Choose Beta at the start of every iteration (V6)
        beta = beta_check(x, A, b)
        # Choose between MAP and Dykstra's method
        if beta == 1:
            e = e_dykstra
        else:
            e = e_MAP # these are all zeros and do not change

        # Iterate over every halfspace
        for m, (normal, offset) in enumerate(zip(A, b)):

            # Get m - n index using modulo operator, which ensures
            # we get an index between 0 and n (non-negative)
            index = (m - n) % n  # this is essentially just m-n with zeros for m<n
            x_temp = x.copy() # temporary variable (x_m)

            # Check if current point is in the halfspace (V9)
            if plot_active_halfspaces:
                if not is_in_half_space(x + e[index], normal, offset):
                    # Set item to 1 if halfspace is active, 0 otherwise
                    active_half_spaces[m][i] = 1

            # Update x_m+1
            x = project_onto_half_space(x_temp + e[index], normal, offset)

            # Update e_m with Dykstra's method
            e_dykstra[m] = e_dykstra[index]+ (x_temp - x)

            # # Path
            # x_historical[i][m] = x.copy()
            # BUGFIX: Previous version
            path.append(x.copy())  # Add the updated x to the path


            # Errors
            if plot_errors:
                # errors_for_plotting[i][m] = e[m].copy()
                # BUGFIX: Previous version
                errors_for_plotting.append(e.copy()) # update error matrix


        # Track the squared error (V4)
        if track_error:
            distance = actual_projection - x
            error = round(np.dot(distance, distance), 10) # num error check
            # Check stalling, modulo used to avoid negative index
            i_minus_one = (i - 1) % max_iter
            # Define conditions for if check
            is_equal1 = squared_errors[i_minus_one] == error
            is_equal2 = stalled_errors[i_minus_one] == error
            # Check if we have converged
            if error < min_error:
                converged_errors[i] = error
                stalled_errors[i] = None
            # Check if we are stalling
            elif is_equal1 or is_equal2:
                stalled_errors[i] = error
                converged_errors[i] = None
            else:
                stalled_errors[i] = None
                converged_errors[i] = None
            # Append error
            squared_errors[i] = error

    # # Path
    # path = x_historical.copy()

    if track_error and plot_errors and plot_active_halfspaces:
        error_tuple = (squared_errors, stalled_errors, converged_errors)
        return x, path, error_tuple, errors_for_plotting, active_half_spaces
    elif track_error and plot_active_halfspaces:
        error_tuple = (squared_errors, stalled_errors, converged_errors)
        return x, path, error_tuple, None, active_half_spaces
    elif track_error and plot_errors:
        error_tuple = (squared_errors, stalled_errors, converged_errors)
        return x, path, error_tuple, errors_for_plotting, None
    elif track_error:
        error_tuple = (squared_errors, stalled_errors, converged_errors)
        return x, path, error_tuple, None, None
    else:
        return x, path, None, None, None