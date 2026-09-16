"""
This module implements a unified class-based architecture for projecting a point 
onto the intersection of convex sets (specifically, half-spaces).

Classes:
- ConvexProjectionSolver (base class):
    Abstract base class for projection onto intersection of half-spaces.
- DykstraProjectionSolver:
    Standard Dykstra's algorithm implementation.
- DykstraMapHybridSolver:
    Hybrid of MAP and Dykstra's algorithm.
- DykstraStallDetectionSolver:
    Modified Dykstra with stalling detection and exit mechanism.

Additional Features:
- Error tracking: Option to track and plot errors at each iteration.
- Convergence and stalling detection.
- Generalised for any number of dimensions.
- Inactive half-space removal.
- Active and inactive half-space plotting.
"""

from __future__ import annotations

import numpy as np
from abc import ABC, abstractmethod
from projection_result import ProjectionResult
from gradient import quadprog_solve_qp


class ConvexProjectionSolver(ABC):
    """
    Abstract base class for projecting a point onto the intersection of 
    convex sets (half-spaces).
    """

    @staticmethod
    def _normalise(normal: np.ndarray, offset: np.ndarray) -> tuple:
        """
        Normalises half space normal and constant offset.

        Args:
            normal: Normal vector of the half space.
            offset: Constant offset of the half space.

        Returns:
            tuple: Unit normal vector and normalised offset.
        """
        norm = np.linalg.norm(normal)
        if norm == 0:
            raise ValueError("Warning: Zero-norm normal vector encountered.")
        
        unit_normal = normal / norm
        constant_offset = offset / norm
        return unit_normal, constant_offset

    @staticmethod
    def _is_in_half_space(point: np.ndarray, unit_normal: np.ndarray,
                          constant_offset: np.ndarray) -> bool:
        """
        Checks if a point lies within a single half space.

        Args:
            point: Point to check.
            unit_normal: *Unit* normal vector of the half space.
            constant_offset: *Normalised* offset of the half space.

        Returns:
            bool: True if point is within the half space, else False.
        """
        dp = np.dot(point, unit_normal)
        return dp <= constant_offset

    @staticmethod
    def _project_onto_half_space(point: np.ndarray, normal: np.ndarray,
                                 offset: np.ndarray) -> np.ndarray:
        """
        Projects a point onto a single half space 'H_i'.
        A half space is defined by H_i := {x | <x,n_i> <= c_i}, with boundary
        B_i := {x | <x,n_i> = c_i}, and the projection of a point z onto H_i is
        given by: P_H_i(z) = z - (<z,n_i> - c_i)*n_i if z is outside H_i.

        Args:
            point: Point to project.
            normal: Normal vector of the half space.
            offset: Constant offset of the half space.

        Returns:
            np.ndarray: Projected point.
        """
        unit_normal, constant_offset = ConvexProjectionSolver._normalise(normal, offset)

        if ConvexProjectionSolver._is_in_half_space(point, unit_normal, constant_offset):
            return point
        else:
            boundary_projection = (point - (np.dot(point, unit_normal)
                                            - constant_offset) * unit_normal)
            return boundary_projection

    @staticmethod
    def _delete_inactive_half_spaces(z: np.ndarray, N: np.ndarray, c: np.ndarray) -> tuple:
        """
        Deletes inactive half spaces.

        Args:
            z: Point used to check for inactive half spaces.
            N: Matrix of normal vectors.
            c: Vector of constant offsets.

        Returns:
            tuple: Updated matrix of normal vectors and vector of constant offsets.
        """
        indices_to_remove = np.zeros_like(c, dtype=bool)

        for m, (normal, offset) in enumerate(zip(N, c)):
            unit_normal, constant_offset = ConvexProjectionSolver._normalise(normal, offset)
            if ConvexProjectionSolver._is_in_half_space(z, unit_normal, constant_offset):
                indices_to_remove[m] = True

        new_N = N[~indices_to_remove]
        new_c = c[~indices_to_remove]
        return new_N, new_c

    @staticmethod
    def _find_optimal_solution(point: np.ndarray, N: np.ndarray, c: np.ndarray,
                               dimensions: int) -> np.ndarray:
        """
        Solves a quadratic programming problem to find the optimal solution that
        minimises the Euclidean distance between a given point and a target,
        subject to linear constraints.

        The function solves: min_x ∥x − point∥^2 subject to Gx <= h.

        Args:
            point: Target point in space for optimisation.
            N: Constraint matrix G in the quadratic programming formulation.
            c: Constraint vector h in the quadratic programming formulation.
            dimensions: Dimensionality of the space.

        Returns:
            np.ndarray: Optimal projection of the point.
        """
        A = np.eye(dimensions)
        b = point.copy()
        P = 2 * np.matmul(A.T, A)
        q = -2 * np.matmul(A.T, b)
        G = N
        h = c

        actual_projection = quadprog_solve_qp(P, q, G, h)
        return actual_projection

    @staticmethod
    def _beta_check(point: np.ndarray, N: np.ndarray, c: np.ndarray) -> int:
        """
        Selects a value of beta based on whether the passed point lies
        within the intersection of half-spaces.

        Args:
            point: Point to check.
            N: Matrix of normal vectors.
            c: Vector of constant offsets.

        Returns:
            int: 1 if the point is within the intersection, else 0.
        """
        beta = 1
        not_in_intersection = False
        rounded_point = np.around(point, decimals=10)
        
        for _, (normal, offset) in enumerate(zip(N, c)):
            unit_normal, constant_offset = ConvexProjectionSolver._normalise(normal, offset)
            if not ConvexProjectionSolver._is_in_half_space(rounded_point, unit_normal, constant_offset):
                not_in_intersection = True
        
        if not_in_intersection:
            beta = 0
        return beta

    def __init__(self, z: np.ndarray, N: np.ndarray, c: np.ndarray,
                 max_iter: int, track_error: bool = False,
                 min_error: float = 1e-3, dimensions: int = 2,
                 plot_errors: bool = False,
                 plot_active_halfspaces: bool = False,
                 delete_spaces: bool = False):
        """
        Initialise the solver.

        Args:
            z: Initial point to project.
            N: Matrix of normal vectors for half-spaces.
            c: Vector of constant offsets for half-spaces.
            max_iter: Maximum number of iterations.
            track_error: Whether to track squared error at each iteration.
            min_error: Minimum error threshold for convergence.
            dimensions: Number of dimensions.
            plot_errors: Whether to track errors for plotting.
            plot_active_halfspaces: Whether to track active half-spaces.
            delete_spaces: Whether to delete inactive half-spaces at start.
        """
        self.z = z.copy()
        if delete_spaces:
            self.N, self.c = self._delete_inactive_half_spaces(z, N, c)
        else:
            self.N = N.copy()
            self.c = c.copy()
        
        self.max_iter = max_iter
        self.track_error = track_error
        self.min_error = min_error
        self.dimensions = dimensions
        self.plot_errors = plot_errors
        self.plot_active_halfspaces = plot_active_halfspaces

        # Initialise variables
        self.n = self.N.shape[0]
        self.x = self.z.copy()
        self.errors = np.zeros_like(self.z)
        self.e = [self.errors.copy() for _ in range(self.n)]
        # Initialise errors_for_plotting as 3D NumPy array
        self.errors_for_plotting: np.ndarray = np.zeros((max_iter, self.n, len(self.z)))
        # Track historical projections for path and quiver plotting, size max_iter + 1 for initial point
        self.x_historical: np.ndarray = np.zeros((self.max_iter + 1, self.n, len(self.z)))
        # Initialise first point as the initial point z
        self.x_historical[0, :, :] = self.z.copy()
        self.actual_projection = self._find_optimal_solution(self.z, self.N, self.c, dimensions)
        # Active halfspaces tracking, sized for max_iter + 1 to include initial state
        self.active_half_spaces: np.ndarray = np.zeros((self.n, max_iter + 1))
        # Error tracking arrays sized for max_iter + 1 to include initial error
        self.squared_errors = np.zeros(max_iter + 1)
        self.stalled_errors = np.zeros(max_iter + 1)
        self.converged_errors = np.zeros(max_iter + 1)

    @abstractmethod
    def _update_error(self, m: int, x_temp: np.ndarray, x: np.ndarray, index: int) -> None:
        """
        Update the error vector for half-space m.
        This method must be implemented by subclasses.

        Args:
            m: Index of current half-space.
            x_temp: Temporary point before projection.
            x: Projected point.
            index: Index for error lookup.
        """
        pass

    def _initialize_iteration(self, i: int) -> None:
        """
        Initialise variables for the current iteration.
        Can be overridden by subclasses for custom behaviour.

        Args:
            i: Current iteration number.
        """
        pass

    def _check_activity(self, m: int, i: int, x_temp: np.ndarray, normal: np.ndarray,
                        offset: np.ndarray, index: int) -> bool:
        """
        Check if a half-space is active.

        Args:
            m: Index of half-space.
            i: Current iteration.
            x_temp: Temporary point before projection.
            normal: Normal vector of half-space.
            offset: Offset of half-space.
            index: Error index.
            
        Returns:
            bool: True if the half-space is active, False otherwise.
        """
        return not self._is_in_half_space(x_temp + self.e[index], normal, offset)

    def _track_activity(self, cycle_index: int) -> None:
        """
        Track active half-spaces for the current cycle.
        This should be called after all n half-spaces have been processed.

        Args:
            cycle_index: Index for storing activity data (0 for initial, 1 for first cycle, etc).
        """
        for m, (normal, offset) in enumerate(zip(self.N, self.c)):
            index = (m - self.n) % self.n
            if not self._is_in_half_space(self.x + self.e[index], normal, offset):
                self.active_half_spaces[m][cycle_index] = 1

    def _track_error(self, i: int) -> None:
        """
        Track squared error, convergence, and stalling.

        Args:
            i: Current iteration.
        """
        if self.track_error:
            distance = self.actual_projection - self.x
            error = round(np.dot(distance, distance), 10)
            i_minus_one = (i - 1) % self.max_iter
            is_equal1 = self.squared_errors[i_minus_one] == error
            is_equal2 = self.stalled_errors[i_minus_one] == error

            if error < self.min_error:
                self.converged_errors[i] = error
                self.stalled_errors[i] = None
            elif is_equal1 or is_equal2:
                self.stalled_errors[i] = error
                self.converged_errors[i] = None
            else:
                self.stalled_errors[i] = None
                self.converged_errors[i] = None

            self.squared_errors[i] = error

    @abstractmethod
    def solve(self) -> ProjectionResult:
        """
        Solve the projection problem.
        Must be implemented by subclasses.

        Returns:
            ProjectionResult: Object containing all solver results.
        """
        pass


class DykstraProjectionSolver(ConvexProjectionSolver):
    """
    Standard implementation of Dykstra's algorithm for projecting a point 
    onto the intersection of half-spaces.
    """

    def _update_error(self, m: int, x_temp: np.ndarray, x: np.ndarray, index: int) -> None:
        """
        Update error vector using standard Dykstra's method.

        Args:
            m: Index of current half-space.
            x_temp: Temporary point before projection.
            x: Projected point.
            index: Index for error lookup.
        """
        self.e[m] = self.e[index] + 1 * (x_temp - x)  # change 1 to 0 for MAP

    def solve(self) -> ProjectionResult:
        """
        Projects a point 'z' onto the intersection of convex sets H_i (half spaces)
        using standard Dykstra's algorithm.

        Returns:
            ProjectionResult: Object containing projection and tracking data.
        """
        # Track error and activity at the initial point
        self._track_error(0)
        self._track_activity(0)
        
        # Main body of Dykstra's algorithm
        for i in range(self.max_iter):
            # Iterate over every half plane
            for m, (normal, offset) in enumerate(zip(self.N, self.c)):
                index = (m - self.n) % self.n
                x_temp = self.x.copy()

                # Check if current point is in the halfspace
                self._check_activity(m, i, x_temp, normal, offset, index)

                # Update x_m+1
                self.x = self._project_onto_half_space(x_temp + self.e[index], normal, offset)

                # Update e_m
                self._update_error(m, x_temp, self.x, index)

                # Store historical data for path and quiver plotting, offset by 1
                self.x_historical[i + 1][m] = self.x.copy()

                # Errors
                if self.plot_errors:
                    self.errors_for_plotting[i][m] = self.e[m].copy()

            # Track the squared error and activity after each complete cycle through all n half-spaces
            self._track_error(i + 1)
            self._track_activity(i + 1)

        return self._format_output()

    def _format_output(self) -> ProjectionResult:
        """Format and return output as ProjectionResult object."""
        return ProjectionResult(
            projection=self.x,
            path=self.x_historical,
            squared_errors=self.squared_errors if self.track_error else None,
            stalled_errors=self.stalled_errors if self.track_error else None,
            converged_errors=self.converged_errors if self.track_error else None,
            errors_for_plotting=self.errors_for_plotting if self.plot_errors else None,
            active_half_spaces=self.active_half_spaces if self.plot_active_halfspaces else None
        )


class DykstraMapHybridSolver(ConvexProjectionSolver):
    """
    Hybrid implementation combining MAP and Dykstra's algorithm.
    Switches between MAP and Dykstra based on whether the current point 
    lies within the feasible region.
    """

    def __init__(self, *args, **kwargs):
        """Initialise the hybrid solver with separate error vectors for MAP and Dykstra."""
        super().__init__(*args, **kwargs)
        self.e_dykstra = [self.errors] * self.n
        self.e_MAP = [self.errors] * self.n

    def _initialize_iteration(self, i: int) -> None:
        """
        Initialise iteration by choosing between MAP and Dykstra.

        Args:
            i: Current iteration.
        """
        beta = self._beta_check(self.x, self.N, self.c)
        if beta == 1:
            self.e = self.e_dykstra
        else:
            self.e = self.e_MAP

    def _update_error(self, m: int, x_temp: np.ndarray, x: np.ndarray, index: int) -> None:
        """
        Update error vector using Dykstra's method (MAP errors remain zero).

        Args:
            m: Index of current half-space.
            x_temp: Temporary point before projection.
            x: Projected point.
            index: Index for error lookup.
        """
        self.e_dykstra[m] = self.e_dykstra[index] + (x_temp - x)

    def solve(self) -> ProjectionResult:
        """
        Projects a point 'z' onto the intersection of convex sets (half spaces)
        using a hybrid version of Dykstra and MAP.

        Returns:
            ProjectionResult: Object containing projection and tracking data.
        """
        # Track error and activity at the initial point
        self._track_error(0)
        self._track_activity(0)
        
        # Main body of Dykstra's algorithm
        for i in range(self.max_iter):
            # Choose Beta at the start of every iteration
            self._initialize_iteration(i)

            # Iterate over every halfspace
            for m, (normal, offset) in enumerate(zip(self.N, self.c)):
                index = (m - self.n) % self.n
                x_temp = self.x.copy()

                # Check if current point is in the halfspace
                self._check_activity(m, i, x_temp, normal, offset, index)

                # Update x_m+1
                self.x = self._project_onto_half_space(x_temp + self.e[index], normal, offset)

                # Update e_m with Dykstra's method
                self._update_error(m, x_temp, self.x, index)

                # Store historical data for path and quiver plotting, offset by 1
                self.x_historical[i + 1][m] = self.x.copy()

                # Errors
                if self.plot_errors:
                    self.errors_for_plotting[i][m] = self.e[m].copy()

            # Track the squared error and activity after each complete cycle through all n half-spaces
            self._track_error(i + 1)
            self._track_activity(i + 1)

        return self._format_output()

    def _format_output(self) -> ProjectionResult:
        """Format and return output as ProjectionResult object."""
        return ProjectionResult(
            projection=self.x,
            path=self.x_historical,
            squared_errors=self.squared_errors if self.track_error else None,
            stalled_errors=self.stalled_errors if self.track_error else None,
            converged_errors=self.converged_errors if self.track_error else None,
            errors_for_plotting=self.errors_for_plotting if self.plot_errors else None,
            active_half_spaces=self.active_half_spaces if self.plot_active_halfspaces else None
        )


class DykstraStallDetectionSolver(ConvexProjectionSolver):
    """
    Modified Dykstra's algorithm with stalling detection and exit mechanism.
    Can detect when the algorithm stalls and attempt to exit stalling 
    in a single iteration.
    """

    def __init__(self, *args, **kwargs):
        """Initialise the stall detection solver with additional tracking."""
        super().__init__(*args, **kwargs)
        self.stalling = False
        self.k_stalling = 1
        self.m_stalling = None
        self.prev_x_no_ffw = None

    def _update_error(self, m: int, x_temp: np.ndarray, x: np.ndarray, index: int) -> None:
        """
        Update error vector using standard Dykstra's method.

        Args:
            m: Index of current half-space.
            x_temp: Temporary point before projection.
            x: Projected point.
            index: Index for error lookup.
        """
        self.e[m] = self.e[index] + 1 * (x_temp - x)  # change to 0 for MAP

    def _handle_stalling(self, i: int) -> None:
        """
        Handle stalling detection and fast-forwarding.

        Args:
            i: Current iteration.
        """
        if self.stalling and self.m_stalling is not None:
            n_fast_forward = int(min(
                [np.ceil(- np.dot(self.e[m], normal) / (np.dot(self.x_historical[i][m-1], normal) - offset))
                 if np.dot(self.x_historical[i][m-1], normal) < offset else 1e6
                 for m, (normal, offset) in enumerate(zip(self.N, self.c))]
            ))
            n_fast_forward -= 1

            print(f"Fast forwarding {n_fast_forward} rounds to exit stalling at iteration {i}. ")

            # Update all errors for the following round
            for m, (normal, offset) in enumerate(zip(self.N, self.c)):
                self.e[m] = self.e[m] + n_fast_forward * (self.x_historical[i][m-1] - self.x_historical[i][m])
                if not self._is_in_half_space(self.x + self.e[(m - self.n) % self.n], normal, offset):
                    self.active_half_spaces[m][i] = 1

            self.stalling = False
            self.m_stalling = None

    def solve(self) -> ProjectionResult:
        """
        Projects a point 'z' onto the intersection of convex sets (half spaces)
        using modified Dykstra's algorithm with stalling detection.

        Returns:
            ProjectionResult: Object containing projection and tracking data.
        """
        self.stalling = False
        
        # Track error at the initial point
        self._track_error(0)
        self._track_activity(0)

        # Main body of Dykstra's algorithm
        for i in range(self.max_iter):
            # Iterate over every half plane
            for m, (normal, offset) in enumerate(zip(self.N, self.c)):
                index = (m - self.n) % self.n
                x_temp = self.x.copy()

                # Handle stalling detection and fast-forward
                self._handle_stalling(i)

                # Check if current point is in the halfspace
                self._check_activity(m, i, x_temp, normal, offset, index)

                # Update x_m+1
                self.x = self._project_onto_half_space(x_temp + self.e[index], normal, offset)

                # Update e_m
                self._update_error(m, x_temp, self.x, index)

                # Store historical data for path and quiver plotting, offset by 1
                self.x_historical[i + 1][m] = self.x.copy()

                # Check for stalling
                if i > 0:
                    if ((not self.stalling) and (self.active_half_spaces[m][i] == 1) and
                            np.array_equal(self.x_historical[i + 1][m], self.x_historical[i][m])):
                        self.stalling = True
                        self.m_stalling = m
                        print(f"Stalling detected at iteration {i} and half-space {self.m_stalling}")

                # Errors
                if self.plot_errors:
                    self.errors_for_plotting[i][m] = self.e[m].copy()

            # Track the squared error after each complete cycle through all n half-spaces
            self._track_error(i + 1)
            self._track_activity(i + 1)

        return self._format_output()

    def _format_output(self) -> ProjectionResult:
        """Format and return output as ProjectionResult object."""
        return ProjectionResult(
            projection=self.x,
            path=self.x_historical,
            squared_errors=self.squared_errors if self.track_error else None,
            stalled_errors=self.stalled_errors if self.track_error else None,
            converged_errors=self.converged_errors if self.track_error else None,
            errors_for_plotting=self.errors_for_plotting if self.plot_errors else None,
            active_half_spaces=self.active_half_spaces if self.plot_active_halfspaces else None
        )


# Backwards compatibility wrapper

def dykstra_projection(z: np.ndarray, N: np.ndarray, c: np.ndarray,
                       max_iter: int, track_error: bool = False,
                       min_error: float = 1e-3, dimensions: int = 2,
                       plot_errors: bool = False,
                       plot_active_halfspaces: bool = False,
                       delete_spaces: bool = False) -> tuple:
    """
    Backwards compatibility wrapper for DykstraProjectionSolver.
    """
    solver = DykstraProjectionSolver(
        z, N, c, max_iter, track_error, min_error, dimensions,
        plot_errors, plot_active_halfspaces, delete_spaces
    )
    result = solver.solve()
    return (result.projection, result.path, 
            (result.squared_errors, result.stalled_errors, result.converged_errors) if track_error else None,
            result.errors_for_plotting, result.active_half_spaces)