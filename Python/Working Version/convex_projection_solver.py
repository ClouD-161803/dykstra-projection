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
- Input validation and dimension inference.
- Active and inactive half-space plotting.
"""

from __future__ import annotations


import warnings
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
        A half space is defined by H_i := {x | <x,n_i> <= b_i}, with boundary
        B_i := {x | <x,n_i> = b_i}, and the projection of a point z onto H_i is
        given by: P_H_i(z) = z - (<z,n_i> - b_i)*n_i if z is outside H_i.

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
    def _delete_inactive_half_spaces(z: np.ndarray, A: np.ndarray,
                                     b: np.ndarray) -> tuple:
        """Deprecated compatibility helper that preserves every constraint.

        A half-space containing ``z`` cannot be removed safely: it may still
        bound the feasible intersection and determine the final projection.
        """
        del z
        warnings.warn(
            "_delete_inactive_half_spaces is deprecated and no longer removes "
            "constraints.",
            DeprecationWarning,
            stacklevel=2,
        )
        return A.copy(), b.copy()

    @staticmethod
    def _validate_problem(z: np.ndarray, A: np.ndarray, b: np.ndarray,
                          max_iter: int, dimensions: int | None) -> tuple:
        """Validate and copy a half-space projection problem."""
        z_array = np.asarray(z, dtype=float)
        A_array = np.asarray(A, dtype=float)
        b_array = np.asarray(b, dtype=float)

        if z_array.ndim != 1 or z_array.size == 0:
            raise ValueError("z must be a non-empty one-dimensional point.")
        if A_array.ndim != 2:
            raise ValueError("A must be a two-dimensional constraint matrix.")
        if b_array.ndim != 1:
            raise ValueError("b must be a one-dimensional vector of offsets.")
        if A_array.shape[0] != b_array.shape[0]:
            raise ValueError("A and b must contain the same number of constraints.")
        if A_array.shape[1] != z_array.size:
            raise ValueError(
                "Each normal in A must have the same dimension as z "
                f"({z_array.size})."
            )
        if not (np.isfinite(z_array).all() and np.isfinite(A_array).all()
                and np.isfinite(b_array).all()):
            raise ValueError("z, A, and b must contain only finite values.")
        if A_array.shape[0] and np.any(np.linalg.norm(A_array, axis=1) == 0):
            raise ValueError("A must not contain zero-norm constraint normals.")
        if (dimensions is not None and
                (isinstance(dimensions, (bool, np.bool_)) or
                 not isinstance(dimensions, (int, np.integer)) or
                 dimensions != z_array.size)):
            raise ValueError(
                "dimensions must match the dimension of z "
                f"({z_array.size}), or be omitted."
            )
        if (isinstance(max_iter, (bool, np.bool_)) or
                not isinstance(max_iter, (int, np.integer)) or max_iter < 0):
            raise ValueError("max_iter must be a non-negative integer.")

        return z_array.copy(), A_array.copy(), b_array.copy(), int(max_iter)

    @staticmethod
    def _find_optimal_solution(point: np.ndarray, A: np.ndarray, b: np.ndarray,
                               dimensions: int | None = None) -> np.ndarray:
        """
        Solves a quadratic programming problem to find the optimal solution that
        minimises the Euclidean distance between a given point and a target,
        subject to linear constraints.

        The function solves: min_x ∥x − point∥^2 subject to Gx <= h.

        Args:
            point: Target point in space for optimisation.
            A: Constraint matrix G in the quadratic programming formulation.
            b: Constraint vector h in the quadratic programming formulation.
            dimensions: Optional compatibility check for point dimensionality.

        Returns:
            np.ndarray: Optimal projection of the point.
        """
        if dimensions is not None and dimensions != point.size:
            raise ValueError("dimensions must match the dimension of point.")
        if A.shape[0] == 0:
            return point.copy()

        identity = np.eye(point.size)
        P = 2 * np.matmul(identity.T, identity)
        q = -2 * np.matmul(identity.T, point)
        G = A
        h = b

        actual_projection = quadprog_solve_qp(P, q, G, h)
        return actual_projection

    @staticmethod
    def _beta_check(point: np.ndarray, A: np.ndarray, b: np.ndarray) -> int:
        """
        Selects a value of beta based on whether the passed point lies
        within the intersection of half-spaces.

        Args:
            point: Point to check.
            A: Matrix of normal vectors.
            b: Vector of constant offsets.

        Returns:
            int: 1 if the point is within the intersection, else 0.
        """
        beta = 1
        not_in_intersection = False
        rounded_point = np.around(point, decimals=10)
        
        for _, (normal, offset) in enumerate(zip(A, b)):
            unit_normal, constant_offset = ConvexProjectionSolver._normalise(normal, offset)
            if not ConvexProjectionSolver._is_in_half_space(rounded_point, unit_normal, constant_offset):
                not_in_intersection = True
        
        if not_in_intersection:
            beta = 0
        return beta

    def __init__(self, z: np.ndarray, A: np.ndarray, b: np.ndarray,
                 max_iter: int, track_error: bool = False,
                 min_error: float = 1e-3, dimensions: int | None = None,
                 plot_errors: bool = False,
                 plot_active_halfspaces: bool = False,
                 delete_spaces: bool = False):
        """
        Initialise the solver.

        Args:
            z: Initial point to project.
            A: Matrix of normal vectors for half-spaces.
            b: Vector of constant offsets for half-spaces.
            max_iter: Maximum number of iterations.
            track_error: Whether to track squared error at each iteration.
            min_error: Minimum error threshold for convergence.
            dimensions: Optional consistency check. If omitted, inferred from z.
            plot_errors: Whether to track errors for plotting.
            plot_active_halfspaces: Whether to track active half-spaces.
            delete_spaces: Deprecated compatibility option. Constraints are retained
                because initial satisfaction does not imply redundancy.
        """
        self.z, self.A, self.b, self.max_iter = self._validate_problem(
            z, A, b, max_iter, dimensions
        )
        if delete_spaces:
            warnings.warn(
                "delete_spaces is deprecated and no longer removes constraints; "
                "a constraint satisfied at the initial point can still determine "
                "the projection.",
                DeprecationWarning,
                stacklevel=2,
            )

        self.track_error = track_error
        self.min_error = float(min_error)
        if not np.isfinite(self.min_error) or self.min_error < 0:
            raise ValueError("min_error must be a finite, non-negative value.")
        self.dimensions = self.z.size
        self.plot_errors = plot_errors
        self.plot_active_halfspaces = plot_active_halfspaces

        # Initialise variables
        self.n = self.A.shape[0]
        self.x = self.z.copy()
        self.errors = np.zeros_like(self.z)
        self.e = [self.errors.copy() for _ in range(self.n)]
        # Initialise errors_for_plotting as 3D NumPy array
        self.errors_for_plotting: np.ndarray = np.zeros(
            (self.max_iter, self.n, len(self.z))
        )
        # Track historical projections for path and quiver plotting, size max_iter + 1 for initial point
        self.x_historical: np.ndarray = np.zeros((self.max_iter + 1, self.n, len(self.z)))
        # Initialise first point as the initial point z
        self.x_historical[0, :, :] = self.z.copy()
        self.actual_projection = self._find_optimal_solution(self.z, self.A, self.b)
        # Active halfspaces tracking, sized for max_iter + 1 to include initial state
        self.active_half_spaces: np.ndarray = np.zeros((self.n, self.max_iter + 1))
        # Error tracking arrays sized for max_iter + 1 to include initial error
        self.squared_errors = np.zeros(self.max_iter + 1)
        self.stalled_errors = np.full(self.max_iter + 1, np.nan)
        self.converged_errors = np.full(self.max_iter + 1, np.nan)

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
        for m, (normal, offset) in enumerate(zip(self.A, self.b)):
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
            is_equal1 = i > 0 and self.squared_errors[i - 1] == error
            is_equal2 = i > 0 and self.stalled_errors[i - 1] == error

            if error < self.min_error:
                self.converged_errors[i] = error
                self.stalled_errors[i] = np.nan
            elif is_equal1 or is_equal2:
                self.stalled_errors[i] = error
                self.converged_errors[i] = np.nan
            else:
                self.stalled_errors[i] = np.nan
                self.converged_errors[i] = np.nan

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
            for m, (normal, offset) in enumerate(zip(self.A, self.b)):
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
        self.e_dykstra = [self.errors.copy() for _ in range(self.n)]
        self.e_MAP = [self.errors.copy() for _ in range(self.n)]

    def _initialize_iteration(self, i: int) -> None:
        """
        Initialise iteration by choosing between MAP and Dykstra.

        Args:
            i: Current iteration.
        """
        beta = self._beta_check(self.x, self.A, self.b)
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
            for m, (normal, offset) in enumerate(zip(self.A, self.b)):
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

    # Plain Dykstra rounds each e_m once per cycle, drifting from the linear
    # drain by about N * eps * d_m over N cycles: a whole cycle's drain once N
    # nears 1/sqrt(eps), so a longer jump cannot reproduce it. At a fixed point
    # the active slacks are rounding noise and ask for about 1/eps cycles.
    MAX_SKIP_CYCLES = np.finfo(float).eps ** -0.5

    def __init__(self, *args, **kwargs):
        """Initialise the stall detection solver with additional tracking."""
        super().__init__(*args, **kwargs)
        self.stalling = False
        self.k_stalling = 1
        self.m_stalling = None
        self.prev_x_no_ffw = None
        self.frozen_visits = 0
        self.active_at_last_visit = np.zeros(self.n, dtype=bool)

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
        Fast-forward through a detected stall.

        While the frozen cycle repeats, an active half-space m changes only
        d_m = e_m . a_m, by its constant slack s_m = a_m^T x_{m-1} - b_m per
        visit, and turns inactive on visit ceil(d_m / -s_m). With N the least
        such count over draining half-spaces (s_m < 0), N - 1 cycles are
        skipped and the switching cycle runs normally.

        Args:
            i: Current iteration.
        """
        if not (self.stalling and self.m_stalling is not None):
            return
        self.stalling = False
        self.m_stalling = None
        self.frozen_visits = 0

        # The frozen window's outputs; entry m - 1 is the input to half-space m.
        visits = self.x_historical[i]
        cycles_to_switch = np.inf
        for m, (normal, offset) in enumerate(zip(self.A, self.b)):
            if not self.active_at_last_visit[m]:
                # One that has just turned inactive moved its input by its old
                # e_m, and the next cycle will not repeat that.
                if not np.array_equal(visits[m], visits[m - 1]):
                    return
                continue
            unit_normal, constant_offset = self._normalise(normal, offset)
            slack = np.dot(visits[m - 1], unit_normal) - constant_offset
            if slack < 0:
                cycles_to_switch = min(cycles_to_switch,
                                       np.dot(self.e[m], unit_normal) / -slack)
        if cycles_to_switch > self.MAX_SKIP_CYCLES:
            return
        n_fast_forward = max(int(np.ceil(cycles_to_switch)), 1) - 1
        if n_fast_forward == 0:
            return

        print(f"Fast forwarding {n_fast_forward} rounds to exit stalling at iteration {i}. ")

        # Inactive half-spaces pass their input through, so they gain nothing.
        for m, (normal, offset) in enumerate(zip(self.A, self.b)):
            self.e[m] = self.e[m] + n_fast_forward * (visits[m - 1] - visits[m])
            if not self._is_in_half_space(self.x + self.e[(m - self.n) % self.n], normal, offset):
                self.active_half_spaces[m][i] = 1

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
            for m, (normal, offset) in enumerate(zip(self.A, self.b)):
                index = (m - self.n) % self.n
                x_temp = self.x.copy()

                # Handle stalling detection and fast-forward
                self._handle_stalling(i)

                # Check if current point is in the halfspace. Taken from the
                # projection branch because an inactive e_m can keep a rounding
                # residue with e_m . a_m > 0.
                self.active_at_last_visit[m] = self._check_activity(
                    m, i, x_temp, normal, offset, index)

                # Update x_m+1
                self.x = self._project_onto_half_space(x_temp + self.e[index], normal, offset)

                # Update e_m
                self._update_error(m, x_temp, self.x, index)

                # Store historical data for path and quiver plotting, offset by 1
                self.x_historical[i + 1][m] = self.x.copy()

                # Stalling as in Definition 1: the last n visits all repeat their
                # outputs from one cycle earlier. A single repeating visit is not
                # enough; a partly frozen cycle fed jumps plain Dykstra never makes.
                if i > 0 and np.array_equal(self.x_historical[i + 1][m], self.x_historical[i][m]):
                    self.frozen_visits += 1
                else:
                    self.frozen_visits = 0
                if self.frozen_visits >= self.n:
                    self.stalling = True
                    self.m_stalling = m

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

def dykstra_projection(z: np.ndarray, A: np.ndarray, b: np.ndarray,
                       max_iter: int, track_error: bool = False,
                       min_error: float = 1e-3, dimensions: int | None = None,
                       plot_errors: bool = False,
                       plot_active_halfspaces: bool = False,
                       delete_spaces: bool = False) -> tuple:
    """
    Backwards compatibility wrapper for DykstraProjectionSolver.
    """
    solver = DykstraProjectionSolver(
        z, A, b, max_iter, track_error, min_error, dimensions,
        plot_errors, plot_active_halfspaces, delete_spaces
    )
    result = solver.solve()
    return (result.projection, result.path, 
            (result.squared_errors, result.stalled_errors, result.converged_errors) if track_error else None,
            result.errors_for_plotting, result.active_half_spaces)
