from __future__ import annotations

import csv
import os
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.figure import Figure
from matplotlib.axes import Axes
from matplotlib.patches import Rectangle
from projection_result import ProjectionResult


class ResultExporter:
    
    @staticmethod
    def export(result: ProjectionResult, output_path: str, solver_name: str,
               initial_point: np.ndarray, N: np.ndarray, c: np.ndarray,
               max_iter: int, **kwargs) -> None:
        
        if os.path.isdir(output_path) or not output_path.endswith('.csv'):
            os_path = Path(output_path)
            os_path.mkdir(parents=True, exist_ok=True)
            filename = f"{solver_name}_{max_iter}_iterations.csv"
            output_path = os.path.join(output_path, filename)
        else:
            directory = os.path.dirname(output_path)
            if directory:
                Path(directory).mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            
            writer.writerow(['METADATA'])
            writer.writerow(['solver_name', solver_name])
            writer.writerow(['max_iterations', max_iter])
            writer.writerow(['dimensions', len(initial_point)])
            writer.writerow(['num_constraints', N.shape[0]])
            
            for key, value in kwargs.items():
                writer.writerow([key, value])
            
            writer.writerow([])
            
            writer.writerow(['INITIAL_POINT'])
            writer.writerow(['z_' + str(i) for i in range(len(initial_point))])
            writer.writerow(initial_point)
            writer.writerow([])
            
            writer.writerow(['FINAL_PROJECTION'])
            writer.writerow(['projection_' + str(i) for i in range(len(result.projection))])
            writer.writerow(result.projection)
            writer.writerow([])
            
            writer.writerow(['CONSTRAINTS_N'])
            for i, normal in enumerate(N):
                writer.writerow([f'normal_{i}'] + normal.tolist())
            writer.writerow([])
            
            writer.writerow(['CONSTRAINTS_C'])
            writer.writerow(['c_' + str(i) for i in range(len(c))])
            writer.writerow(c)
            writer.writerow([])
            
            if result.path is not None:
                writer.writerow(['PATH_HISTORY'])
                writer.writerow(['iteration', 'constraint', 'dim_0', 'dim_1'] + 
                              [f'dim_{i}' for i in range(2, len(initial_point))])
                
                path = result.path
                for i in range(path.shape[0]):
                    for m in range(path.shape[1]):
                        writer.writerow([i, m] + path[i, m].tolist())
                writer.writerow([])
            
            if result.squared_errors is not None:
                writer.writerow(['SQUARED_ERRORS'])
                writer.writerow(['iteration', 'squared_error'])
                for i, error in enumerate(result.squared_errors):
                    writer.writerow([i, error])
                writer.writerow([])
            
            if result.stalled_errors is not None:
                writer.writerow(['STALLED_ERRORS'])
                writer.writerow(['iteration', 'stalled_error'])
                for i, error in enumerate(result.stalled_errors):
                    error_val = (
                        '' if error is None or
                        (isinstance(error, (float, np.floating)) and np.isnan(error))
                        else error
                    )
                    writer.writerow([i, error_val])
                writer.writerow([])
            
            if result.converged_errors is not None:
                writer.writerow(['CONVERGED_ERRORS'])
                writer.writerow(['iteration', 'converged_error'])
                for i, error in enumerate(result.converged_errors):
                    error_val = (
                        '' if error is None or
                        (isinstance(error, (float, np.floating)) and np.isnan(error))
                        else error
                    )
                    writer.writerow([i, error_val])
                writer.writerow([])
            
            if result.errors_for_plotting is not None:
                writer.writerow(['ERRORS_FOR_PLOTTING'])
                writer.writerow(['iteration', 'constraint', 'dim_0', 'dim_1'] +
                              [f'dim_{i}' for i in range(2, len(initial_point))])
                
                errors = result.errors_for_plotting
                for i in range(errors.shape[0]):
                    for m in range(errors.shape[1]):
                        writer.writerow([i, m] + errors[i, m].tolist())
                writer.writerow([])
            
            if result.active_half_spaces is not None:
                active = result.active_half_spaces
                writer.writerow(['ACTIVE_HALFSPACES'])
                writer.writerow(
                    ['constraint'] +
                    [f'iteration_{i}' for i in range(active.shape[1])]
                )

                for m in range(active.shape[0]):
                    writer.writerow([m] + active[m].tolist())
                writer.writerow([])
        
        print(f"Results exported to: {output_path}")
    
    @staticmethod
    def load(csv_path: str) -> dict:
        data = {}
        
        with open(csv_path, 'r', newline='') as csvfile:
            reader = csv.reader(csvfile)
            lines = list(reader)
        
        i = 0
        while i < len(lines):
            if not lines[i] or not lines[i][0]:
                i += 1
                continue
            
            section_name = lines[i][0].strip()
            i += 1
            
            if section_name == 'METADATA':
                data['metadata'] = {}
                while i < len(lines) and lines[i] and lines[i][0]:
                    if lines[i][0] in ['INITIAL_POINT', 'FINAL_PROJECTION', 'CONSTRAINTS_N',
                                       'CONSTRAINTS_C', 'PATH_HISTORY', 'SQUARED_ERRORS',
                                       'STALLED_ERRORS', 'CONVERGED_ERRORS', 
                                       'ERRORS_FOR_PLOTTING', 'ACTIVE_HALFSPACES']:
                        break
                    key, val = lines[i][0], lines[i][1] if len(lines[i]) > 1 else ''
                    try:
                        data['metadata'][key] = int(val)
                    except (ValueError, IndexError):
                        try:
                            data['metadata'][key] = float(val)
                        except (ValueError, IndexError):
                            data['metadata'][key] = val
                    i += 1
            
            elif section_name == 'INITIAL_POINT':
                i += 1
                data['initial_point'] = np.array([float(x) for x in lines[i]])
                i += 1
            
            elif section_name == 'FINAL_PROJECTION':
                i += 1
                data['final_projection'] = np.array([float(x) for x in lines[i]])
                i += 1
            
            elif section_name == 'CONSTRAINTS_N':
                normals = []
                while i < len(lines) and lines[i] and lines[i][0]:
                    if lines[i][0] in ['CONSTRAINTS_C', 'PATH_HISTORY', 'SQUARED_ERRORS',
                                      'STALLED_ERRORS', 'CONVERGED_ERRORS',
                                      'ERRORS_FOR_PLOTTING', 'ACTIVE_HALFSPACES']:
                        break
                    normals.append(np.array([float(x) for x in lines[i][1:]]))
                    i += 1
                data['constraints_N'] = np.array(normals)
            
            elif section_name == 'CONSTRAINTS_C':
                i += 1
                data['constraints_c'] = np.array([float(x) for x in lines[i]])
                i += 1
            
            elif section_name == 'PATH_HISTORY':
                i += 1
                path_data = []
                while i < len(lines) and lines[i] and len(lines[i]) > 2:
                    if lines[i][0] in ['SQUARED_ERRORS', 'STALLED_ERRORS', 'CONVERGED_ERRORS',
                                      'ERRORS_FOR_PLOTTING', 'ACTIVE_HALFSPACES']:
                        break
                    try:
                        int(lines[i][0])
                        path_data.append([float(x) for x in lines[i][2:]])
                    except (ValueError, IndexError):
                        break
                    i += 1
                if path_data:
                    path = np.array(path_data)
                    num_constraints = data.get('metadata', {}).get('num_constraints')
                    if (isinstance(num_constraints, int) and num_constraints > 0 and
                            path.shape[0] % num_constraints == 0):
                        path = path.reshape(-1, num_constraints, path.shape[-1])
                    data['path'] = path
            
            elif section_name == 'SQUARED_ERRORS':
                i += 1
                errors = []
                while i < len(lines) and lines[i] and len(lines[i]) > 1:
                    if lines[i][0] in ['STALLED_ERRORS', 'CONVERGED_ERRORS',
                                      'ERRORS_FOR_PLOTTING', 'ACTIVE_HALFSPACES']:
                        break
                    try:
                        int(lines[i][0])
                        errors.append(float(lines[i][1]))
                    except (ValueError, IndexError):
                        break
                    i += 1
                if errors:
                    data['squared_errors'] = np.array(errors)
            
            elif section_name == 'STALLED_ERRORS':
                i += 1
                errors = []
                while i < len(lines) and lines[i] and len(lines[i]) > 1:
                    if lines[i][0] in ['CONVERGED_ERRORS', 'ERRORS_FOR_PLOTTING', 'ACTIVE_HALFSPACES']:
                        break
                    try:
                        int(lines[i][0])
                        val = lines[i][1] if lines[i][1] else None
                        errors.append(float(val) if val else None)
                    except (ValueError, IndexError):
                        break
                    i += 1
                if errors:
                    data['stalled_errors'] = np.array(errors, dtype=object)
            
            elif section_name == 'CONVERGED_ERRORS':
                i += 1
                errors = []
                while i < len(lines) and lines[i] and len(lines[i]) > 1:
                    if lines[i][0] in ['ERRORS_FOR_PLOTTING', 'ACTIVE_HALFSPACES']:
                        break
                    try:
                        int(lines[i][0])
                        val = lines[i][1] if lines[i][1] else None
                        errors.append(float(val) if val else None)
                    except (ValueError, IndexError):
                        break
                    i += 1
                if errors:
                    data['converged_errors'] = np.array(errors, dtype=object)
            
            elif section_name == 'ERRORS_FOR_PLOTTING':
                i += 1
                errors_data = []
                while i < len(lines) and lines[i] and len(lines[i]) > 2:
                    if lines[i][0] in ['ACTIVE_HALFSPACES']:
                        break
                    try:
                        int(lines[i][0])
                        errors_data.append([float(x) for x in lines[i][2:]])
                    except (ValueError, IndexError):
                        break
                    i += 1
                if errors_data:
                    errors = np.array(errors_data)
                    num_constraints = data.get('metadata', {}).get('num_constraints')
                    if (isinstance(num_constraints, int) and num_constraints > 0 and
                            errors.shape[0] % num_constraints == 0):
                        errors = errors.reshape(-1, num_constraints, errors.shape[-1])
                    data['errors_for_plotting'] = errors
            
            elif section_name == 'ACTIVE_HALFSPACES':
                i += 1
                active_data = []
                while i < len(lines) and lines[i]:
                    if not lines[i][0]:
                        break
                    try:
                        int(lines[i][0])
                        active_data.append([int(float(x)) for x in lines[i][1:]])
                    except (ValueError, IndexError):
                        break
                    i += 1
                if active_data:
                    data['active_halfspaces'] = np.array(active_data)
            
            else:
                i += 1
        
        return data


class Visualiser:
    """
    Unified visualisation class for projection solver results.
    
    Attributes:
        result: ProjectionResult object containing solver outputs.
        nc_pairs: List of tuples (label, cmap, N, c) defining half-spaces.
        max_iter: Number of iterations performed.
        x_range: X-axis range for plotting.
        y_range: Y-axis range for plotting.
    """
    
    def __init__(self, result: ProjectionResult, nc_pairs: list, 
                 max_iter: int, x_range: list[float], y_range: list[float],
                 solver_name: str = "Dykstra's Algorithm") -> None:
        """
        Initialise the visualiser.
        
        Args:
            result: ProjectionResult object from solver.
            nc_pairs: List of (label, cmap, N, c) tuples for half-spaces.
            max_iter: Number of iterations.
            x_range: [min_x, max_x] for plotting domain.
            y_range: [min_y, max_y] for plotting domain.
            solver_name: Name of the solver used.
        """
        self.result = result
        self.nc_pairs = nc_pairs
        self.max_iter = max_iter
        self.x_range = x_range
        self.y_range = y_range
        self.solver_name = solver_name
        self.fig: Figure | None = None
        self.ax_main: Axes | None = None
        self.ax_error: Axes | None = None
        self.fontsize_title = 20
        self.fontsize_label = 18
        self.fontsize_tick = 15
        self.fontsize_legend = 17

    def plot_2d_space(self, N: np.ndarray, c: np.ndarray, X: np.ndarray, Y: np.ndarray,
                      label: str, cmap: str, ax: Axes) -> None:
        """
        Plot a 2D region defined by the intersection of half-spaces.

        Args:
            N: Matrix of normal vectors.
            c: Vector of constant offsets.
            X: 2D array of x coordinates.
            Y: 2D array of y coordinates.
            label: Label for the plot.
            cmap: Colourmap name.
            ax: Axes handle for plotting.
        """
        Z = np.ones_like(X)

        for i in range(N.shape[0]):
            dot_product = np.dot(np.vstack([X.ravel(), Y.ravel()]).T, N[i])
            Z = np.where(dot_product.reshape(X.shape) > c[i], 0, Z)

        colourmap = plt.get_cmap(cmap)
        colour = colourmap(0.69)

        ax.contourf(X, Y, Z, levels=[0.5, 1.5], colors=[colour], alpha=0.5)
        ax.plot([], [], color=colour, alpha=0.5, label=label)

    def plot_1d_space(self, N: np.ndarray, c: np.ndarray, label: str, 
                      cmap: str, ax: Axes) -> None:
        """
        Plot a 1D region (line) defined by the intersection of half-spaces.

        Args:
            N: Matrix of normal vectors.
            c: Vector of constant offsets.
            label: Label for the plot.
            cmap: Colourmap name.
            ax: Axes handle for plotting.
        """
        colourmap = plt.get_cmap(cmap)
        colour = colourmap(0.69)

        if N[0, 1] == 0:
            ax.axvline(x=c[0] / N[0, 0], linestyle='-', linewidth=2,
                        label='Vertical line', color=colour)
        elif N[0, 0] == 0:
            ax.axhline(y=c[0] / N[0, 1], linestyle='-', linewidth=2,
                        label='Horizontal line', color=colour)
        else:
            x_line = np.linspace(self.x_range[0], self.x_range[1], 100)
            y_line = (c[0] - N[0, 0] * x_line) / N[0, 1]
            ax.plot(x_line, y_line, linewidth=2, label=label, color=colour)

    def plot_half_spaces(self, ax: Axes) -> None:
        """
        Plot all half-spaces.

        Args:
            ax: Axes handle for plotting.
        """
        try:
            x = np.linspace(self.x_range[0], self.x_range[1], 500)
            y = np.linspace(self.y_range[0], self.y_range[1], 500)
            X, Y = np.meshgrid(x, y)

            for label, cmap, N, c in self.nc_pairs:
                rank = np.linalg.matrix_rank(N)
                if rank == 1:
                    self.plot_1d_space(N, c, label, cmap, ax)
                elif rank == 2:
                    self.plot_2d_space(N, c, X, Y, label, cmap, ax)
                else:
                    raise ValueError("Dimension not supported. "
                                   "Please provide N and c for 1D or 2D cases.")

            ax.set_aspect('equal')
            ax.set_xlabel('X coordinate', fontsize=self.fontsize_label)
            ax.set_ylabel('Y coordinate', fontsize=self.fontsize_label)
            # ax.set_title(f"{self.solver_name} Executed for {self.max_iter} iterations", fontsize=self.fontsize_title)
            ax.set_xlim(self.x_range[0], self.x_range[1])
            ax.set_ylim(self.y_range[0], self.y_range[1])
            ax.tick_params(axis='both', which='major', labelsize=self.fontsize_tick)
            ax.grid(True)
            ax.legend(fontsize=self.fontsize_legend)

        except TypeError as e:
            print(f"TypeError occurred: {e}. "
                  f"Please ensure nc_pairs is a list of tuples.")
        except ValueError as e:
            print(f"ValueError occurred: {e}. "
                  f"Check the format of nc_pairs or the dimensions of N.")

    def plot_path(self, ax: Axes) -> None:
        """
        Plot the path followed by the algorithm with optional quiver plotting.

        Args:
            ax: Axes handle for plotting.
        """
        path = self.result.path
        errors_for_plotting = self.result.errors_for_plotting
        
        if path is None:
            print("Path data not available.")
            return
        
        # Ensure path is proper shape
        if path.ndim != 3:
            print("Path has unexpected shape.")
            return

        flattened_path = path.reshape(-1, path.shape[-1])
        
        x_coords = [point[0] for point in flattened_path]
        y_coords = [point[1] for point in flattened_path]

        # Plot the path
        ax.plot(x_coords, y_coords, marker='.', linestyle='--',
                color='blue', linewidth=0.5, markersize=1,
                label='Projection path')

        # Plot the errors (quivers) - only where errors were tracked
        # Skip i=0 (initial point x0) as it has no errors associated with it
        if errors_for_plotting is not None and errors_for_plotting.ndim == 3:
            max_iter, n_spaces = errors_for_plotting.shape[0], errors_for_plotting.shape[1]
            for i in range(1, max_iter + 1):  # Start from 1 to skip x0
                for m in range(n_spaces):
                    error = errors_for_plotting[i - 1, m]  # Offset by 1 since errors_for_plotting starts from i=0
                    # Only plot quiver if error vector is non-zero
                    if not np.allclose(error, 0):
                        # The point for the quiver is the start of the projection step
                        point = path[i, m]
                        ax.quiver(point[0], point[1], error[0], error[1],
                                 angles='xy', scale_units='xy', scale=1, alpha=0.3, width=0.002, zorder=5)

        ax.legend()

    def plot_errors(self, ax: Axes) -> None:
        """
        Plot the squared error convergence.

        Args:
            ax: Axes handle for plotting.
        """
        if (self.result.squared_errors is None or 
            self.result.stalled_errors is None or 
            self.result.converged_errors is None):
            print("Error tracking data not available.")
            return

        squared_errors = self.result.squared_errors.copy()
        stalled_errors = self.result.stalled_errors
        converged_errors = self.result.converged_errors
        
        iterations = np.arange(0, len(squared_errors), 1)
        
        ax.plot(iterations, squared_errors, color='red',
                label='Error', linestyle='-', marker='o', markersize=4)
        ax.plot(iterations, stalled_errors, color='#D5B60A',
                label='Stalling', linestyle='-', marker='o', markersize=4)
        ax.plot(iterations, converged_errors, color='green',
                label='Converged\n(error under 1e-3)', linestyle='-', marker='o', markersize=4)
        ax.scatter(len(squared_errors) - 1, squared_errors[-1],
                   color='green', marker='*', s=100, zorder=5,
                   label=f'Final error is {format(squared_errors[-1], ".2e")}')

        ax.set_xlabel('Cycle', fontsize=self.fontsize_label)
        ax.set_ylabel('Squared error', fontsize=self.fontsize_label)
        ax.set_title('Convergence of squared error', fontsize=self.fontsize_title)
        ax.set_yscale('log')
        ax.tick_params(axis='both', which='major', labelsize=self.fontsize_tick)
        ax.grid(True, axis='x', alpha=0.3)
        ax.legend(fontsize=self.fontsize_legend)

    def plot_active_halfspaces(self, fig: Figure, gs: gridspec.GridSpec) -> None:
        """
        Plot the activity of half-spaces over iterations.
        Colors match the error tracking per iteration: green for converged, yellow for stalling, red for errors.

        Args:
            fig: Figure handle.
            gs: GridSpec handle.
        """
        if self.result.active_half_spaces is None:
            print("Active half-space data not available.")
            return

        active_spaces = self.result.active_half_spaces
        num_of_spaces = active_spaces.shape[0]
        iterations = np.arange(0, active_spaces.shape[1], 1)

        for i in range(num_of_spaces):
            ax = fig.add_subplot(gs[i, 1])
            if ax is None:
                continue
            
            active_space = active_spaces[i]
            
            # Plot the active space data in black
            ax.plot(iterations, active_space, color='black',
                   linestyle='-', marker='o', linewidth=1.5, markersize=4)
            
            # Add legend entry
            ax.plot([], [], color='black', label=f'Halfspace {i}', linestyle='-', marker='o', linewidth=1.5, markersize=4)
            
            ax.set_ylim(-0.1, 1.1)
            ax.set_yticks([0, 1])
            ax.set_yticklabels(['Inactive', 'Active'], fontsize=self.fontsize_tick)
            ax.tick_params(axis='x', which='major', labelsize=self.fontsize_tick)
            
            # Only show x-axis labels on the bottom plot
            if i < num_of_spaces - 1:
                ax.set_xticklabels([])
            else:
                ax.set_xlabel('Cycle', fontsize=self.fontsize_label)

            if i == 0:
                ax.set_title('Halfspace activity', fontsize=self.fontsize_title)

            ax.grid(True, axis='x', alpha=0.3)
            ax.legend(loc='center right', fontsize=self.fontsize_legend)

    def visualise(self, plot_original_point: np.ndarray | None = None,
                  plot_optimal_point: np.ndarray | None = None) -> None:
        """
        Create a comprehensive visualisation of the projection results.

        Args:
            plot_original_point: Original point z (optional).
            plot_optimal_point: Optimal solution (optional).
        """
        # Give each activity trace its own row. The previous fixed three-row
        # grid failed whenever a problem contained more than three constraints.
        num_halfspaces = (
            self.result.active_half_spaces.shape[0]
            if self.result.active_half_spaces is not None else 0
        )
        total_rows = max(3, num_halfspaces)
        self.fig = plt.figure(
            figsize=(16, max(10, 2 * total_rows)), layout="constrained"
        )
        gs = gridspec.GridSpec(total_rows, 2, figure=self.fig, wspace=0.35)
        self.ax_main = self.fig.add_subplot(gs[0:total_rows - 1, 0])
        self.ax_error = self.fig.add_subplot(gs[total_rows - 1, 0])

        if self.ax_main is None or self.ax_error is None:
            print("Failed to create axes.")
            return

        self.plot_half_spaces(self.ax_main)

        self.plot_path(self.ax_main)

        if plot_original_point is not None:
            self.ax_main.scatter(plot_original_point[0], plot_original_point[1],
                               color='blue', marker='o', label='Original point', zorder=5)

        self.ax_main.scatter(self.result.projection[0], self.result.projection[1],
                           color='green', marker='*', s=100, label='Projection', zorder=5)

        if plot_optimal_point is not None:
            self.ax_main.scatter(plot_optimal_point[0], plot_optimal_point[1],
                               color='green', marker='*', s=40, label='Optimal solution', zorder=5)

        self.ax_main.legend(fontsize=self.fontsize_legend)

        if self.result.squared_errors is not None:
            self.plot_errors(self.ax_error)

        if self.result.active_half_spaces is not None:
            self.plot_active_halfspaces(self.fig, gs)

        plt.show()


class VerticalVisualiser(Visualiser):
    """
    Extended visualiser that arranges all graphs vertically in a single column.
    Inherits from Visualiser and overrides the visualise method to create a 
    different layout with smaller halfspace activity plots.
    """

    def visualise(self, plot_original_point: np.ndarray | None = None,
                  plot_optimal_point: np.ndarray | None = None) -> None:
        """
        Create a vertical layout visualisation with:
        - Main projection plot (top)
        - Error convergence plot (middle)
        - A combined half-space activity plot (bottom)

        Args:
            plot_original_point: Original point z (optional).
            plot_optimal_point: Optimal solution (optional).
        """
        # Reserve rows for the projection, error, and combined activity plots.
        if self.result.active_half_spaces is not None:
            num_halfspaces = self.result.active_half_spaces.shape[0]
            total_rows = 9
        else:
            num_halfspaces = 0
            total_rows = 7

        self.fig = plt.figure(
            figsize=(12, 12 if num_halfspaces else 10), layout="constrained"
        )
        gs = gridspec.GridSpec(total_rows, 1, figure=self.fig)
        
        self.ax_main = self.fig.add_subplot(gs[0:4, 0])
        self.ax_error = self.fig.add_subplot(gs[4:7, 0])
        self.ax_activity = self.fig.add_subplot(gs[7:9, 0]) if num_halfspaces > 0 else None

        if self.ax_main is None or self.ax_error is None:
            print("Failed to create axes.")
            return

        self.plot_half_spaces(self.ax_main)
        self.plot_path(self.ax_main)

        if plot_original_point is not None:
            self.ax_main.scatter(plot_original_point[0], plot_original_point[1],
                               color='blue', marker='o', label='Original point', zorder=5)

        self.ax_main.scatter(self.result.projection[0], self.result.projection[1],
                           color='green', marker='*', s=100, label='Projection', zorder=5)

        if plot_optimal_point is not None:
            self.ax_main.scatter(plot_optimal_point[0], plot_optimal_point[1],
                               color='red', marker='*', s=50, label='Optimal solution', zorder=5)
        self.ax_main.legend(fontsize=self.fontsize_legend)

        if self.result.squared_errors is not None:
            self.plot_errors(self.ax_error)
            self.ax_error.set_title("")
            # Remove xlabel from error plot if activity plot will be displayed below
            if self.result.active_half_spaces is not None:
                self.ax_error.set_xlabel('')

        if self.result.active_half_spaces is not None and self.ax_activity is not None:
            active_spaces = self.result.active_half_spaces
            iterations = np.arange(0, active_spaces.shape[1], 1)
            markers = ['o', 's', '^', 'D', 'v', '<', '>', 'p', '*', 'h']
            
            for i in range(num_halfspaces):
                marker = markers[i % len(markers)]
                active_space = active_spaces[i]
                self.ax_activity.plot(iterations, active_space, color='black',
                                     linestyle='-', marker=marker, linewidth=1.5, markersize=4,
                                     label=f'halfspace {i}')
            
            self.ax_activity.set_ylim(-0.1, 1.1)
            self.ax_activity.set_yticks([0, 1])
            self.ax_activity.set_yticklabels(['0', '1'], fontsize=self.fontsize_tick)
            self.ax_activity.tick_params(axis='x', which='major', labelsize=self.fontsize_tick)
            self.ax_activity.set_xlabel('Cycle', fontsize=self.fontsize_label)
            self.ax_activity.set_ylabel('Half-space activity', fontsize=self.fontsize_label)
            self.ax_activity.grid(True, axis='x', alpha=0.3)
            self.ax_activity.legend(loc='center right', fontsize=self.fontsize_legend)

        plt.show()


class ComparisonVisualiser:
    
    def __init__(self, result1: ProjectionResult, result2: ProjectionResult,
                 nc_pairs1: list, nc_pairs2: list,
                 max_iter: int, x_range: list[float], y_range: list[float],
                 solver_name1: str, solver_name2: str,
                 initial_point: np.ndarray,
                 display_result_index: int = 0,
                 display_halfspace_index: int = 0) -> None:
        
        self.result1 = result1
        self.result2 = result2
        self.nc_pairs1 = nc_pairs1
        self.nc_pairs2 = nc_pairs2
        self.max_iter = max_iter
        self.x_range = x_range
        self.y_range = y_range
        self.solver_name1 = solver_name1
        self.solver_name2 = solver_name2
        self.initial_point = initial_point
        self.display_result_index = display_result_index
        self.display_halfspace_index = display_halfspace_index
        self.fig: Figure | None = None
        self.fontsize_title = 16
        self.fontsize_label = 14
        self.fontsize_tick = 12
        self.fontsize_legend = 12
    
    def plot_2d_space(self, N: np.ndarray, c: np.ndarray, X: np.ndarray, Y: np.ndarray,
                      label: str, cmap: str, ax: Axes) -> None:
        
        Z = np.ones_like(X)
        for i in range(N.shape[0]):
            dot_product = np.dot(np.vstack([X.ravel(), Y.ravel()]).T, N[i])
            Z = np.where(dot_product.reshape(X.shape) > c[i], 0, Z)

        colourmap = plt.get_cmap(cmap)
        colour = colourmap(0.69)
        ax.contourf(X, Y, Z, levels=[0.5, 1.5], colors=[colour], alpha=0.5)
        ax.plot([], [], color=colour, alpha=0.5, label=label)

    def plot_1d_space(self, N: np.ndarray, c: np.ndarray, label: str, 
                      cmap: str, ax: Axes) -> None:
        
        colourmap = plt.get_cmap(cmap)
        colour = colourmap(0.69)
        if N[0, 1] == 0:
            ax.axvline(x=c[0] / N[0, 0], linestyle='-', linewidth=2,
                        label='Vertical line', color=colour)
        elif N[0, 0] == 0:
            ax.axhline(y=c[0] / N[0, 1], linestyle='-', linewidth=2,
                        label='Horizontal line', color=colour)
        else:
            x_line = np.linspace(self.x_range[0], self.x_range[1], 100)
            y_line = (c[0] - N[0, 0] * x_line) / N[0, 1]
            ax.plot(x_line, y_line, linewidth=2, label=label, color=colour)

    def plot_top_projection(self, ax: Axes) -> None:
        
        result = self.result1 if self.display_result_index == 0 else self.result2
        nc_pairs = self.nc_pairs1 if self.display_result_index == 0 else self.nc_pairs2
        solver_name = self.solver_name1 if self.display_result_index == 0 else self.solver_name2
        
        try:
            x = np.linspace(self.x_range[0], self.x_range[1], 500)
            y = np.linspace(self.y_range[0], self.y_range[1], 500)
            X, Y = np.meshgrid(x, y)

            for label, cmap, N, c in nc_pairs:
                rank = np.linalg.matrix_rank(N)
                if rank == 1:
                    self.plot_1d_space(N, c, label, cmap, ax)
                elif rank == 2:
                    self.plot_2d_space(N, c, X, Y, label, cmap, ax)

            ax.set_aspect('equal')
            ax.set_xlabel(r'$x$ coordinate', fontsize=self.fontsize_label)
            ax.set_ylabel(r'$y$ coordinate', fontsize=self.fontsize_label)
            ax.set_title(f"{solver_name} - {self.max_iter} iterations", fontsize=self.fontsize_title)
            ax.set_xlim(self.x_range[0], self.x_range[1])
            ax.set_ylim(self.y_range[0], self.y_range[1])
            ax.tick_params(axis='both', which='major', labelsize=self.fontsize_tick)
            ax.grid(True)
            
            path = result.path
            if path is not None:
                flattened_path = path.reshape(-1, path.shape[-1])
                x_coords = [point[0] for point in flattened_path]
                y_coords = [point[1] for point in flattened_path]
                ax.plot(x_coords, y_coords, marker='.', linestyle='--',
                        color='blue', linewidth=0.5, markersize=1, label='Projection path')

            ax.scatter(self.initial_point[0], self.initial_point[1],
                      color='blue', marker='o', label='Original point', zorder=5)
            ax.scatter(result.projection[0], result.projection[1],
                      color='green', marker='*', s=100, label='Projection', zorder=5)
            
            ax.legend(fontsize=self.fontsize_legend)

        except (TypeError, ValueError) as e:
            print(f"Error plotting: {e}")

    def plot_error_comparison(self, ax: Axes) -> None:
        
        if (self.result1.squared_errors is None or 
            self.result2.squared_errors is None):
            print("Error tracking data not available.")
            return

        squared_errors1 = self.result1.squared_errors.copy()
        squared_errors2 = self.result2.squared_errors.copy()
        stalled_errors1 = self.result1.stalled_errors
        
        iterations = np.arange(0, len(squared_errors1), 1)
        
        ax.plot(iterations, squared_errors1, color='#1f77b4', linestyle='-', 
                marker='^', markersize=4, label=f'{self.solver_name1}')
        ax.plot(iterations, squared_errors2, color='#ff7f0e', linestyle='-', 
                marker='s', markersize=4, label=f'{self.solver_name2}')
        
        stalled_indices = []
        if stalled_errors1 is not None:
            for i in range(len(stalled_errors1)):
                error = stalled_errors1[i]
                if error is not None and not (isinstance(error, float) and np.isnan(error)):
                    error_val = float(error)
                    rect_width = 0.8
                    rect_height = error_val * 0.5
                    rect = Rectangle((i - rect_width/2, error_val - rect_height/2), 
                                    rect_width, rect_height, 
                                    alpha=0.3, color='yellow', zorder=1)
                    ax.add_patch(rect)
                    stalled_indices.append(i)
        
        ax.scatter(len(squared_errors1) - 1, squared_errors1[-1],
                  color='#1f77b4', marker='*', s=150, zorder=5,
                  label=f'{self.solver_name1} (final)')
        ax.scatter(len(squared_errors2) - 1, squared_errors2[-1],
                  color='#ff7f0e', marker='*', s=150, zorder=5,
                  label=f'{self.solver_name2} (final)')
        
        if stalled_indices:
            ax.plot([], [], color='yellow', marker='s', linestyle='', markersize=8,
                   label='Stalling')
        
        # ax.set_xlabel('Cycle', fontsize=self.fontsize_label)
        ax.set_ylabel('Squared error', fontsize=self.fontsize_label)
        # ax.set_title('Error comparison', fontsize=self.fontsize_title)
        ax.set_yscale('log')
        ax.tick_params(axis='both', which='major', labelsize=self.fontsize_tick)
        ax.grid(True, axis='x', alpha=0.3, which='both')
        ax.legend(fontsize=self.fontsize_legend, loc='best')

    def plot_halfspace_comparison(self, ax: Axes) -> None:
        
        if (self.result1.active_half_spaces is None or 
            self.result2.active_half_spaces is None):
            print("Halfspace activity data not available.")
            return
        
        active_spaces1 = self.result1.active_half_spaces
        active_spaces2 = self.result2.active_half_spaces
        
        if (self.display_halfspace_index >= active_spaces1.shape[0] or
            self.display_halfspace_index >= active_spaces2.shape[0]):
            print(f"Halfspace index {self.display_halfspace_index} out of range.")
            return
        
        iterations = np.arange(0, active_spaces1.shape[1], 1)
        active_space1 = active_spaces1[self.display_halfspace_index]
        active_space2 = active_spaces2[self.display_halfspace_index]
        
        ax.plot(iterations, active_space1, color='#1f77b4', linestyle='-', 
                marker='^', linewidth=1.5, markersize=4, 
                label=f'{self.solver_name1}')
        ax.plot(iterations, active_space2, color='#ff7f0e', linestyle='-', 
                marker='s', linewidth=1.5, markersize=4, 
                label=f'{self.solver_name2}')
        
        ax.set_ylim(-0.1, 1.1)
        ax.set_yticks([0, 1])
        ax.set_yticklabels(['0', '1'], fontsize=self.fontsize_tick)
        ax.tick_params(axis='x', which='major', labelsize=self.fontsize_tick)
        ax.set_xlabel('Cycle', fontsize=self.fontsize_label)
        ax.set_ylabel(f'Halfspace {self.display_halfspace_index} activity', fontsize=self.fontsize_label)
        # ax.set_title(f'Halfspace {self.display_halfspace_index} activity', fontsize=self.fontsize_title)
        ax.grid(True, axis='x', alpha=0.3, which='both')
        ax.legend(fontsize=self.fontsize_legend)

    def visualise(self) -> None:
        
        self.fig = plt.figure(figsize=(12, 12), layout="constrained")
        gs = gridspec.GridSpec(3, 1, figure=self.fig, height_ratios=[4, 3, 3])
        
        ax_top = self.fig.add_subplot(gs[0])
        ax_middle = self.fig.add_subplot(gs[1])
        ax_bottom = self.fig.add_subplot(gs[2])
        
        if ax_top is None or ax_middle is None or ax_bottom is None:
            print("Failed to create axes.")
            return
        
        self.plot_top_projection(ax_top)
        self.plot_error_comparison(ax_middle)
        self.plot_halfspace_comparison(ax_bottom)
        
        plt.show()
