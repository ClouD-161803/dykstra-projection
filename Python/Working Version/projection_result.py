"""
This module defines the ProjectionResult dataclass for storing solver outputs.
"""

from __future__ import annotations

from dataclasses import dataclass
import numpy as np


@dataclass
class ProjectionResult:
    """
    Data class for storing projection solver results.
    
    Attributes:
        projection: Final projected point.
        path: Historical path of projections (x_historical array).
        squared_errors: Squared errors at each iteration (if track_error=True).
        stalled_errors: Stalled errors at each iteration (if track_error=True).
        converged_errors: Converged errors at each iteration (if track_error=True).
        errors_for_plotting: Error vectors for quiver plotting (if plot_errors=True).
        active_half_spaces: Active half-spaces tracking (if plot_active_halfspaces=True).
        settled_at: First cycle whose recorded state is the limit rather than that
            cycle's Dykstra iterate, for the LTI solvers that settle (None otherwise).
        certificate: How the settlement was proven: "kkt" when the limit passed the
            KKT test and is the projection itself, "finality" when the active set was
            proven final and the limit is the fixed point of the last cycle map.
    """
    projection: np.ndarray
    path: np.ndarray | None = None
    squared_errors: np.ndarray | None = None
    stalled_errors: np.ndarray | None = None
    converged_errors: np.ndarray | None = None
    errors_for_plotting: np.ndarray | None = None
    active_half_spaces: np.ndarray | None = None
    settled_at: int | None = None
    certificate: str | None = None

    def has_error_tracking(self) -> bool:
        """Check if error tracking data is available."""
        return self.squared_errors is not None
    
    def has_error_plotting_data(self) -> bool:
        """Check if error plotting data is available."""
        return self.errors_for_plotting is not None
    
    def has_active_halfspace_data(self) -> bool:
        """Check if active half-space data is available."""
        return self.active_half_spaces is not None

    def is_settled(self) -> bool:
        """Check if the result is a settled limit rather than the last iterate."""
        return self.settled_at is not None
