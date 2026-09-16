"""Regression tests for plotting and CSV result handling."""

import csv
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np


WORKING_VERSION = Path(__file__).resolve().parents[1] / "Python" / "Working Version"
sys.path.insert(0, str(WORKING_VERSION))

from projection_result import ProjectionResult
from visualiser import ResultExporter, Visualiser


class VisualiserRegressionTests(unittest.TestCase):
    def test_horizontal_layout_handles_more_than_three_constraints(self) -> None:
        result = ProjectionResult(
            projection=np.array([0.0, 0.0]),
            path=np.zeros((3, 6, 2)),
            squared_errors=np.array([1.0, 0.5, 0.25]),
            stalled_errors=np.full(3, np.nan),
            converged_errors=np.full(3, np.nan),
            active_half_spaces=np.zeros((6, 3)),
        )
        ab_pairs = [
            (
                "Box",
                "Greys",
                np.array([[1.0, 0.0], [-1.0, 0.0], [0.0, 1.0], [0.0, -1.0]]),
                np.ones(4),
            )
        ]
        visualiser = Visualiser(result, ab_pairs, 2, [-1.0, 1.0], [-1.0, 1.0])

        with patch("visualiser.plt.show"):
            visualiser.visualise()

        self.assertEqual(len(visualiser.fig.axes), 8)
        plt.close(visualiser.fig)


class ResultExporterRegressionTests(unittest.TestCase):
    def test_csv_loader_restores_cycle_by_constraint_shapes(self) -> None:
        result = ProjectionResult(
            projection=np.array([0.5, 0.5]),
            path=np.arange(12, dtype=float).reshape(3, 2, 2),
            squared_errors=np.array([1.0, 0.25, 0.0]),
            stalled_errors=np.array([np.nan, 0.25, np.nan]),
            converged_errors=np.array([np.nan, np.nan, 0.0]),
            errors_for_plotting=np.arange(8, dtype=float).reshape(2, 2, 2),
            active_half_spaces=np.array([[0, 1, 1], [1, 0, 0]]),
        )

        with tempfile.TemporaryDirectory() as temporary_directory:
            output_path = Path(temporary_directory) / "result.csv"
            ResultExporter.export(
                result=result,
                output_path=str(output_path),
                solver_name="Dykstra",
                initial_point=np.array([1.0, 1.0]),
                A=np.array([[1.0, 0.0], [0.0, 1.0]]),
                b=np.array([1.0, 1.0]),
                max_iter=2,
            )
            loaded = ResultExporter.load(str(output_path))

        np.testing.assert_array_equal(
            loaded["constraints_A"], np.array([[1.0, 0.0], [0.0, 1.0]])
        )
        np.testing.assert_array_equal(loaded["constraints_b"], np.array([1.0, 1.0]))
        self.assertEqual(loaded["path"].shape, (3, 2, 2))
        self.assertEqual(loaded["errors_for_plotting"].shape, (2, 2, 2))
        self.assertEqual(loaded["active_halfspaces"].shape, (2, 3))

    def test_csv_loader_accepts_legacy_constraint_sections(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            legacy_path = Path(temporary_directory) / "legacy-result.csv"
            with legacy_path.open("w", newline="") as csvfile:
                csv.writer(csvfile).writerows([
                    ["METADATA"],
                    ["num_constraints", "1"],
                    [],
                    ["CONSTRAINTS_N"],
                    ["normal_0", "1.0", "0.0"],
                    [],
                    ["CONSTRAINTS_C"],
                    ["c_0"],
                    ["1.0"],
                    [],
                ])
            loaded = ResultExporter.load(str(legacy_path))

        np.testing.assert_array_equal(loaded["constraints_A"], np.array([[1.0, 0.0]]))
        np.testing.assert_array_equal(loaded["constraints_b"], np.array([1.0]))


if __name__ == "__main__":
    unittest.main()
