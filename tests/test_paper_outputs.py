"""Regression coverage for the write-up figure and timing scripts."""

from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np


WORKING_VERSION = Path(__file__).resolve().parents[1] / "Python" / "Working Version"
sys.path.insert(0, str(WORKING_VERSION))

import paper_figures
import performance_timing
from paper_figures import (
    bounded_problem,
    corner,
    infeasible_start,
    phase_scalars,
    three_plane_arrangement,
)
from lti_ver3 import rigorous_deactivation_horizon
from lti_ver5 import LTIVer5Solver


ANALYTIC_FIGURES = (
    paper_figures.vertex_gamma_corner_figure,
    paper_figures.triple_loop_drift_figure,
    paper_figures.envelope_bounds_figure,
)


class ArrangementTests(unittest.TestCase):
    def test_vertex_floors_take_the_values_the_figure_labels(self) -> None:
        _, _, floors, _ = three_plane_arrangement()
        np.testing.assert_allclose(floors, [0.671141, -0.493720, -0.431576], atol=1e-6)

    def test_loop_residuals_take_the_slopes_the_figure_labels(self) -> None:
        _, _, _, residuals = three_plane_arrangement()
        np.testing.assert_allclose(residuals, [0.254888, -0.346483, -0.396375], atol=1e-6)

    def test_corners_lie_on_both_of_their_boundaries(self) -> None:
        A, b, _, _ = three_plane_arrangement()
        for i, j in ((0, 1), (0, 2), (1, 2)):
            with self.subTest(boundaries=(i, j)):
                vertex = corner(A, b, i, j)
                np.testing.assert_allclose(A[[i, j]] @ vertex, b[[i, j]], atol=1e-12)

    def test_the_draining_half_space_sheds_on_the_marked_cycle(self) -> None:
        A, b, _, _ = three_plane_arrangement()
        history = phase_scalars(A, b, np.array([0.9, 0.6]), np.array([4.0, 6.0, 5.0]), 20)
        self.assertEqual(int(np.argmax(history[:, 2] <= 1e-12)), 13)
        self.assertTrue(np.all(history >= -1e-12))


class EnvelopeTests(unittest.TestCase):
    def test_the_bounds_bracket_the_drift_line_crossing(self) -> None:
        rho, level, drift, amplitude = 0.85, 1.0, -0.06, 0.9
        lower = rigorous_deactivation_horizon(level, drift, amplitude, rho)
        upper = rigorous_deactivation_horizon(level, drift, -amplitude, rho)
        self.assertLess(lower, -level / drift)
        self.assertLess(-level / drift, upper)
        np.testing.assert_allclose([lower, upper], [15.448455, 17.534589], atol=1e-5)


class TestProblemTests(unittest.TestCase):
    def test_the_generated_region_is_bounded_and_has_an_outside_point(self) -> None:
        np.random.seed(0)
        A, b = bounded_problem(24, 6)
        self.assertEqual(A.shape, (24, 6))
        self.assertTrue(np.any(A @ infeasible_start(A, b, 6) > b))

    def test_the_generator_repeats_under_a_fixed_seed(self) -> None:
        np.random.seed(7)
        first = bounded_problem(12, 4)
        np.random.seed(7)
        second = bounded_problem(12, 4)
        np.testing.assert_array_equal(first[0], second[0])
        np.testing.assert_array_equal(first[1], second[1])


class CertifiedSettleTests(unittest.TestCase):
    def test_a_certified_result_agrees_with_the_reference_projection(self) -> None:
        for num_planes, num_dimensions in ((12, 4), (24, 8), (40, 12)):
            with self.subTest(size=(num_planes, num_dimensions)):
                np.random.seed(11 + num_planes)
                A, b = bounded_problem(num_planes, num_dimensions)
                z = infeasible_start(A, b, num_dimensions)
                solver = LTIVer5Solver(z, A, b, 2000, dimensions=num_dimensions)
                projection = solver.solve().projection
                if not solver.settled:
                    continue
                self.assertTrue(np.all(A @ projection <= b + 1e-8))
                np.testing.assert_allclose(projection, solver.actual_projection, atol=1e-6)


class PerformanceTimingTests(unittest.TestCase):
    def test_the_cycle_probe_finds_a_budget_reaching_the_tolerance(self) -> None:
        np.random.seed(3)
        A, b = bounded_problem(12, 4)
        z = infeasible_start(A, b, 4)
        cycles = performance_timing.cycles_to_tolerance(LTIVer5Solver, z, A, b, 4)
        self.assertIsNotNone(cycles)
        self.assertGreaterEqual(cycles, 1)

    def test_a_tier_reports_medians_for_every_solver(self) -> None:
        measured = performance_timing.time_tier(12, 4, 2)
        for name, _, _, _ in performance_timing.SOLVERS:
            with self.subTest(solver=name):
                self.assertIn(name, measured)
                self.assertGreater(measured[name]["seconds"], 0.0)


class FigureOutputTests(unittest.TestCase):
    def test_the_analytic_figures_write_both_file_formats(self) -> None:
        original = paper_figures.OUTPUT_DIR
        with tempfile.TemporaryDirectory() as temporary_directory:
            paper_figures.OUTPUT_DIR = temporary_directory
            try:
                for figure in ANALYTIC_FIGURES:
                    with self.subTest(figure=figure.__name__):
                        figure()
                written = sorted(path.suffix for path in Path(temporary_directory).iterdir())
            finally:
                paper_figures.OUTPUT_DIR = original
        self.assertEqual(written, [".pdf", ".pdf", ".pdf", ".png", ".png", ".png"])


if __name__ == "__main__":
    unittest.main()
