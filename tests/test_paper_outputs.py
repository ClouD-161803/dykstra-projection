"""Regression coverage for the write-up figure and timing scripts."""

from contextlib import contextmanager
import importlib.util
import math
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
from lti_numerics import rigorous_deactivation_horizon
from lti_solver import LTIVer5Solver
import c_solvers
import problem_generators
import square_line_example
from convex_projection_solver import ConvexProjectionSolver


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
        settled = 0
        for num_planes, num_dimensions in ((12, 4), (24, 8), (40, 12)):
            with self.subTest(size=(num_planes, num_dimensions)):
                np.random.seed(11 + num_planes)
                A, b = bounded_problem(num_planes, num_dimensions)
                z = infeasible_start(A, b, num_dimensions)
                solver = LTIVer5Solver(z, A, b, 2000, dimensions=num_dimensions)
                projection = solver.solve().projection
                if not solver.settled:
                    continue
                settled += 1
                self.assertTrue(np.all(A @ projection <= b + 1e-8))
                np.testing.assert_allclose(projection, solver.actual_projection, atol=1e-6)
        # Otherwise every size skipped its checks and the test proved nothing
        self.assertGreater(settled, 0)


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



BENCHMARK_GROUP = "needs the benchmark dependency group: uv sync --group benchmark"
HAS_SCIPY = importlib.util.find_spec("scipy") is not None
HAS_QPSOLVERS = importlib.util.find_spec("qpsolvers") is not None


@contextmanager
def redirected_output():
    """Send every file the figure, table and timing scripts write to a temporary directory."""
    original = paper_figures.OUTPUT_DIR
    with tempfile.TemporaryDirectory() as temporary_directory:
        paper_figures.OUTPUT_DIR = temporary_directory
        try:
            yield Path(temporary_directory)
        finally:
            paper_figures.OUTPUT_DIR = original


class BoundedTestPlanesTests(unittest.TestCase):
    def test_a_seed_names_the_problem_the_global_stream_draws(self) -> None:
        np.random.seed(problem_generators.problem_seed(24, 3))
        A, b = bounded_problem(24, 6)
        z = infeasible_start(A, b, 6)
        expected = problem_generators.problem(24, 6, problem_generators.problem_seed(24, 3))
        for drawn, seeded in zip((A, b, z), expected):
            np.testing.assert_array_equal(drawn, seeded)

    def test_the_spanning_rows_bound_the_region_and_the_region_is_nonempty(self) -> None:
        for num_planes, num_dimensions in ((5, 4), (24, 8), (100, 26)):
            with self.subTest(size=(num_planes, num_dimensions)):
                A, b, z = problem_generators.problem(num_planes, num_dimensions, 7)
                spanning = A[:num_dimensions + 1]
                # Positive weights of one each sum the spanning rows to zero, and the first
                # p of them span the space, hence they positively span it
                np.testing.assert_allclose(spanning.sum(axis=0), 0.0, atol=1e-12)
                self.assertEqual(np.linalg.matrix_rank(spanning[:-1]), num_dimensions)
                self.assertAlmostEqual(np.linalg.norm(spanning[-1]), math.sqrt(num_dimensions))
                inside = ConvexProjectionSolver._find_optimal_solution(z, A, b)
                self.assertTrue(np.all(A @ inside <= b + 1e-9 * (1.0 + np.abs(b))))
                self.assertTrue(np.any(A @ z > b))

    def test_too_few_half_spaces_to_bound_the_region_are_refused(self) -> None:
        with self.assertRaises(ValueError):
            problem_generators.BoundedTestPlanes(4, 4)


class CauchyNormalTests(unittest.TestCase):
    def test_directions_crowd_the_axes_against_a_uniform_law(self) -> None:
        # The entries s tan(theta) are standard Cauchy, and the share of directions within
        # pi/8 of the positive first axis is half of P(|C_2| < tan(pi/8) |C_1|), an integral
        # over the angle of C_1 evaluated here by the midpoint rule
        ratio = math.tan(math.pi / 8)
        angles = (np.arange(200_000) + 0.5) / 200_000 * (math.pi / 2)
        exact = 0.5 * (2 / math.pi) * np.mean(np.arctan(ratio * np.tan(angles)))
        rows = problem_generators.cauchy_normals(np.random.RandomState(2026), 200_000, 2)
        share = np.mean(np.abs(np.arctan2(rows[:, 1], rows[:, 0])) < math.pi / 8)
        standard_error = math.sqrt(share * (1 - share) / len(rows))
        self.assertAlmostEqual(exact, 0.16435, places=4)
        self.assertLess(abs(share - exact), 5 * standard_error)
        self.assertGreater(share - 0.125, 20 * standard_error)


@unittest.skipUnless(HAS_SCIPY, BENCHMARK_GROUP)
class RandomTestPlanesTests(unittest.TestCase):
    def test_an_accepted_draw_is_bounded_and_nonempty_by_independent_checks(self) -> None:
        from scipy.optimize import nnls

        for num_planes, num_dimensions in ((6, 2), (12, 4), (20, 6)):
            with self.subTest(size=(num_planes, num_dimensions)):
                generator = problem_generators.RandomTestPlanes(num_planes, num_dimensions)
                A, b = generator.draw(np.random.RandomState(num_planes))
                self.assertGreaterEqual(generator.draws_used, 1)
                # Every coordinate direction and its negative is a nonnegative combination of
                # the rows exactly when the region is bounded
                for direction in np.vstack([np.eye(num_dimensions), -np.eye(num_dimensions)]):
                    _, residual = nnls(A.T, direction)
                    self.assertLess(residual, 1e-8)
                z = np.zeros(num_dimensions) + 100.0
                inside = ConvexProjectionSolver._find_optimal_solution(z, A, b)
                self.assertTrue(np.all(A @ inside <= b + 1e-8 * (1.0 + np.abs(A) @ np.abs(inside))))

    def test_the_sampler_rejects_draws_and_repeats_under_a_seed(self) -> None:
        generator = problem_generators.RandomTestPlanes(20, 6)
        first = generator.draw(np.random.RandomState(1))
        self.assertGreater(generator.draws_used, 1)
        second = generator.draw(np.random.RandomState(1))
        np.testing.assert_array_equal(first[0], second[0])
        np.testing.assert_array_equal(first[1], second[1])

    def test_an_unbounded_or_empty_region_is_rejected(self) -> None:
        test = problem_generators.RandomTestPlanes.is_bounded_and_nonempty
        self.assertFalse(test(np.array([[1.0, 0.0], [0.0, 1.0]]), np.ones(2)))
        self.assertFalse(test(np.array([[1.0], [-1.0]]), np.array([-1.0, -1.0])))
        self.assertTrue(test(np.array([[1.0], [-1.0]]), np.array([1.0, 1.0])))


class PaperTexTests(unittest.TestCase):
    def test_the_vertex_figure_draws_the_corners_arrows_and_floors(self) -> None:
        geometry = paper_figures.vertex_corner_geometry()
        for base, tip in zip(geometry["normal_bases"], geometry["normal_tips"]):
            self.assertAlmostEqual(np.linalg.norm(tip - base), 0.28)
        floors = geometry["floors"]
        self.assertAlmostEqual(np.linalg.norm(np.subtract(*geometry["gamma_1"])), floors[0])
        self.assertAlmostEqual(np.linalg.norm(np.subtract(*geometry["gamma_3"])), -floors[2])
        text = paper_figures.vertex_corner_tikz(geometry)
        for drawn in ("(1.6711,0.7426)", "(-1.1,1.186)", "(-1.1,-0.8785)", "(2.6,1.286)"):
            self.assertIn(drawn, text)

    def test_the_drift_figure_holds_the_phase_and_its_drift_lines(self) -> None:
        data = paper_figures.triple_loop_drift_data()
        np.testing.assert_allclose(data["history"][-1], [7.1552, 1.3762, 0.0], atol=5e-5)
        np.testing.assert_allclose(data["drift"][0], [3.6451, 6.0996, 4.9899], atol=5e-5)
        np.testing.assert_allclose(data["drift"][1], data["history"][1])
        self.assertEqual(data["shed"], 13)
        text = paper_figures.triple_loop_drift_pgfplots(data)
        self.assertIn("(13,0.0000)", text)
        self.assertIn("entered at $x = (0.9, 0.6)$", text)

    def test_the_drawn_trajectory_first_reaches_zero_inside_the_bracket(self) -> None:
        crossings = paper_figures.envelope_crossings()
        self.assertLess(crossings["lower"], crossings["switch"])
        self.assertLess(crossings["switch"], crossings["upper"])
        self.assertAlmostEqual(float(paper_figures.active_trajectory(crossings["switch"])), 0.0,
                               places=9)
        before = np.linspace(0.0, crossings["switch"] - 1e-6, 20001)
        self.assertTrue(np.all(paper_figures.active_trajectory(before) > 0.0))
        lower = (paper_figures.ENVELOPE_LEVEL + paper_figures.ENVELOPE_DRIFT * crossings["lower"]
                 - paper_figures.ENVELOPE_AMPLITUDE
                 * paper_figures.ENVELOPE_RHO ** crossings["lower"])
        self.assertAlmostEqual(lower, 0.0, places=9)
        text = paper_figures.envelope_bounds_pgfplots(crossings)
        self.assertIn("domain=0:16.022", text)
        self.assertIn("(axis cs:3.265,", text)

    def test_written_tex_names_the_command_that_produced_it(self) -> None:
        with redirected_output() as directory:
            path = paper_figures.save_tex("x\n", "fig-test", "python make.py")
            self.assertEqual(Path(path).parent, directory)
            self.assertIn("python make.py", Path(path).read_text().splitlines()[0])


class PerformanceErrorTests(unittest.TestCase):
    def test_the_accelerated_curve_follows_dykstra_until_it_certifies(self) -> None:
        curves = paper_figures.error_curves(24, 8, 0, 800)
        self.assertEqual(curves["certificate"], "kkt")
        # The closed forms reach Dykstra's iterates up to rounding, which accumulates
        self.assertLess(curves["path_difference"], 1e-9)
        self.assertLess(curves["projection_error"], 1e-9)
        self.assertEqual(curves["first_accelerated"], min(curves["settle"],
                                                          curves["first_dykstra"]))
        sample = paper_figures.thinned_cycles(curves, 800)
        self.assertIn(curves["settle"] - 1, sample)
        self.assertIn(curves["settle"], sample)
        text = paper_figures.performance_error_pgfplots(curves, 800)
        self.assertIn(f"settle ({curves['settle']})", text)
        self.assertIn(f"({curves['settle']},1e-10)", text)


class RunToToleranceTests(unittest.TestCase):
    def test_a_short_run_writes_the_table_and_the_scaling_panel(self) -> None:
        implementations = ("python", "c") if c_solvers.libraries_built() else ("python",)
        with redirected_output() as directory:
            summaries = performance_timing.run(performance_timing.SMALL_TIERS, 2,
                                               implementations)
            written = {path.name for path in directory.iterdir()}
            records = performance_timing.read_records(str(directory / "performance_timing.csv"))
            rebuilt = performance_timing.report(records, implementations, "rebuilt")
        for implementation in implementations:
            self.assertEqual(performance_timing.table_rows(rebuilt[implementation]),
                             performance_timing.table_rows(summaries[implementation]))
        for implementation in implementations:
            with self.subTest(implementation=implementation):
                self.assertIn(f"tab-dykvsacc-{implementation}.tex", written)
                self.assertIn(f"fig-performance-right-{implementation}.tex", written)
                for summary in summaries[implementation].values():
                    self.assertLessEqual(summary["accelerated"]["cycles"],
                                         summary["Dykstra"]["cycles"])
                    self.assertLessEqual(summary["accelerated"]["cycles"],
                                         summary["accelerated, certified"]["cycles"])
                    self.assertGreater(summary["Dykstra"]["seconds"], 0.0)
        self.assertIn("performance_timing.csv", written)


class SquareLineExampleTests(unittest.TestCase):
    def test_the_stall_and_its_length_match_plain_dykstra(self) -> None:
        results = square_line_example.analyse()
        state = results["state"]
        self.assertEqual(state["active"], [2, 3, 6])
        self.assertTrue(state["stationary"])
        np.testing.assert_allclose(state["beta"][[1, 2, 5]], [-0.2, 0.4, 1 / math.sqrt(5)])
        self.assertEqual((state["n_stall"], results["t_switch"]), (16, 17))
        self.assertEqual((results["repeated_through"], results["leaves"]), (16, 17))
        self.assertAlmostEqual(results["bound_first"], 18.5)
        self.assertEqual([o["n_stall"] for o in results["omegas"].values()], [6, 16, 46, 496])
        self.assertEqual([o["leaves"] for o in results["omegas"].values()], [7, 17, 47, 497])

    def test_the_tie_from_minus_four_needs_exact_arithmetic(self) -> None:
        results = square_line_example.analyse()
        np.testing.assert_allclose(results["ratios_4"], [15.0, 14.0])
        self.assertEqual((results["exact_ratio_4"], results["exact_leaves_4"],
                          results["exact_moved_4"]), (15, 16, 17))
        self.assertEqual((results["leaves_4"], results["moved_4"]), (17, 17))

    def test_the_line_first_order_and_the_accelerated_solver(self) -> None:
        results = square_line_example.analyse()
        np.testing.assert_allclose(results["bauschke_corner"], [-1.0, 1.0])
        self.assertEqual((results["bauschke"]["n_stall"], results["bauschke_count"]), (12, 12))
        np.testing.assert_allclose(results["line_foot"], [-3.36, 2.68])
        self.assertEqual((results["certified_at"], results["certificate"]), (2, "kkt"))
        np.testing.assert_allclose(results["certified_point"], [0.0, 1.0], atol=1e-12)
        self.assertEqual(results["trace_without_kkt"][:5],
                         ["run exactly 1", "evaluated 2", "jumped over 3-15", "evaluated 16",
                          "run exactly 17"])


if __name__ == "__main__":
    unittest.main()
