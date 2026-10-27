"""Regression coverage for the write-up figure and timing scripts."""

from contextlib import contextmanager
import importlib.util
import math
from pathlib import Path
import tempfile
import unittest

import numpy as np

from dykstra import paper_figures, performance_timing
from dykstra.paper_figures import (
    bounded_problem,
    corner,
    infeasible_start,
    phase_scalars,
    three_plane_arrangement,
)
from dykstra.lti_numerics import rigorous_deactivation_horizon
from dykstra.lti_solver import LTIVer5Solver
from dykstra import benchmark, c_solvers, c_validation, problem_generators, square_line_example
from dykstra.convex_projection_solver import ConvexProjectionSolver


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


@unittest.skipUnless(c_solvers.libraries_built(), "needs the C solvers: make -C C")
class CValidationTests(unittest.TestCase):
    def test_c_and_python_agree_to_rounding_on_a_reduced_battery(self) -> None:
        original = c_validation.TIERS, c_validation.SEEDS
        c_validation.TIERS, c_validation.SEEDS = ((12, 4), (20, 6)), 2
        try:
            with redirected_output():
                results = c_validation.run()
        finally:
            c_validation.TIERS, c_validation.SEEDS = original
        self.assertLess(results["worst"]["dykstra distance"], 1e-10)
        self.assertLess(results["worst"]["accelerated distance"], 1e-10)
        self.assertEqual(results["mismatched"], 0)
        self.assertEqual(results["certified"], 4)
        self.assertLess(results["certified_worst"], 1e-9)


class BenchmarkOptionTests(unittest.TestCase):
    def test_the_profiles_set_the_options_qpbenchmark_sets(self) -> None:
        self.assertEqual(benchmark.solver_options("osqp", "low accuracy"),
                         {"eps_abs": 1e-3, "eps_rel": 0.0, "time_limit": benchmark.TIME_LIMIT})
        self.assertEqual(benchmark.solver_options("quadprog", "high accuracy"), {})
        self.assertEqual(benchmark.solver_options("piqp", "mid accuracy")["check_duality_gap"],
                         True)
        self.assertEqual(benchmark.solver_options("scs", "default"),
                         {"time_limit_secs": benchmark.TIME_LIMIT})
        self.assertEqual(len(benchmark.TIERS) * benchmark.SEEDS, 45)
        self.assertTrue(all(n == 4 * p - 4 for n, p in benchmark.TIERS))


@unittest.skipUnless(HAS_QPSOLVERS and HAS_SCIPY, BENCHMARK_GROUP)
@unittest.skipUnless(c_solvers.libraries_built(), "needs the C solvers: make -C C")
class BenchmarkRunTests(unittest.TestCase):
    def test_a_reduced_benchmark_solves_every_problem_and_writes_the_tables(self) -> None:
        with redirected_output() as directory:
            records = benchmark.run(benchmark.TIERS[:1], 1, oracle=True)
            written = {path.name for path in directory.iterdir()}
        self.assertTrue({"benchmark.csv", "tab-sm-bench-tiers.tex", "tab-sm-bench-profiles.tex",
                         "tab-benchmark-dense.tex"} <= written)
        solvers = {r["solver"] for r in records}
        self.assertTrue(set(benchmark.QP_SOLVERS) | {"accelerated", "Dykstra", "oracle"} <= solvers)
        for record in records:
            with self.subTest(solver=record["solver"], profile=record["profile"]):
                self.assertTrue(record["solved"])
                self.assertLess(record["projection error"], 1e-2)
        for record in records:
            if record["solver"] in ("accelerated", "Dykstra"):
                self.assertLess(record["primal residual"], 1e-10)
                self.assertLess(record["projection error"], 1e-9)


from dykstra import activity_figures, activity_study
from dykstra.activity_study import Run, change_counts, change_cycles, exact_entries, record_runs
from dykstra.lti_numerics import cycle_map


ACTIVITY_FIGURES = ("half_space_activity_single", "half_space_activity_study",
                    "half_space_activity_changes", "activity_toggle_tail", "multi_toggle_example",
                    "activity_change_limit_2d", "activity_trend_final", "activity_trend_changes",
                    "am_eigs_study")


class ActivityRecorderTests(unittest.TestCase):
    def assert_records_match_exact_cycle(self, generator: int, n: int, p: int, horizon: int) -> None:
        seeds = [activity_study.study_seed(n, p, index) for index in range(4)]
        problems = [activity_study.draw_problem(generator, n, p, seed) for seed in seeds]
        for problem, run in zip(problems, record_runs(problems, horizon, seeds)):
            with self.subTest(n=n, p=p, seed=run.seed):
                # Continued to the horizon, a run stopped at a fixed point changes nothing more
                cycles, masks, _ = exact_entries(*problem, horizon)
                np.testing.assert_array_equal(run.cycles, cycles)
                np.testing.assert_array_equal(run.masks, masks)

    def test_the_batched_recorder_matches_exact_cycle_on_generator_2(self) -> None:
        self.assert_records_match_exact_cycle(2, 32, 2, 1500)
        self.assert_records_match_exact_cycle(2, 16, 4, 1500)

    @unittest.skipUnless(HAS_SCIPY, BENCHMARK_GROUP)
    def test_the_batched_recorder_matches_exact_cycle_on_generator_1(self) -> None:
        self.assert_records_match_exact_cycle(1, 6, 2, 1500)
        self.assert_records_match_exact_cycle(1, 10, 3, 1500)

    def test_exact_cycle_walks_the_core_solver_path(self) -> None:
        from dykstra.convex_projection_solver import DykstraProjectionSolver

        A, b, z = activity_study.draw_problem(2, 12, 3, activity_study.study_seed(12, 3, 0))
        _, _, points = exact_entries(A, b, z, 60)
        path = DykstraProjectionSolver(z, A, b, 60).solve().path
        np.testing.assert_allclose(path[:, -1], points[:, -1], rtol=0, atol=1e-12 * np.abs(z).max())

    def test_a_converged_run_stops_at_a_fixed_point_on_its_tight_set(self) -> None:
        problems = [activity_study.draw_problem(2, 4, 2, activity_study.study_seed(4, 2, k))
                    for k in range(3)]
        for run in record_runs(problems, 5000):
            with self.subTest(seed=run.seed):
                self.assertIsNotNone(run.fixed_at)
                self.assertTrue(run.converged)
                self.assertLessEqual(run.converged_at, run.cycles_run)
                np.testing.assert_array_equal(run.masks[-1], run.tight)

    def test_seeds_differ_where_the_tier_seeds_coincide(self) -> None:
        # problem_generators.problem_seed has no p, and its index runs into the next n at 1000
        self.assertEqual(problem_generators.problem_seed(4, 1000), problem_generators.problem_seed(5, 0))
        self.assertNotEqual(activity_study.study_seed(8, 4, 0), activity_study.study_seed(8, 2, 0))
        self.assertNotEqual(activity_study.study_seed(4, 2, 1000), activity_study.study_seed(5, 2, 0))
        seed = activity_study.study_seed(8, 4, 3)
        for drawn, expected in zip(activity_study.draw_problem(2, 8, 4, seed),
                                   problem_generators.problem(8, 4, seed)):
            np.testing.assert_array_equal(drawn, expected)


class ActivityCountTests(unittest.TestCase):
    # The supplement's example: active in cycle 1, inactive in cycles 2 to 29, active in cycles 30
    # to 68, inactive from cycle 69, beside a half-space active throughout
    RUN = Run(seed=0, cycles=np.array([0, 1, 2, 30, 69]),
              masks=np.array([[False, True], [True, True], [False, True], [True, True],
                              [False, True]]),
              cycles_run=100, fixed_at=100, converged=True, converged_at=90,
              tight=np.array([False, True]))

    def test_changes_count_from_cycle_2_unless_cycle_1_is_asked_for(self) -> None:
        np.testing.assert_array_equal(change_counts(self.RUN), [3, 0])
        np.testing.assert_array_equal(change_counts(self.RUN, opening=True), [4, 0])
        np.testing.assert_array_equal(change_cycles(self.RUN, 0), [2, 30, 69])
        np.testing.assert_array_equal(change_cycles(self.RUN, 0, opening=True), [1, 2, 30, 69])

    def test_the_gap_before_the_reactivation_spans_cycles_2_to_29(self) -> None:
        self.assertEqual(activity_figures.longest_gap(self.RUN, 0), 28)

    def test_an_activation_after_cycle_0_ends_a_wait_but_no_gap(self) -> None:
        # Active only in cycle 0, inactive in cycles 1 to 4, then active from cycle 5
        run = Run(seed=0, cycles=np.array([0, 1, 5]), masks=np.array([[True], [False], [True]]),
                  cycles_run=10, fixed_at=10, converged=True, converged_at=8,
                  tight=np.array([True]))
        self.assertEqual(activity_figures.longest_gap(run, 0), 0)
        self.assertEqual(activity_figures.longest_wait(run, 0), 4)
        self.assertEqual(activity_figures.longest_wait(self.RUN, 0), 28)

    def test_counts_and_cumulative_changes_hold_between_entries(self) -> None:
        cycles = np.array([1, 29, 30, 68, 69, 100])
        counts, cumulative = activity_study.sampled(self.RUN, cycles)
        np.testing.assert_array_equal(counts, [2, 1, 2, 2, 1, 1])
        np.testing.assert_array_equal(cumulative, [0, 1, 2, 2, 3, 3])
        _, cumulative = activity_study.sampled(self.RUN, cycles, opening=True)
        np.testing.assert_array_equal(cumulative, [1, 2, 3, 3, 4, 4])

    def test_percentiles_of_counts_are_counts(self) -> None:
        values = np.array([0, 0, 1, 1, 1, 2, 2, 3, 5, 8])
        np.testing.assert_array_equal(activity_figures.percentile(values, [5, 50, 95]), [0, 1, 8])


class CycleMapSpectrumTests(unittest.TestCase):
    def test_a_wedge_in_three_dimensions_contracts_at_the_squared_cosine(self) -> None:
        # Two active unit normals at angle theta span a plane of R^3, on which A_m is the product
        # of two projectors, with eigenvalues 0 and cos(theta)^2; the third eigenvalue of A_m is 1
        theta = 0.3
        unit_A = np.array([[1.0, 0.0, 0.0], [np.cos(theta), np.sin(theta), 0.0], [0.0, 0.0, 1.0]])
        unit_b = np.zeros(3)
        values, left_out = activity_study.span_eigenvalues(unit_A, unit_b,
                                                           np.array([True, True, False]))
        self.assertEqual(left_out, 1)
        np.testing.assert_allclose(np.sort(np.abs(values)), [0.0, np.cos(theta) ** 2], atol=1e-14)
        full = np.linalg.eigvals(cycle_map(unit_A, unit_b, (True, True, False)).A_m)
        np.testing.assert_allclose(np.sort(np.abs(full)), [0.0, np.cos(theta) ** 2, 1.0], atol=1e-14)

    def test_an_empty_episode_has_no_spectrum_on_a_span(self) -> None:
        values, left_out = activity_study.span_eigenvalues(np.eye(2), np.ones(2),
                                                           np.array([False, False]))
        self.assertEqual(values.size, 0)
        self.assertEqual(left_out, 2)


@unittest.skipUnless(HAS_SCIPY, BENCHMARK_GROUP)
class ActivityStudyOutputTests(unittest.TestCase):
    def test_a_quick_run_writes_the_nine_figures_and_the_numbers(self) -> None:
        with redirected_output() as directory:
            numbers = activity_study.run(quick=True, workers=1)
            written = {path.name for path in directory.iterdir()}
        expected = {name + suffix for name in ACTIVITY_FIGURES for suffix in (".png", ".pdf")}
        self.assertTrue(expected | {"activity_numbers.json"} <= written)
        rows = numbers["curves"] + numbers["limit"] + numbers["trend"] + [numbers["example"]]
        for row in rows:
            self.assertGreater(row["recorder_checks"], 0)
            self.assertEqual(row["recorder_check_failures"], 0)
        self.assertTrue(numbers["example"]["agrees_with_exact_cycle"])
        self.assertLess(numbers["single"]["core_path_difference"], 1e-12)
        self.assertLess(numbers["eigenvalues"]["largest_modulus"], 1.0)


from contextlib import redirect_stdout
import io

from dykstra import stall_angles, stall_grid
from dykstra.convex_projection_solver import DykstraProjectionSolver


STALL_FIGURES = ("stall_grid_frozen", "stall_grid_plateau", "stall_angle_cycles",
                 "stall_angle_stallcycles", "stall_angle_probability", "accelerated_stall_study")


def batch_record(z: np.ndarray, A: np.ndarray, b: np.ndarray, horizon: int) -> dict:
    """The stall_grid record of one problem run alone."""
    reference = ConvexProjectionSolver._find_optimal_solution(z, A, b)
    batch = stall_grid.run_batch(z[None], A[None], b[None], reference[None], horizon)
    return stall_grid.summarise_runs(batch, z.size)[0]


class StallDetectorTests(unittest.TestCase):
    def test_the_batch_run_tracks_the_core_solver_cycle_by_cycle(self) -> None:
        drawn = [problem_generators.problem(12, 4, stall_grid.grid_seed(12, 4, i))
                 for i in range(3)]
        A, b, z = (np.array([d[k] for d in drawn]) for k in range(3))
        reference = np.array([ConvexProjectionSolver._find_optimal_solution(z[i], A[i], b[i])
                              for i in range(3)])
        errors = stall_grid.run_batch(z, A, b, reference, 300)["errors"]
        for i in range(3):
            solver = DykstraProjectionSolver(z[i], A[i], b[i], 300, track_error=True)
            expected = np.asarray(solver.solve().squared_errors)
            np.testing.assert_allclose(errors[i], expected, rtol=0.0, atol=1e-12 * (1 + expected.max()))

    def test_the_square_and_line_stall_lasts_as_the_theorem_says_under_both_detectors(self) -> None:
        for line_first in (False, True):
            A, b = square_line_example.square_line_problem(line_first)
            states = square_line_example.dykstra_cycles(square_line_example.START, A, b, 1)
            n_stall = square_line_example.stalling_state(A, b, *states[1][:2])["n_stall"]
            record = batch_record(square_line_example.START, A, b, 40)
            for detector in stall_grid.DETECTORS:
                with self.subTest(line_first=line_first, detector=detector):
                    # cycle 1 ends in the stalling state, so cycles 2 to N_stall repeat it
                    self.assertEqual(record[f"stalling_cycles_{detector}"], n_stall - 1)
                    self.assertEqual(record[f"longest_period_rank_{detector}"], 2)

    def test_the_relative_detector_ignores_the_scale_and_an_untouched_coordinate(self) -> None:
        A, b = square_line_example.square_line_problem()
        z = square_line_example.START
        expected = batch_record(z, A, b, 40)["stalling_cycles_relative"]
        scaled = batch_record(1000.0 * z, A, 1000.0 * b, 40)["stalling_cycles_relative"]
        widened = batch_record(np.append(z, 1e6), np.hstack([A, np.zeros((len(b), 1))]), b,
                               40)["stalling_cycles_relative"]
        self.assertEqual((scaled, widened), (expected, expected))

    def test_the_two_approximate_stall_rules_part_on_a_small_repeat(self) -> None:
        errors = np.array([[1.0, 1.0, 1e-3, 1e-3 + 2e-11, 1e-12, 1e-12]])
        decimals = stall_grid.approximate_stall_flags(errors, "decimals")[0]
        relative = stall_grid.approximate_stall_flags(errors, "relative")[0]
        np.testing.assert_array_equal(decimals, [False, True, False, True, False, False])
        np.testing.assert_array_equal(relative, [False, True, False, False, False, False])
        with self.assertRaises(ValueError):
            stall_grid.approximate_stall_flags(errors, "rounded")

    def test_the_relative_rule_reproduces_stalled_errors(self) -> None:
        A, b = square_line_example.square_line_problem()
        z = square_line_example.START
        result = DykstraProjectionSolver(z, A, b, 40, track_error=True,
                                         min_error=stall_grid.CONVERGED_SQUARED_ERROR).solve()
        flags = stall_grid.approximate_stall_flags(np.asarray(result.squared_errors), "relative")
        np.testing.assert_array_equal(flags, ~np.isnan(np.asarray(result.stalled_errors)))
        self.assertTrue(flags.any())


class StallFamilyTests(unittest.TestCase):
    def test_the_corner_normals_meet_at_theta_and_the_projection_is_the_constructed_one(self) -> None:
        for family in stall_angles.FAMILIES:
            for p in stall_angles.ANGLE_DIMENSIONS:
                with self.subTest(family=family, p=p):
                    rng = np.random.RandomState(5)
                    geometry = stall_angles.BUILDERS[family](0.1, p, rng)
                    normals = stall_angles.corner_normals(geometry)
                    cosines = (normals @ normals.T)[np.triu_indices(len(normals), 1)]
                    np.testing.assert_allclose(cosines, np.cos(0.1), rtol=0.0, atol=1e-12)
                    self.assertEqual(np.linalg.matrix_rank(normals), 2 if family == "wedge" else p)
                    z, reference = stall_angles.canonical_start(geometry, rng)
                    self.assertTrue(np.all(normals @ (z - geometry.corner) > 0.0))
                    quadprog = ConvexProjectionSolver._find_optimal_solution(z, geometry.A, geometry.b)
                    np.testing.assert_allclose(quadprog, reference, rtol=0.0, atol=1e-12)

    def test_a_captured_start_is_one_that_projects_onto_every_corner_boundary(self) -> None:
        for family in stall_angles.FAMILIES:
            with self.subTest(family=family):
                rng = np.random.RandomState(6)
                geometry = stall_angles.BUILDERS[family](1.0, 3, rng)
                starts, captured, references = stall_angles.approaching_starts(geometry, 300, rng)
                rows = geometry.corner_rows
                for z, is_captured, reference in zip(starts, captured, references):
                    x = ConvexProjectionSolver._find_optimal_solution(z, geometry.A, geometry.b)
                    on_boundaries = np.abs(geometry.A[:rows] @ x - geometry.b[:rows]).max() < 1e-9
                    self.assertEqual(is_captured, on_boundaries)
                    if is_captured:
                        np.testing.assert_allclose(x, reference, rtol=0.0, atol=1e-12)
                self.assertTrue(captured.any())

    def test_the_wedge_capture_rate_is_theta_over_pi_plus_theta_in_any_dimension(self) -> None:
        for p in stall_angles.ANGLE_DIMENSIONS:
            rng = np.random.RandomState(7)
            geometry = stall_angles.wedge(0.3, p, rng)
            _, captured, _ = stall_angles.approaching_starts(geometry, 40000, rng)
            exact = 0.3 / (np.pi + 0.3)
            # four standard deviations of the binomial share
            self.assertLess(abs(captured.mean() - exact), 4 * math.sqrt(exact * (1 - exact) / 40000))

    def test_the_wedge_contracts_at_cos_theta_to_the_fourth_and_neither_family_stalls(self) -> None:
        theta_index = stall_angles.ANGLES.index(0.3)
        for family in stall_angles.FAMILIES:
            records = stall_angles.sweep_cell(family, 3, theta_index, 3, 300)
            for record in records:
                with self.subTest(family=family, index=record["index"]):
                    self.assertGreaterEqual(record["converged_at"], 0)
                    self.assertEqual(record["stalling_cycles_relative"], 0)
                    self.assertEqual(record["stalling_cycles_absolute"], 0)
                    if family == "wedge":
                        self.assertAlmostEqual(record["contraction"], np.cos(0.3) ** 4, places=6)

    def test_the_censored_median_is_censored_only_when_a_middle_run_is(self) -> None:
        self.assertEqual(stall_angles.censored_median(np.array([1, 2, 3, -1]), 50), (2.5, False))
        self.assertEqual(stall_angles.censored_median(np.array([3, -1, 5, -1]), 50), (50.0, True))


class AcceleratedStallTests(unittest.TestCase):
    def test_the_accelerated_solver_certifies_where_dykstra_is_in_a_slow_phase(self) -> None:
        use_c = c_solvers.libraries_built()
        record = stall_angles.accelerated_problem("vertex", 0, 0, 2000, use_c)
        self.assertGreater(record["dykstra_cycles"], 50)
        self.assertEqual(record["accelerated_cycles"], 2)
        self.assertEqual(record["accelerated_certificate"], "kkt")
        if use_c:
            self.assertEqual((record["dykstra_c_cycles"], record["accelerated_c_cycles"]),
                             (record["dykstra_cycles"], record["accelerated_cycles"]))


class StallStudyOutputTests(unittest.TestCase):
    def test_quick_runs_write_the_six_figures(self) -> None:
        with redirected_output() as directory, redirect_stdout(io.StringIO()):
            shape = (stall_grid.QUICK_HALF_SPACES, stall_grid.QUICK_DIMENSIONS)
            records = stall_grid.run_grid(*shape, stall_grid.QUICK_PROBLEMS,
                                          stall_grid.QUICK_HORIZON, workers=1)
            numbers = stall_grid.report(records, *shape, stall_grid.QUICK_HORIZON)
            angles, horizon = stall_angles.QUICK_ANGLE_INDICES, stall_angles.QUICK_HORIZON
            stall_angles.report_sweep(stall_angles.run_sweep(
                angles, stall_angles.QUICK_PROBLEMS, horizon, 1), horizon)
            stall_angles.report_capture(stall_angles.run_capture(
                angles, stall_angles.QUICK_SAMPLES, stall_angles.QUICK_SAMPLES_FIT, horizon, 1))
            stall_angles.report_accelerated(stall_angles.run_accelerated(
                stall_angles.QUICK_ACCELERATED_INDICES, stall_angles.QUICK_ACCELERATED_PROBLEMS,
                stall_angles.QUICK_CAP, 1), stall_angles.QUICK_CAP)
            written = {path.name for path in directory.iterdir()}
        expected = {name + suffix for name in STALL_FIGURES for suffix in (".png", ".pdf")}
        self.assertTrue(expected <= written)
        self.assertEqual(numbers["problems"], 4 * len(stall_grid.grid_cells(*shape)))
        self.assertEqual(numbers["spanning_relative"], numbers["stalled_relative"])


class WedgeFamilyTests(unittest.TestCase):
    def test_a_wedge_problem_is_the_sweep_s_wedge_and_canonical_start_from_one_stream(self) -> None:
        for theta, p in benchmark.WEDGE_TIERS:
            with self.subTest(theta=theta, p=p):
                family = problem_generators.WedgeTestPlanes(theta, p)
                A, b, z = family.problem(11)
                rng = np.random.RandomState(11)
                geometry = stall_angles.wedge(theta, p, rng)
                start, reference = stall_angles.canonical_start(geometry, rng)
                for drawn, expected in zip((A, b, z), (geometry.A, geometry.b, start)):
                    np.testing.assert_array_equal(drawn, expected)
                self.assertEqual(len(b), family.num_planes)
                quadprog = ConvexProjectionSolver._find_optimal_solution(z, A, b)
                np.testing.assert_allclose(quadprog, reference, rtol=0.0, atol=1e-12)

    def test_stall_angles_imports_the_shared_geometry_rather_than_redefining_it(self) -> None:
        for name in ("Geometry", "haar_rotation", "corner_normals", "canonical_start",
                     "START_DISTANCE", "BOX_HALF_WIDTH", "CORNER_HALF_WIDTH"):
            self.assertIs(getattr(stall_angles, name), getattr(problem_generators, name))


if __name__ == "__main__":
    unittest.main()
