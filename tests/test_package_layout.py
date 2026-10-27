"""The layout of the dykstra package: its modules import from the repository root with no
entry added to sys.path, the C libraries and the generated outputs are found beside the package
whatever the working directory, and every figure and table script runs under -m with --quick."""

from concurrent.futures import ThreadPoolExecutor
import importlib.util
import os
from pathlib import Path
import pkgutil
import subprocess
import sys
import tempfile
import unittest

import numpy as np

import dykstra
from dykstra import c_solvers


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
MODULES = sorted(module.name for module in pkgutil.iter_modules(dykstra.__path__))
LIBRARY_SUFFIX = {"win32": ".dll", "darwin": ".dylib"}.get(sys.platform, ".so")
C_LIBRARIES_BUILT = all((REPOSITORY_ROOT / "C" / f"{name}{LIBRARY_SUFFIX}").exists()
                        for name in ("accelerated", "dykstra"))
HAS_BENCHMARK_GROUP = all(importlib.util.find_spec(name) is not None
                          for name in ("qpsolvers", "scipy"))

# Imports the modules named on its command line and prints the backend matplotlib ended with.
# It fails if any import added to or removed from sys.path
IMPORT_EVERY_MODULE = """
import importlib, sys
before = list(sys.path)
import matplotlib
for name in sys.argv[1:]:
    importlib.import_module("dykstra." + name)
assert sys.path == before, sorted(set(sys.path) ^ set(before))
print(matplotlib.get_backend())
"""

PRINT_OUTPUT_DIRECTORIES = """
from dykstra import c_solvers, paper_figure, paper_figures
print(paper_figures.OUTPUT_DIR)
print(paper_figure.OUTPUT_DIR)
print(c_solvers.C_FOLDER)
"""

# Every figure and table script of the paper and the supplement, the flags of its quick run,
# and the files that run must write
QUICK_RUNS = {
    "paper_figures": (("--quick",), (
        "fig-vertexcorner.tex", "fig-ykdrift.tex", "fig-envelopes.tex", "fig-performance-left.tex",
        "vertex_gamma_corner.pdf", "triple_loop_drift.pdf", "envelope_bounds.pdf",
        "performance_error.pdf", "performance_error.png")),
    "performance_timing": (("--quick",), (
        "performance_timing.csv", "tab-dykvsacc-python.tex", "fig-performance-right-python.tex",
        "performance_scaling_python.pdf")),
    "benchmark": (("--quick",), (
        "benchmark.csv", "tab-benchmark-dense.tex", "tab-sm-bench-tiers.tex",
        "tab-sm-bench-profiles.tex")),
    "c_validation": (("--quick",), ("tab-c-validation.tex",)),
    "square_line_example": (("--quick",), ("tab-square-line.tex",)),
    "stall_grid": (("--quick", "--workers", "1"), (
        "stall_grid_frozen.pdf", "stall_grid_plateau.pdf", "stall_grid_problems.csv")),
    "stall_angles": (("--quick", "--workers", "1"), (
        "stall_angle_cycles.pdf", "stall_angle_stallcycles.pdf", "stall_angle_probability.pdf",
        "accelerated_stall_study.pdf")),
    "activity_study": (("--quick", "--workers", "1"), (
        "half_space_activity_single.pdf", "half_space_activity_study.pdf",
        "half_space_activity_changes.pdf", "activity_toggle_tail.pdf",
        "multi_toggle_example.pdf", "activity_change_limit_2d.pdf", "activity_trend_final.pdf",
        "activity_trend_changes.pdf", "am_eigs_study.pdf", "activity_numbers.json")),
}


def missing_requirement(module: str) -> str | None:
    """Why the quick run of a script cannot run here, or None."""
    if module in ("benchmark", "activity_study") and not HAS_BENCHMARK_GROUP:
        return "needs the benchmark dependency group: uv sync --group benchmark"
    if module in ("benchmark", "c_validation") and not C_LIBRARIES_BUILT:
        return "needs the C solvers: make -C C"
    return None


def run_python(arguments: list[str], cwd: Path, **environment: str) -> subprocess.CompletedProcess:
    """Run the interpreter of this test in cwd under the Agg backend and without PYTHONPATH,
    unless environment sets it."""
    env = {key: value for key, value in os.environ.items() if key != "PYTHONPATH"}
    env.update(MPLBACKEND="Agg", **environment)
    return subprocess.run([sys.executable, *arguments], cwd=cwd, env=env, capture_output=True,
                          text=True, timeout=300)


def file_times(directory: Path) -> dict:
    """The modification time of every file under a directory."""
    return {path: path.stat().st_mtime_ns for path in directory.rglob("*") if path.is_file()}


class PackageImportTests(unittest.TestCase):
    def test_the_package_names_the_repository_it_sits_in(self) -> None:
        self.assertEqual(dykstra.REPOSITORY_ROOT, REPOSITORY_ROOT)

    def test_every_module_imports_headlessly_without_touching_sys_path(self) -> None:
        self.assertIn("convex_projection_solver", MODULES)
        completed = run_python(["-c", IMPORT_EVERY_MODULE, *MODULES], REPOSITORY_ROOT)
        self.assertEqual(completed.returncode, 0, completed.stderr)
        self.assertEqual(completed.stdout.split()[-1].lower(), "agg")

    def test_no_module_or_test_file_adds_to_sys_path(self) -> None:
        insertion = "sys.path." + "insert"
        sources = sorted(Path(dykstra.__file__).parent.glob("*.py"))
        for path in sources + sorted(Path(__file__).parent.glob("test_*.py")):
            with self.subTest(file=path.name):
                self.assertNotIn(insertion, path.read_text(encoding="utf-8"))


class CLibraryLocationTests(unittest.TestCase):
    def test_the_libraries_are_looked_up_in_the_c_folder_beside_the_package(self) -> None:
        for name in ("accelerated", "dykstra"):
            self.assertEqual(c_solvers.library_path(name),
                             REPOSITORY_ROOT / "C" / f"{name}{LIBRARY_SUFFIX}")

    @unittest.skipUnless(C_LIBRARIES_BUILT, "needs the C solvers: make -C C")
    def test_both_libraries_load_and_project(self) -> None:
        self.assertTrue(c_solvers.libraries_built())
        z, A, b = np.array([3.0, 1.0]), np.array([[1.0, 0.0], [0.0, 1.0]]), np.array([1.0, 2.0])
        for solve in (c_solvers.accelerated_c_projection, c_solvers.dykstra_c_projection):
            with self.subTest(solver=solve.__name__):
                np.testing.assert_allclose(solve(z, A, b, 10).projection, [1.0, 1.0])


class OutputDirectoryTests(unittest.TestCase):
    def test_outputs_and_libraries_resolve_beside_the_package_from_another_directory(self) -> None:
        with tempfile.TemporaryDirectory() as elsewhere:
            completed = run_python(["-c", PRINT_OUTPUT_DIRECTORIES], Path(elsewhere),
                                   PYTHONPATH=str(REPOSITORY_ROOT))
        self.assertEqual(completed.returncode, 0, completed.stderr)
        self.assertEqual(completed.stdout.splitlines()[-3:],
                         [str(REPOSITORY_ROOT / "results" / "paper"),
                          str(REPOSITORY_ROOT / "results"), str(REPOSITORY_ROOT / "C")])


class QuickRunTests(unittest.TestCase):
    def test_every_figure_and_table_script_runs_quickly_into_a_temporary_directory(self) -> None:
        """Each quick run creates its output directory with tempfile, here under a TMPDIR of its
        own, and leaves results/ at the repository root as it was."""
        runnable = {module: flags for module, flags in QUICK_RUNS.items()
                    if missing_requirement(module) is None}
        if "performance_timing" in runnable and not C_LIBRARIES_BUILT:
            flags, written = runnable["performance_timing"]
            runnable["performance_timing"] = (flags + ("--implementation", "python"), written)
        results_before = file_times(REPOSITORY_ROOT / "results")
        with tempfile.TemporaryDirectory() as scratch:
            directories = {module: Path(scratch) / module for module in runnable}
            for directory in directories.values():
                directory.mkdir()

            def quick_run(module: str) -> subprocess.CompletedProcess:
                return run_python(["-m", f"dykstra.{module}", *runnable[module][0]],
                                  REPOSITORY_ROOT, TMPDIR=str(directories[module]))

            with ThreadPoolExecutor(max_workers=len(runnable)) as pool:
                completed = dict(zip(runnable, pool.map(quick_run, runnable)))
            written = {module: {path.name for path in directory.rglob("*") if path.is_file()}
                       for module, directory in directories.items()}
        for module in QUICK_RUNS:
            with self.subTest(script=module):
                if module not in runnable:
                    self.skipTest(missing_requirement(module))
                self.assertEqual(completed[module].returncode, 0, completed[module].stderr)
                self.assertLessEqual(set(runnable[module][1]), written[module])
        self.assertEqual(file_times(REPOSITORY_ROOT / "results"), results_before)


if __name__ == "__main__":
    unittest.main()
