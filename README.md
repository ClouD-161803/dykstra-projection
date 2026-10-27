# Dykstra Projection

This repository holds a Python implementation of Dykstra's algorithm for projecting a point onto the intersection of finitely many half-spaces. It also holds the accelerated solver of the paper below, in Python and in C, two experimental variants, 2-D visualisations, CSV export, and the scripts that reproduce every figure and table of the paper and its supplement.

## The paper

The paper is "Analysis of Dykstra's Algorithm for Polyhedra using Dynamical Systems Reformulation" by Riki Demma, Claudio Vestini, and Idris Kempf, typeset for a SIAM journal and accompanied by a supplement. Its LaTeX sources are kept outside this repository.

In the paper, Dykstra's iteration on a polyhedron is reformulated as a linear time-invariant (LTI) system between changes of the active set. The reformulation is used to analyse stalling and to derive an accelerated solver, which compacts each run of cycles with a constant active set. The stall detection solver of this repository implements the fast-forward of an earlier preprint by Claudio Vestini and Idris Kempf, "Fast-Forwarding Stalling in Dykstra's Algorithm", [arXiv:2511.18132](https://doi.org/10.48550/arXiv.2511.18132), 2025.

## Setup

The environment is defined in `pyproject.toml` and fixed in `uv.lock`, and it requires Python 3.10 to 3.13 and [uv](https://docs.astral.sh/uv/). From the repository root,

```bash
uv sync
```

creates `.venv` with numpy 2.1.1, matplotlib 3.9.2, and quadprog 0.1.12. The benchmark against general-purpose QP (quadratic programming) solvers and the activity study also depend on the dependency group named benchmark, installed with

```bash
uv sync --group benchmark
```

A plain `uv sync` removes that group from an existing `.venv` again, so keep the flag once you use it. Do not run the code under Python 3.14, for which numpy 2.1.1 does not have a wheel.

The code is a package, `dykstra/`, that is run from the repository root and not installed. Run its modules as `python -m dykstra.<module>` with the environment's interpreter, e.g. `.venv/bin/python` or `uv run python`.

## Quick start

From the repository root,

```bash
.venv/bin/python -m dykstra.main
```

projects `[-2.0, 1.4]` onto the intersection of a unit box and the line `x / 2 + y = 1`, prints the finite-iteration result and the reference solution, and opens a 2-D visualisation. A graphical matplotlib backend is required for the window.

## Defining a projection problem

Each row of `A` is a half-space normal, and the corresponding entry of `b` is its offset:

```text
A @ x <= b
```

The following, run from the repository root, projects a point onto a square:

```python
import numpy as np

from dykstra.convex_projection_solver import DykstraProjectionSolver

z = np.array([-2.0, 1.4])
A = np.array([
    [1.0, 0.0],   # x <= 1
    [-1.0, 0.0],  # x >= -1
    [0.0, 1.0],   # y <= 1
    [0.0, -1.0],  # y >= -1
])
b = np.array([1.0, 1.0, 1.0, 1.0])

solver = DykstraProjectionSolver(z, A, b, max_iter=30, track_error=True)
result = solver.solve()

print(result.projection)
print(solver.actual_projection)  # quadprog reference used for error tracking
```

The normals need not be of unit length. The solver infers the dimension from `z` and checks that every constraint has the same dimension. The numerics work in any dimension, while the visualisers are for 2-D problems.

All supplied constraints are retained. A constraint that contains the starting point can still determine the projection, so `delete_spaces=True` is kept only as a deprecated option that warns and does not remove constraints.

## Solvers and outputs

- `DykstraProjectionSolver` is the standard cyclic projection method.
- `DykstraMapHybridSolver` combines the method of alternating projections (MAP) with Dykstra's updates.
- `DykstraStallDetectionSolver` detects a stalling period and skips it in one step, reaching the state plain Dykstra reaches after it.

`solve()` returns a `ProjectionResult`, whose optional fields are populated by the corresponding flags.

- `track_error=True` records the squared error against the reference solution of quadprog, and classifies each cycle as converged, stalled, or neither.
- `plot_errors=True` records the correction vectors for quiver plots.
- `plot_active_halfspaces=True` records the activity of each constraint by cycle.

`Visualiser` uses a horizontal layout and scales to any number of activity traces. `VerticalVisualiser` combines the activity traces on one lower axis. `ComparisonVisualiser` draws two solvers' results side by side. `ResultExporter` writes solver metadata, constraints, paths, errors, and activity to a sectioned CSV file, and its loader restores the arrays to their cycle-by-constraint shape.

The quadprog reference is exact, while a finite number of Dykstra cycles generally is not. A result of plain Dykstra is therefore an approximation of the projection after the stated number of cycles.

## The accelerated solvers

`dykstra/lti_solver.py` holds the accelerated solvers. Between changes of the active half-space set Dykstra's iteration is an LTI system, so an episode of constant activity can be stepped through its cycle map, evaluated in closed form, or skipped, with one exact Dykstra cycle whenever the set changes. The accelerated solvers take the same `A @ x <= b` problem as the core solvers.

`LTISolver` runs one of five presets, each adding one technique to the one before so that their effect can be compared. `LTIVer1Solver` to `LTIVer5Solver` select them.

- `cycle_map` (`LTIVer1Solver`) steps every cycle through the episode's cycle map.
- `closed_form` (`LTIVer2Solver`) evaluates regular episodes in closed form and settles once the active set is proven final.
- `envelope` (`LTIVer3Solver`) also jumps the runs of cycles that an envelope bound certifies free of switches.
- `frozen_stall` (`LTIVer4Solver`) also fast-forwards stalling periods, in which only the auxiliaries move.
- `deflated_modal` (`LTIVer5Solver`), the accelerated solver of the paper, also scans singular episodes in their deflated modal form, and settles only on a point that passes the KKT (Karush–Kuhn–Tucker) test.

Every preset returns Dykstra's iterate after `max_iter` cycles unless it settles first, and records the same path, errors, and corrections as `DykstraProjectionSolver` up to that point. A settled result holds the limit from `result.settled_at` on, and `result.certificate` records how it was proven. `"kkt"` means the point passed the KKT conditions and is the projection, to within `1e-9` of the distance moved beyond the rounding of the data, which nearly dependent active normals amplify. `"finality"` means the active set was proven final, with drifts below rounding taken as zero, and the point is the fixed point of the last cycle map. `result.is_settled()` tells a settled limit from the last iterate.

```python
from dykstra.lti_solver import LTIVer5Solver

result = LTIVer5Solver(z, A, b, max_iter=30).solve()
result.projection, result.settled_at, result.certificate
```

The setting `schedule` decides at which episode entries `deflated_modal` tries the KKT test. `"work"`, the default, tries it at the first entry and afterwards only once the work done since the last failed test exceeds that test's work, `"every"` at every entry, `"doubling"` at entries 1, 2, 4, 8, and so on, and `"never"` not at all, so that the solver never settles. With `profile=True` a solver also leaves the time, the flops, and the counts of every stage of its last solve in `solver.profile`, without changing any output.

`dykstra/lti_numerics.py` holds the numerical pieces that do not depend on solver state. `dykstra/oracle.py` uses them too: it records the activity schedule of one Dykstra run and replays it with every change known in advance, for comparison. Each preset and the oracle has a runner that opens a window, e.g. `.venv/bin/python -m dykstra.run_lti_ver5` and `.venv/bin/python -m dykstra.run_oracle`.

## C solvers

`C/` holds two C sources without dependencies, for the same `A @ x <= b` problem: `accelerated.c`, the `deflated_modal` preset (`LTIVer5Solver`), and `dykstra.c`, plain Dykstra, the baseline it is timed against. The C solvers do not record a history, and each returns the final point, a status, a few counters, and the multipliers. `accelerated.c` returns Dykstra's iterate at the cycle budget unless it settles, and it settles only on a point that passes the KKT test. It follows `dykstra/lti_numerics.py` piece for piece, except that it scans a singular episode by iterating the contracting block rather than diagonalising it, estimates the condition number of `I - A_m`, factors the Lawson-Hanson solves incrementally, and drops every coordinate that is not touched by any constraint before the solve.

Build the two shared libraries in place, from the repository root:

```bash
make -C C                                              # gcc or clang
powershell -ExecutionPolicy Bypass -File C/build.ps1   # Windows, MSVC
```

Then call them through `dykstra/c_solvers.py`, which finds the libraries in `C/` whatever the working directory:

```python
from dykstra.c_solvers import accelerated_c_projection, dykstra_c_projection

result = accelerated_c_projection(z, A, b, max_iter, schedule="work")
result.projection, result.settled, result.cycles, result.multipliers
```

The C accelerated solver takes the same `schedule` as `LTIVer5Solver`, tries the KKT test at the same entries, and settles on the same cycle, which `tests/test_c_solvers.py` checks under all four values.

`tests/test_c_solvers.py` holds both to the contract of the Python accelerated solver and is skipped until the libraries are built. The suite has been run against the Makefile build on macOS arm64 with Apple clang. `C/build.ps1` builds the libraries with MSVC on Windows, and the current sources have not been tested with that build.

## Reproducing the paper's figures and tables

Every figure and table drawn from data in the paper and its supplement has a script in `dykstra/`, and each script runs from the repository root with one command, e.g.

```bash
.venv/bin/python -m dykstra.paper_figures
.venv/bin/python -m dykstra.performance_timing
```

The [command table in AGENTS.md](AGENTS.md#reproducing-the-figures-and-tables) gives the script, the command, and the output files for each figure and table. The scripts write into `results/paper` at the repository root, which git ignores, as PNG and PDF figures, TikZ or pgfplots code for the figures drawn inline in the paper, and LaTeX table rows. Each script prints every number quoted in the paper from its outputs, and takes `--quick` for a short run into a temporary directory. The timing scripts print the machine, the compiler, the package versions, and the method, and their timings are those of the machine they run on. `benchmark.py`, `activity_study.py`, and `singular_phases.py` depend on the benchmark group, and the C libraries must be built before `performance_timing.py`, `benchmark.py`, `c_validation.py`, `cost_model.py`, `compaction_only.py`, and `oracle_comparison.py` are run in full. `benchmark.py --family all` runs the benchmark on the dense, wedge, and stall families and writes the headline table with all three.

## Tests

From the repository root,

```bash
.venv/bin/python -m unittest discover -s tests
```

runs 268 tests. With the C libraries built and the benchmark group installed, the suite skips none of them.

## Repository layout

```text
.
├── README.md
├── AGENTS.md                    # operating guide for coding agents, with the command table
├── ARCHITECTURE.md              # how the code is put together
├── pyproject.toml, uv.lock      # the environment
├── dykstra/                     # the package, run as python -m dykstra.<module>
│   ├── convex_projection_solver.py  # the three core solvers
│   ├── projection_result.py
│   ├── gradient.py              # the quadprog reference projection
│   ├── visualiser.py            # visualisers and CSV export
│   ├── edge_rounder.py
│   ├── lti_solver.py            # LTISolver and its five presets
│   ├── lti_numerics.py          # cycle maps, closed forms, the KKT test
│   ├── oracle.py                # activity-schedule replay
│   ├── c_solvers.py             # ctypes access to the C solvers
│   ├── problem_generators.py    # the supplement's two generators and the wedge and stall families
│   ├── paper_figures.py, performance_timing.py, benchmark.py, cost_model.py, compaction_only.py,
│   ├── square_line_example.py, c_validation.py, stall_grid.py, stall_angles.py,
│   ├── activity_study.py, activity_figures.py, activity_conjecture.py,
│   ├── singular_phases.py, oracle_comparison.py
│   │                            # the figure and table scripts
│   ├── main.py, paper_figure.py # interactive 2-D examples
│   └── lti_examples.py, run_*.py
├── C/                           # accelerated.c, dykstra.c, and their build files
├── tests/                       # unittest suite
└── results/                     # written by the scripts, ignored by git
```

Everything removed from the tree remains in git history, i.e. the earlier report and paper sources in `Latex/`, the development versions in `Python/Previous Versions/`, the superseded implementations in `Python/Working Version/bin/`, the recorded CSV files, and `requirements.txt`.

The reference papers and the grant-application material are kept outside this repository. They are third-party or personal documents that are not ours to redistribute, and both folder names are gitignored. The paper's bibliography is the list of references.

## Author and funding

The repository's author is Claudio Vestini, University of Oxford. The project was funded by Keble Research Grant KSRG118.
