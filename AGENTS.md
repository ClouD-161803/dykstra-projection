# AGENTS.md

This file holds the instructions for any coding agent working in this repository. `ARCHITECTURE.md`, beside it, is the longer explanation of how the code is put together. Read it before changing anything in `dykstra/`.

## What this is

This repository holds a Python implementation of Dykstra's algorithm for projecting a point onto the intersection of finitely many half-spaces, with two experimental variants. Beside it are the accelerated solver of the paper, called the LTI (linear time-invariant) solvers in the code, C versions of that solver and of plain Dykstra, 2-D visualisers, CSV export, and the scripts that rebuild every figure and table of the paper and its supplement. It is research code behind a paper, not a library. `dykstra/` is a package only so that its modules import one another by absolute name. It is not installed, and it does not have a public API to keep stable.

The paper is "Analysis of Dykstra's Algorithm for Polyhedra using Dynamical Systems Reformulation" by Riki Demma, Claudio Vestini, and Idris Kempf, typeset for a SIAM journal with a supplement. Its LaTeX sources are kept outside this repository. On the author's machine they sit beside the checkout in `../paper/`.

The repository's author is Claudio Vestini, University of Oxford, and the project was funded by Keble Research Grant KSRG118.

The repository is **public** on GitHub, as `ClouD-161803/dykstra-projection`. Everything committed here is world-readable, hence do not add credentials, personal documents, or third-party PDFs.

## Problem convention

Constraints are rows of `A` with offsets `b`, meaning `A @ x <= b`. Each row of `A` is a half-space normal, not necessarily of unit length, since the solvers normalise internally. The dimension is inferred from the point `z`, and every constraint is validated against it. The numerics work in any dimension, while the visualisers are 2-D only.

`delete_spaces=True` is a deprecated no-op kept for compatibility, and it warns. Every supplied constraint is retained, because a constraint that already contains the starting point can still determine the projection.

## Layout

```
dykstra/                         the package, run from the repository root as python -m dykstra.<module>
  __init__.py                    REPOSITORY_ROOT, the folder that holds dykstra/, C/ and results/
  convex_projection_solver.py    ConvexProjectionSolver (abstract), the three core solvers, dykstra_projection()
  projection_result.py           ProjectionResult, returned by every solve()
  gradient.py                    quadprog_solve_qp, the reference projection by quadratic programming
  visualiser.py                  ResultExporter, Visualiser, VerticalVisualiser, ComparisonVisualiser
  edge_rounder.py                rounded_box_constraints, polygonal corner rounding
  lti_solver.py                  LTISolver, its five presets, LTIVer1Solver to LTIVer5Solver
  lti_numerics.py                cycle maps, closed forms, envelope horizons, the KKT certificate
  oracle.py                      replays a recorded activity schedule, for comparison
  c_solvers.py                   ctypes access to the C solvers
  problem_generators.py          Generators 1 and 2 of the supplement and the wedge and stall families, seeded
  paper_figures.py               four figures of the paper, and the output folder, palette and save helpers
  performance_timing.py          tab:dykvsacc and fig:performance, solvers timed to a fixed accuracy
  benchmark.py                   tab:benchmark and the supplement's two benchmark tables, against QP solvers
  cost_model.py                  the cost of an episode against a Dykstra cycle, in Python and in C
  compaction_only.py             the accelerated solver without its KKT test, and the stall-length check
  square_line_example.py         the square-and-line stalling example of the paper
  c_validation.py                the supplement's comparison of the C and Python solvers
  stall_grid.py, stall_angles.py the supplement's stall studies, and the grids of fig:stallfreq
  activity_study.py              the supplement's activity study, drawn by activity_figures.py
  activity_conjecture.py         the activity-change study of thm:threetwo, on three half-planes
  singular_phases.py             the phases whose active normals do not span R^p, and the norm of T
  oracle_comparison.py           the oracle against the Python accelerated solver, in time and work
  main.py, paper_figure.py       interactive 2-D examples, which open a window
  lti_examples.py, run_*.py      the shared LTI example and one runner per preset and the oracle
tests/                           nine unittest files, and core_solver_reference.npz for the bitwise test
C/                               accelerated.c (the deflated_modal preset) and dykstra.c, with a Makefile and build.ps1
results/                         written by the scripts and ignored by git
pyproject.toml, uv.lock          the environment
```

The KKT (Karush–Kuhn–Tucker) certificate is the optimality test that lets an accelerated solve stop at the projection, and a QP is the quadratic program of projecting `z`, which the general-purpose solvers of `benchmark.py` solve. An episode is a run of cycles with a constant active set, which the accelerated solver compacts. The paper calls it a phase, and the code and these files call it an episode.

The reference papers and the grant-application material are kept outside the repository, because it is public and neither the publishers' PDFs nor the department's documents are ours to redistribute. `Books & Articles/` and `Application/` are listed in `.gitignore`, so that a restored copy cannot be committed. Do not restore them. On the author's machine they sit beside the checkout in `../materials/`. Cite papers from the paper's bibliography instead of adding a PDF.

## Environment

`pyproject.toml` declares Python `>= 3.10, < 3.14` and fixes `numpy==2.1.1`, `matplotlib==3.9.2`, and `quadprog==0.1.12`, and `uv.lock` fixes the version of every package. Both are committed. From the repository root,

```bash
uv sync                      # creates .venv with the fixed versions
uv sync --group benchmark    # the same, plus the dependency group named benchmark
```

The benchmark group holds qpsolvers, the nine QP solvers that `benchmark.py` compares beside quadprog, and scipy, which Generator 1 calls for its linear programs. `benchmark.py` and `activity_study.py` import from the group, and the tests that depend on it are skipped without it. A plain `uv sync` removes the group's packages from an existing `.venv`, since uv makes the environment match the groups asked for. Once you use the group, sync with `uv sync --group benchmark` every time.

On the author's machine the system Python is 3.9 and Homebrew's is 3.14, both outside the range, and the interpreter in `.venv` is a uv-managed 3.12.

Do not run this code under Python 3.14, or under any interpreter outside `.venv`. A wheel of `numpy==2.1.1` does not exist for 3.14, so such an interpreter imports whatever numpy it has, and results are then produced against dependencies of unknown versions. A `__pycache__/*.cpython-314.pyc` under `dykstra/` or `tests/` is evidence of such a run. Ignore it, and do not take it as proof of a working environment.

`gradient.py` imports `quadprog` at module scope, so `convex_projection_solver` and every module built on it fail to import without it.

## Running things

```bash
# the tests, from the repository root
.venv/bin/python -m unittest discover -s tests       # 268 tests

# one test class
.venv/bin/python -m unittest tests.test_package_layout.QuickRunTests

# the C libraries, built in place in C/ (the binaries are ignored by git)
make -C C                                              # gcc or clang
powershell -ExecutionPolicy Bypass -File C/build.ps1   # Windows, MSVC

# the interactive example (needs a GUI matplotlib backend)
.venv/bin/python -m dykstra.main
```

Every command runs from the repository root. `python -m` puts the working directory on `sys.path`, which is where the `dykstra` package is found, and discovery imports the test files from `tests/` without a `tests/__init__.py`. `sys.path` is not edited by any module or test file, and `tests/test_package_layout.py` fails if one does. The repository does not contain a `conftest.py` or a pytest configuration, and plain `pytest` does not find the package, because it puts `tests/` rather than the root on `sys.path`.

With the C libraries built and the benchmark group installed, the suite runs 268 tests in about 40 s on an Apple M5 Pro and skips none of them. Without the libraries the tests that call the C solvers are skipped, and without the group the tests of Generator 1, the benchmark, the activity study, and the quick run of `singular_phases.py`. The bitwise comparison of the core solvers is skipped on any platform but the one its fixture was recorded on.

The suite is headless. `tests/test_visualiser_regressions.py` selects the Agg backend before importing pyplot, `paper_figures.py` selects it on import, hence so does every script that imports it, and the quick runs start with `MPLBACKEND=Agg`. `main.py`, `paper_figure.py`, and the `run_*.py` runners open a window.

Every output of a script goes under `results/` at the repository root, whatever the working directory. The figure and table scripts write to `results/paper`, and `paper_figure.py` writes its two CSV files to `results/`. Git ignores `results/`, so its files are not records, and every one of them can be regenerated with the commands below.

### Reproducing the figures and tables

Each command runs from the repository root and writes into `results/paper`. A figure is written as a PNG and a PDF, a figure that the paper draws with TikZ or pgfplots is also written as that code with its coordinates, `fig-<label>.tex`, and table rows are written as `tab-<label>.tex`. The article includes no PDF: fig:stallfreq is drawn from `fig-stallgrid-stationary.tex` and fig:performance from the `fig-performance` files, and the supplement draws fig:vertexcorner, fig:ykdrift, and fig:envelopes the same way. The supplement includes thirteen PDFs: the nine activity figures, `stall_angle_cycles`, `stall_angle_stallcycles`, `stall_angle_probability`, and `accelerated_stall_study`. Every other PNG and PDF, among them `stall_grid_frozen`, `stall_grid_plateau`, `accelerated_stall_study_distance10`, and the PNG and PDF of each figure drawn from a `.tex` file, is written for checking and is not included by either document. Each script prints every number quoted in the paper from its outputs.

| figure or table | script | command | outputs in `results/paper` |
|---|---|---|---|
| fig:vertexcorner, fig:ykdrift, fig:envelopes, and the left panel of fig:performance | `paper_figures.py` | `.venv/bin/python -m dykstra.paper_figures` | `fig-vertexcorner.tex`, `fig-ykdrift.tex`, `fig-envelopes.tex`, `fig-performance-left.tex`, and `vertex_gamma_corner`, `triple_loop_drift`, `envelope_bounds`, `performance_error` as PNG and PDF |
| tab:dykvsacc, and the right panel and the whole of fig:performance | `performance_timing.py` | `make -C C` once, then `.venv/bin/python -m dykstra.performance_timing` | `tab-dykvsacc-c.tex`, `tab-dykvsacc-python.tex`, `fig-performance-right-c.tex`, `fig-performance-right-python.tex`, `fig-performance-c.tex`, `fig-performance-python.tex`, `performance_scaling_c` and `performance_scaling_python` as PNG and PDF, and `performance_timing.csv` |
| the comparison of the schedules of the KKT test | `performance_timing.py` | `make -C C` once, then `.venv/bin/python -m dykstra.performance_timing --schedule S` for S in `work`, `every`, and `doubling`, optionally each again with `--family wedge`, then `.venv/bin/python -m dykstra.performance_timing --compare-schedules` | `tab-kkt-schedules.tex` and `kkt_schedules.csv`, beside the outputs of each run under names that end in its family and schedule, e.g. `tab-dykvsacc-c-every.tex` |
| tab:benchmark with its dense, wedge, and stall columns, tab:sm-bench-tiers, and tab:sm-bench-profiles | `benchmark.py` | `uv sync --group benchmark` and `make -C C` once, then `.venv/bin/python -m dykstra.benchmark --family all` | `tab-benchmark.tex`, the headline with the three families as column groups, and per family its headline, tier and profile tables, and records: `tab-benchmark-dense.tex`, `tab-sm-bench-tiers.tex`, `tab-sm-bench-profiles.tex`, and `benchmark.csv` for the dense family, and `tab-benchmark-<family>.tex`, `tab-sm-bench-tiers-<family>.tex`, `tab-sm-bench-profiles-<family>.tex`, and `benchmark-<family>.csv` for the wedge and the stall family |
| the cost model of the paper's section on the accelerated algorithm, i.e. the cost of an episode against a Dykstra cycle | `cost_model.py` | `make -C C` once, then `.venv/bin/python -m dykstra.cost_model` | `tab-cost-model.tex`, `cost_model.csv`, `cost_model_problems.csv`, `cost_model_kkt.csv`, and `cost_model_fits.csv` |
| the compaction-only table of the same section, i.e. the accelerated solver under the schedule `never`, and the stall-length check | `compaction_only.py` | `make -C C` once, then `.venv/bin/python -m dykstra.compaction_only` | `tab-compaction-only.tex`, `compaction_only.csv`, `tab-stall-length.tex`, and `stall_length.csv` |
| the square-and-line example of the paper's section on the length of a stalling period | `square_line_example.py` | `.venv/bin/python -m dykstra.square_line_example` | `tab-square-line.tex` |
| the C-implementation validation paragraph of the supplement's benchmark section | `c_validation.py` | `make -C C` once, then `.venv/bin/python -m dykstra.c_validation` | `tab-c-validation.tex` |
| fig:stallfreq | `stall_grid.py` | `.venv/bin/python -m dykstra.stall_grid` | `stall_grid_frozen` and `stall_grid_plateau` as PNG and PDF, the pgfplots matrix plots `fig-stallgrid-stationary.tex`, which fig:stallfreq draws, `fig-stallgrid-approximate.tex`, and `fig-stallgrid-approximate-tendecimal.tex`, five `stall_grid_*.csv` files of records, and six of n-by-p grids |
| fig:sm-angle-cycles, fig:sm-angle-stallcycles, fig:sm-angle-capture, and fig:sm-acc-stall | `stall_angles.py` | `.venv/bin/python -m dykstra.stall_angles` | `stall_angle_cycles`, `stall_angle_stallcycles`, `stall_angle_probability`, `accelerated_stall_study`, and `accelerated_stall_study_distance10` as PNG and PDF, `tab-angle-numbers.tex`, and five CSV files |
| the nine figures of the supplement's activity study, fig:activity_single to fig:am_eigs | `activity_study.py` | `uv sync --group benchmark` once, then `.venv/bin/python -m dykstra.activity_study` | `half_space_activity_single`, `half_space_activity_study`, `half_space_activity_changes`, `activity_toggle_tail`, `multi_toggle_example`, `activity_change_limit_2d`, `activity_trend_final`, `activity_trend_changes`, and `am_eigs_study` as PNG and PDF, and `activity_numbers.json` |
| the activity-change study of thm:threetwo, its maxima and the two exact counterexamples | `activity_conjecture.py` | `.venv/bin/python -m dykstra.activity_conjecture` | `tab-conjecture.tex` and `conjecture.csv` |
| the singular phases, those whose active normals do not span R^p, with their shares of the phases and the cycles, and the largest norm and spectral radius of T = Q^T A_m Q, the cycle map on the span of those normals, which sec:sm-coverage reports, over the activity study's curve battery and the dense tiers of tab:dykvsacc | `singular_phases.py` | `uv sync --group benchmark` once, then `.venv/bin/python -m dykstra.singular_phases` | `tab-singular-phases.tex` and `singular_phases.csv` |
| the oracle against the Python accelerated solver of subsec:results-oracle, in time and in the work of each, on the dense problems of `benchmark.py` | `oracle_comparison.py` | `make -C C` once, then `.venv/bin/python -m dykstra.oracle_comparison` | `tab-oracle-comparison.tex` and `oracle_comparison.csv` |

tab:coverage is written by hand in the paper, hence a script does not exist for it. `paper_figure.py` draws the comparison figure of an earlier report, which is not included in the SIAM paper. It writes `Dykstra_30_iterations.csv` and `Algorithm (1)_30_iterations.csv` to `results/` and opens a window.

Each script in the table takes `--quick`, which runs small sizes into a temporary directory in a few seconds and leaves `results/` alone. `QuickRunTests` in `tests/test_package_layout.py` runs eight of them that way under `python -m`, and `CostModelTests`, `CompactionOnlyTests`, `ActivityConjectureTests`, `SingularPhasesTests`, and `OracleComparisonTests` in `tests/test_paper_outputs.py` run the quick sizes of the other five in the test process. The stall, activity, compaction-only, and singular-phase scripts also take `--workers`. `performance_timing.py` takes `--implementation python` when the C libraries are not built, `--family wedge` to time the wedge family instead of the dense tiers, `--schedule` to set where the accelerated solver tries the KKT test, and `--from-csv` to rebuild its outputs from `performance_timing.csv` without timing again. A run of one implementation keeps the records of the other in the records file, which `--from-csv` and `--compare-schedules` read beside its own. A run of another family or schedule than the paper's writes its outputs under names that end in them, and leaves the paper's names alone. `benchmark.py` takes `--family dense`, the default, `wedge`, `stall`, or `all`, and `--from-csv`, and only a run of all three writes `tab-benchmark.tex`. `cost_model.py` also takes `--from-csv`. `stall_angles.py` adds the C cycle counts to its accelerated study only when the libraries are built, `cost_model.py` and `oracle_comparison.py` stop without them, and `singular_phases.py` stops without the benchmark group. Every `.tex` output opens with a comment giving the command that generated it.

On an Apple M5 Pro a full run takes about 22 min for `cost_model`, 19 min for `performance_timing`, 3 min for `activity_study` and for `activity_conjecture` on ten worker processes, 3 min for `benchmark --family all`, 2 min for `oracle_comparison`, 48 s for `compaction_only` on four worker processes, 17 s for `paper_figures`, 16 s for `stall_angles`, 11 s for `singular_phases` on ten worker processes, and under 10 s for each of the others. The runs of `performance_timing` under the schedules `every` and `doubling` take about 29 and 16 min, and with `--implementation python` each of the three runs takes about a minute less. `performance_timing.py`, `benchmark.py`, `cost_model.py`, and `oracle_comparison.py` print the machine, the compiler, the package versions, and the timing method before their results, and their timings are those of the machine they run on.

## Conventions

- British spelling in new names and prose: `visualiser`, `normalise`. Some existing identifiers use the `-ize` form (`_initialize_iteration`), and you should match the file you are in rather than rename.
- Solvers subclass `ConvexProjectionSolver` and override `_update_error`, `solve`, and `_format_output`. Shared geometry (`_normalise`, `_is_in_half_space`, `_project_onto_half_space`, `_project_onto_unit_half_space`, `_validate_problem`, `_beta_check`) is defined on the base class as static methods, and new shared geometry goes there, not into a subclass. Every `solve()` starts with `_reset_state()`, so that a second call starts afresh and leaves the first result alone.
- `solve()` always returns a `ProjectionResult`. Its optional fields are populated by the constructor flags `track_error`, `plot_errors`, and `plot_active_halfspaces`, except `settled_at` and `certificate`, which an LTI solver sets when it settles. Test for them with the `has_*` predicates and `is_settled()`, not by comparing to `None` at the call site.
- `track_error=True` compares against the `quadprog` QP solution. That reference is exact, and a finite number of Dykstra cycles generally is not. Never call the Dykstra output "the projection" in a paper claim without saying how many cycles produced it.
- An LTI solver returns Dykstra's iterate at the budget unless `result.is_settled()`. Only `result.certificate == "kkt"` makes the result the projection, to within `1e-9` of the distance moved beyond the rounding of the data, which nearly dependent active normals amplify. A `"finality"` limit takes drifts below rounding as zero. Say which one a paper claim rests on.
- Every random problem comes from `problem_generators.py` and a seed, which names the same problem on every installation, since `problem` and the `problem` methods of the two families seed a legacy `RandomState`. The dense tiers are Generator 2, `problem(num_planes, num_dimensions, seed)` with the seed `problem_seed(num_planes, index)`, i.e. 100000 + 1000 times the half-spaces plus the index. The wedge family is `WedgeTestPlanes(theta, p).problem(seed)`, two half-spaces whose normals meet at angle `theta` and a box on the other `p - 2` coordinates, on which Dykstra's algorithm has a slow phase. The stall family is `BoxSlabTestPlanes(p).problem(seed)`, a box with a slab that cuts off one vertex, the square-and-line example lifted to `p` dimensions, on which Dykstra's algorithm has a stalling period. Generator 1, `RandomTestPlanes`, draws by rejection and needs scipy. None of the classes normalises its rows. Each script keeps its seeds in a block of its own, a named constant such as `WEDGE_SEED` or `GRID_SEED`, and a new study takes a new block.
- `LTISolver(..., profile=True)` leaves, after each solve, `SolveProfile.as_dict()` in `solver.profile`: the wall time, the flops, and the entries of every stage in `lti_solver.STAGES`, the counts in `COUNTS`, and one record per KKT test, while `profile=False`, the default, leaves `None`. The flops come from the functions from `lstsq_flops` to `replay_flops` in `lti_numerics.py`. Profiling changes neither the path nor any output, and `test_a_profiled_solve_leaves_every_output_bitwise_unchanged` holds it to that, hence a stage hook only reads the state.
- `schedule`, a setting of `LTISolver` and of `c_solvers.accelerated_c_projection`, decides where the `deflated_modal` preset tries the KKT test. `"work"`, the default, tries it at the first episode entry and afterwards only once the work done since the last failed entry test exceeds that test's work, so that the failed entry tests but the last cost at most the rest of the work. `"every"` tries it at every episode entry, and `"doubling"` at entries 1, 2, 4, 8, and so on. These three also try it at the limit of a singular episode and at a final active set, which the schedule does not decide. `"never"` runs no KKT test anywhere, hence the preset never settles and measures compaction alone. The work is counted in flops from sizes that both languages share, by the functions that follow `SCHEDULES` in `lti_numerics.py`, which `C/accelerated.c` copies, hence a change to the count belongs in both, and `CScheduleTests` in `tests/test_c_solvers.py` holds the C's tests and work to the Python's. Say under which schedule a settle cycle in a paper claim was measured.
- An LTI decision on a quantity that is zero in exact arithmetic, such as a drift, a slack, or an increment, compares it with a rounding floor of 64 eps times the magnitudes that were summed. Every other LTI tolerance is relative rather than an absolute constant, because neither the problem's scale nor a coordinate that is not touched by any constraint may change a decision. Test a new one by scaling the problem and by adding an unrelated large coordinate. A cycle mixes the coordinates its active normals touch, hence the stationary-state test `is_frozen` charges each of them with the largest of their magnitudes, and a coordinate that no active normal touches with its own. `CycleMap.coupled` marks those coordinates in the Python, and `C/accelerated.c` keeps the same flags. The core solvers follow the same rule: `_track_error` and `_beta_check` compare against `RELATIVE_TOLERANCE` (1e-10) of the magnitudes involved, and the stall detection solver compares outputs against the same rounding floor.
- Comments state what the code cannot: the reason, a constraint, a trap. Do not narrate what the code does.
- `C/accelerated.c` follows `lti_numerics.py` piece for piece, and a comment marks each place where it departs from it. It departs in four places. It scans a singular episode by iterating the contracting block in the coordinates of the span, where the Python diagonalises that block. It estimates the condition number of `I - A_m` rather than computing it. Its Lawson-Hanson solves reuse a QR factorisation updated as rows enter and leave the support. It drops every coordinate that is not touched by any constraint before the solve, where the Python takes each episode decision on the touched coordinates. A change to the LTI numerics belongs in both languages, and `tests/test_c_solvers.py` holds the C to the same contract: without a history, the path is checked by solving at every budget.
- `BitwiseReferenceTests` in `tests/test_stress_core.py` holds the three core solvers bit for bit to `tests/core_solver_reference.npz`, which records digests of every output, the printed log included, on 52 problems. A change to the core solvers that is meant to be exact, such as a performance fix, must leave it passing. The comparison runs only on macOS arm64 with numpy 2.1.1 on Accelerate, where the fixture was recorded, and the repository does not hold a script that writes the fixture.
- When fixing a bug, write the regression test first and confirm it fails without the fix. Such tests belong in `tests/test_solver_regressions.py` for the core solvers, in `tests/test_lti_solvers.py` for the LTI solvers and the oracle, and in `tests/test_c_solvers.py` for the C solvers.

## Git

- Never commit. Stage with `git add` and hand back a suggested message, and the author reviews and commits. On the author's machine a hook enforces this.
- `master` is the only branch.
- Everything removed from the tree remains in git history: the earlier report and Elsevier-format paper in `Latex/`, the nine development versions in `Python/Previous Versions/`, the superseded procedural implementations in `Python/Working Version/bin/`, the recorded CSV files of `results/` and `Python/Working Version/results/`, and `requirements.txt`. `git log -- <path>` finds the commits that held one.
- Agent instructions belong in this file. On the author's machine `CLAUDE.md` is gitignored globally, so a `CLAUDE.md` here would be invisible to git and would never reach anyone through the remote.
- Do not commit PDFs, build artefacts, generated outputs, or duplicate copies of files that already exist elsewhere in the tree.

## CI

- Continuous integration here is the author's learning exercise, a rehearsal before he sets up CI for a larger project. Explain, verify, and review, but do not write the workflow file or change the repository's settings for him, and keep any draft workflow back until he has written his own.

## Journal

If the local agent tooling provides a work journal, use it, with the project name `dykstra-projection`: a `start` entry when you begin, one entry as each batch of work is staged, and an `end` entry when you stop. Read the recent entries first, since a `start` with no matching `end` means another session may still be working in this checkout. If no such tooling is configured, skip this section. Nothing in the repository depends on it.

## Claude Code only

Claude Code does not read `AGENTS.md` by name. Point it at this file explicitly, or keep a one-line `CLAUDE.md` containing `@AGENTS.md` (it stays gitignored, so it never reaches the remote).
