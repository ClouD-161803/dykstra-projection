# AGENTS.md

Instructions for any coding agent working in this repository. `ARCHITECTURE.md`, beside this file, is the longer explanation of how the code is put together; read it before changing anything in `Python/Working Version/`.

## What this is

A Python implementation of Dykstra's algorithm for projecting a point onto the intersection of finitely many half-spaces, plus two experimental variants, the LTI-accelerated solvers, 2-D visualisers, CSV export, and the LaTeX report that uses them. Research code behind a single-author paper, not a library: there is no package, no installer, and no public API to keep stable.

Author: Claudio Vestini, University of Oxford. Funded by Keble Research Grant KSRG118.

The repository is **public on GitHub** (`ClouD-161803/dykstra-projection`). Everything committed here is world-readable. Do not add credentials, personal documents, or third-party PDFs.

## Problem convention

Constraints are rows of `A` with offsets `b`, meaning `A @ x <= b`. Each row of `A` is a half-space normal, not necessarily of unit length: the solver normalises internally. The dimension is inferred from the point `z`, and every constraint is validated against it. The numerics work in any dimension; the visualisers are 2-D only.

`delete_spaces=True` is a deprecated no-op kept for compatibility, and warns. Every supplied constraint is retained, because a constraint that already contains the starting point can still determine the projection.

## Layout

```
Python/Working Version/        the only code that matters
  convex_projection_solver.py  ConvexProjectionSolver (ABC), three solvers, dykstra_projection()
  projection_result.py         ProjectionResult, returned by every solve()
  visualiser.py                ResultExporter, Visualiser, VerticalVisualiser, ComparisonVisualiser
  gradient.py                  quadprog_solve_qp, the QP reference projection
  edge_rounder.py              rounded_box_constraints, polygonal corner rounding
  lti_solver.py                LTISolver, its five presets, LTIVer1Solver to LTIVer5Solver
  lti_numerics.py              cycle maps, closed forms, envelope horizons, the KKT certificate
  oracle.py                    replays a recorded activity schedule, for comparison
  lti_examples.py, run_*.py    the shared LTI example and one runner per preset and the oracle
  main.py                      interactive 2-D example
  paper_figure.py              the comparison figure used in the paper
  bin/                         legacy helper implementations, superseded, kept for reference
Python/Previous Versions/      Versions 1-9, development history. READ-ONLY. Never edit or refactor.
tests/                         four unittest files, run from the repo root
Latex/Current Version/         the report and paper sources; Initial Version/ is superseded
results/                       saved experiment CSVs
```

The reference papers and the grant-application material used to live here as `Books & Articles/` and `Application/`. They were moved out of the repository in September 2026, because it is public and neither the publishers' PDFs nor the department's documents are ours to redistribute. Both names are now in `.gitignore`; do not restore them. On the author's machine they sit beside the checkout, in `../dykstra-projection-materials/`. Cite papers from the report's bibliography instead of adding a PDF.

## Environment

`requirements.txt` pins `numpy==2.1.1`, `matplotlib==3.9.2`, `quadprog==0.1.12` and declares Python `>= 3.10, < 3.14`.

On the author's machine the system Python is 3.9.6 and Homebrew's is 3.14.5, so neither satisfies the pin. The only interpreter in range is uv-managed:

```bash
uv venv --python 3.12
uv pip install -r requirements.txt
```

Do not run this code under Python 3.14. `numpy==2.1.1` has no wheel for it, so the import silently resolves to whatever numpy the ambient interpreter has and results are then produced against unpinned dependencies. If you find `__pycache__/*.cpython-314.pyc` in `Python/Working Version/`, it is stale evidence of exactly that; ignore it, and do not treat it as proof of a working environment.

`quadprog` is imported at module scope by `gradient.py`, so `convex_projection_solver` fails to import entirely without it. No other virtual environment on the author's machine carries it.

## Running things

```bash
# tests, from the repo root
python -m unittest discover -s tests -v          # 78 test methods

# the interactive example (needs a GUI matplotlib backend)
cd "Python/Working Version" && python main.py

# the paper comparison figure
cd "Python/Working Version" && python paper_figure.py
```

The test files put `Python/Working Version` on `sys.path` themselves. There is no package, no `__init__.py`, no `conftest.py`, and no pytest configuration. `tests/test_visualiser_regressions.py` calls `matplotlib.use("Agg")` before importing pyplot, so the visualiser tests are headless. `main.py` opens a window and is not.

**Trap.** `paper_figure.py` writes to `"./results"`, relative to the current working directory. Run it from `Python/Working Version` and the CSVs land in `Python/Working Version/results/`; run it from the repo root and they land in `results/`. Both directories exist, both are tracked, and their same-named files differ, so neither set can be dated from the repo. When you touch this, resolve the output directory against the repo root rather than the cwd.

## Conventions

- British spelling in new names and prose: `visualiser`, `normalise`. Some existing identifiers use the `-ize` form (`_initialize_iteration`); match the file you are in rather than renaming.
- Solvers subclass `ConvexProjectionSolver` and override `_update_error`, `solve` and `_format_output`. Shared geometry (`_normalise`, `_is_in_half_space`, `_project_onto_half_space`, `_validate_problem`) lives on the base class as static methods; new shared geometry goes there, not into a subclass.
- `solve()` always returns a `ProjectionResult`. Its optional fields are populated by the constructor flags `track_error`, `plot_errors` and `plot_active_halfspaces`, except `settled_at` and `certificate`, which an LTI solver sets when it settles; test for them with the `has_*` predicates and `is_settled()`, not by comparing to `None` at the call site.
- `track_error=True` compares against the `quadprog` QP solution. That reference is exact; a finite number of Dykstra cycles generally is not. Never call the Dykstra output "the projection" in a paper claim without saying how many cycles produced it.
- An LTI solver returns Dykstra's own iterate at the budget unless `result.is_settled()`. Only `result.certificate == "kkt"` makes the result the projection, to within `1e-9` of the distance moved beyond the rounding of the data, which nearly dependent active normals amplify; a `"finality"` limit takes drifts below rounding as zero. Say which one a paper claim rests on.
- An LTI decision on a quantity that is zero in exact arithmetic, a drift, a slack, an increment, compares it with a rounding floor on the magnitudes that were summed, and every other LTI tolerance is relative, never an absolute constant: the problem's scale and coordinates no constraint touches must not change a decision. Test new ones by scaling the problem and by adding an unrelated large coordinate.
- Comments state what the code cannot: the reason, a constraint, a trap. Do not narrate what the code does.
- When fixing a bug, write the regression test first and confirm it fails without the fix. `tests/test_solver_regressions.py` is where those belong, and `tests/test_lti_solvers.py` for the LTI solvers and the oracle.

## Git

- **Never commit.** Stage with `git add` and hand back a suggested message; the author reviews and commits. On the author's machine a hook enforces this.
- `master` is the default branch. `lti-solvers` carries PR #2 (the LTI solvers with their five presets, an oracle, and `tests/test_lti_solvers.py`), open as of 2026-09-16.
- Agent instructions belong in this file. On the author's machine `CLAUDE.md` is gitignored globally, so a `CLAUDE.md` here would be invisible to git and would never reach anyone through the remote.
- Do not commit PDFs, build artefacts or duplicate copies of files that already exist elsewhere in the tree.

## Journal

If the local agent tooling provides a work journal, use it, with the project name `dykstra-projection`: a `start` entry when you begin, one entry as each batch of work is staged, and an `end` entry when you stop. Read the recent entries first, since a `start` with no matching `end` means another session may still be working in this checkout. If no such tooling is configured, skip this section; nothing in the repository depends on it.

## Claude Code only

Claude Code does not read `AGENTS.md` by name. Point it at this file explicitly, or keep a one-line `CLAUDE.md` containing `@AGENTS.md` (it stays gitignored, so it never reaches the remote).
