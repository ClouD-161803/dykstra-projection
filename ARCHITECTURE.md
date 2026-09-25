# Architecture

How the code in `Python/Working Version/` is put together, and which of its properties are contracts rather than accidents. `AGENTS.md` is the shorter operating guide: environment, commands, conventions, traps. Read that first, this when you are about to change something.

## 1. Scope and nongoals

One author's research code behind a paper. It is not a library: no package, no installer, no stable API, no versioning. Callers are `main.py`, `paper_figure.py`, the LTI runners and the test suite, all inside this repository.

The numerics are dimension-agnostic; the dimension comes from the point being projected. The visualisation layer is 2-D only and always will be, because its job is to show geometry a reader can see. Do not try to generalise the visualisers; add a different output instead.

`Python/Previous Versions/` and `Python/Working Version/bin/` are history. They are kept so that a claim in the report can be traced back to the code that produced it. They are read-only: never refactor them, never fix a bug in them, never import from them.

## 2. The problem

Project a point $z$ onto $\{x : Ax \le b\}$, the intersection of finitely many half-spaces.

The method of alternating projections (MAP) converges to some point of the intersection. Dykstra's algorithm converges to the projection, by carrying a correction vector per constraint and subtracting it before each projection. That difference is the whole reason this repository exists.

For polyhedra Dykstra's algorithm converges finitely, but it can stall: several consecutive cycles in which the iterate does not move while the correction vectors keep growing, after which it moves again. Stalling, and getting out of it in one step instead of many, is what this project is actually about. The report in `Latex/Current Version/` is the long version.

## 3. Module graph

```
gradient.py ────> convex_projection_solver.py <──── projection_result.py
                            ^        │
       edge_rounder.py ─────┘        v
      (constraint generator)   visualiser.py ────> CSV via ResultExporter
                                     ^
                     main.py, paper_figure.py
```

The LTI solvers sit beside the core solvers, not inside them: `lti_solver.py` subclasses `ConvexProjectionSolver` and draws its numerics from `lti_numerics.py`, which knows nothing of solver state; `oracle.py` draws on `lti_numerics.py` but not on `lti_solver.py`; the `run_*.py` runners, through the shared example in `lti_examples.py`, are the LTI entry points.

`convex_projection_solver.py` owns the algorithms and knows nothing about plotting. `projection_result.py` is a dataclass and knows nothing at all. `visualiser.py` reads a `ProjectionResult` and the constraint arrays; it never calls a solver. `gradient.py` wraps `quadprog` and is the only place that library is named. `edge_rounder.py` produces constraint arrays and is optional: nothing else imports it at module scope.

Imports are flat, by module name, resolved through `sys.path`. There is no package and no `__init__.py`. That has not been changed because every entry point either runs inside `Python/Working Version/` or inserts that directory itself, and introducing a package would break the recorded invocations in the report and in the README.

## 4. The solver hierarchy

`ConvexProjectionSolver` is an abstract base class holding two distinct things.

The geometry, as static methods, dependency-free and individually testable: `_normalise` (scales a normal to unit length and its offset with it), `_is_in_half_space`, `_project_onto_half_space`, `_validate_problem`, `_find_optimal_solution` (the QP reference, via `gradient.py`), `_beta_check` (is the point inside the whole intersection), and `_delete_inactive_half_spaces` (legacy, retained only so a test can prove it now keeps everything).

The tracking scaffolding, as instance methods: `_initialize_iteration`, `_check_activity`, `_track_activity`, `_track_error`, `_update_error`. The constructor allocates every tracking array up front at `max_iter + 1` cycles, because the initial state is cycle 0.

Three concrete solvers:

- `DykstraProjectionSolver` is the standard cyclic method and the reference implementation. Its `_update_error` is `e[m] = e[index] + (x_temp - x)`; setting that coefficient to zero would turn it into MAP.
- `DykstraMapHybridSolver` keeps two sets of correction vectors, `e_dykstra` and `e_MAP`, and at the top of each cycle `_initialize_iteration` selects between them with `_beta_check`: Dykstra's corrections while the iterate is inside the intersection, zeroed MAP corrections while it is outside. Only the Dykstra set is ever updated.
- `DykstraStallDetectionSolver` is this project's own contribution. It flags a stall as the paper's Definition 1 does: the last `n` visits, one per constraint and possibly straddling a cycle boundary, all repeated their outputs from one cycle earlier. One repeating visit is not enough, because a partly frozen cycle fed jumps that plain Dykstra never makes. While the cycle repeats, an active half-space's scalar `d_m = e_m . a_m` changes by its constant slack `s_m = a_m^T x_{m-1} - b_m` on every visit, so it turns inactive after `ceil(d_m / -s_m)` visits. `_handle_stalling` takes `N` as the least of these over the half-spaces active at their last visit with `s_m < 0`, advances every correction vector by `N - 1` cycles at once, and lets the switching cycle run normally. It does not jump when nothing drains, when a half-space inactive at its last visit moved its input (it has just switched off, so the cycle will not repeat), or when `N` exceeds `1/sqrt(eps)`, about 6.7e7: beyond that, plain Dykstra's per-cycle rounding of `e_m` is as large as the drain, and at a converged fixed point the slacks are rounding noise that would ask for about 1e16 cycles. The jump lands on the state plain Dykstra reaches after the skipped cycles, so only the path is shortened and the limit is untouched; `tests/test_solver_regressions.py` checks this cycle for cycle. Detection uses exact equality rather than the paper's `ε_stall`, so a stall whose outputs jitter in the last bit, as a line's inactive side can, is caught late. It prints when it fast-forwards.

The extension contract for a fourth variant: subclass `ConvexProjectionSolver`, override `_update_error`, `solve` and `_format_output`, and preserve the invariant every variant shares, that one cycle projects onto every constraint exactly once in the order the rows of `A` are given. Shared geometry goes on the base class, not into the subclass.

`dykstra_projection()` at the bottom of the module is a backwards-compatibility wrapper returning the old five-tuple. New code calls the solver classes.

### The LTI solvers

`LTISolver` in `lti_solver.py` accelerates Dykstra's algorithm by treating it, between changes of the active set, as a linear time-invariant system, the reformulation and accelerated algorithm of the SIAM paper, whose sources live outside this repository. It is one class with five presets, `cycle_map`, `closed_form`, `envelope`, `frozen_stall` and `deflated_modal`, each adding one technique to the one before so that the techniques can be compared; `LTIVer1Solver` to `LTIVer5Solver` default to one preset each and keep the names the runners and experiments use.

`solve()` runs one exact Dykstra cycle, builds the cycle map of the active set it found, and then runs episodes. An episode is a run of cycles with a constant active set; it ends in one of three outcomes, a switch at a given cycle, the budget running out, or a settlement, and a switch is realised by one exact Dykstra cycle before the next episode starts. Within an episode the presets differ in how they get from one cycle to a later one: stepping through the cycle map, evaluating the closed form around its fixed point, jumping a run an envelope bound certifies switch-free, fast-forwarding a frozen stall, or scanning the deflated modal form of a singular episode. They also differ in how they settle: `cycle_map` never does, `closed_form` to `frozen_stall` settle on the fixed point once the active set is proven final, and `deflated_modal` alone tries the KKT certificate, at every episode's entry and wherever its active set is proven final, and settles only on a point that passes it.

The state is the point and one scalar auxiliary per half-space, since Dykstra's correction for a half-space is always a multiple of its unit normal. Exact cycles write their own history rows in `_exact_cycle`. A cycle the solver skipped is replayed from the previous row by `_record_cycle`, which is exact within an episode because an active half-space projects onto its boundary and an inactive one leaves the point alone. After a settlement `_record_limit` fills every remaining row with the limit.

`lti_numerics.py` holds the pieces that need no solver state: the exact cycle, the cycle map and the prediction of the next active set from it, the closed forms, the finality test, the envelope horizon, the frozen-stall crossing, nonnegative least squares, and the KKT certificate.

## 5. The result object

`solve()` always returns a `ProjectionResult`. `projection` and `path` are always populated; `settled_at` and `certificate` are `None` unless an LTI solver settled, and the rest unless the matching constructor flag was set.

| Field | Flag | Shape |
|---|---|---|
| `projection` | always | `(dim,)` |
| `path` | always | `(max_iter + 1, constraints, dim)` |
| `squared_errors` | `track_error` | `(max_iter + 1,)` |
| `stalled_errors` | `track_error` | `(max_iter + 1,)`, `nan` where not stalled |
| `converged_errors` | `track_error` | `(max_iter + 1,)`, `nan` where not converged |
| `errors_for_plotting` | `plot_errors` | `(max_iter, constraints, dim)` |
| `active_half_spaces` | `plot_active_halfspaces` | `(constraints, max_iter + 1)` |
| `settled_at` | LTI solvers that settle | cycle, or `None` |
| `certificate` | LTI solvers that settle | `"kkt"`, `"finality"`, or `None` |

Note the transposition: `active_half_spaces` is constraint-major, everything else is cycle-major.

`squared_errors`, `stalled_errors` and `converged_errors` partition the same sequence three ways, so a plot can colour each cycle by regime. `_track_error` rounds the squared distance to ten decimals before comparing, and calls a cycle stalled when its error equals the previous cycle's; without the rounding, floating-point noise would hide every stall.

The flags are opt-in because the arrays are the expensive part: `path` and `errors_for_plotting` are both cycles by constraints by dimension.

Use the `has_error_tracking`, `has_error_plotting_data` and `has_active_halfspace_data` predicates at a call site rather than testing a field against `None`, and `is_settled` for the settlement.

An LTI result is Dykstra's own iterate after `max_iter` cycles, with the same path, errors and corrections cycle for cycle, unless it settled; then it is the limit from `settled_at` on, in the result and in every history row. `active_half_spaces` means something slightly different for the two families: the core solvers test each half-space at the cycle's end point shifted by its own correction, as if it were visited first; the LTI solvers record the set that was active during the cycle.

## 6. Visualisation and export

Three visualiser classes, not interchangeable. `Visualiser` lays the half-space panels out horizontally. `VerticalVisualiser` subclasses it and puts the activity traces on a lower axis, for problems with more constraints than fit across. `ComparisonVisualiser` is separate, not a subclass, and draws two solvers' results against each other: the shared projection panel, the error comparison, and the activity comparison. `paper_figure.py` uses it.

`ResultExporter.export` writes a sectioned CSV, not a table. The sections, in order: `METADATA` (solver name, iteration count, dimension, constraint count, `settled_at` and `certificate` for a settled result, then any keyword arguments passed through), `INITIAL_POINT`, `FINAL_PROJECTION`, `CONSTRAINTS_A`, `CONSTRAINTS_B`, then `PATH_HISTORY`, `SQUARED_ERRORS`, `STALLED_ERRORS`, `CONVERGED_ERRORS`, `ERRORS_FOR_PLOTTING` and `ACTIVE_HALFSPACES` for whichever tracking was enabled. If `output_path` names a directory, the filename becomes `<solver_name>_<max_iter>_iterations.csv`.

`ResultExporter.load` reads that back and reshapes the flattened path and error blocks to cycle by constraint using `num_constraints` from the metadata. It also accepts the legacy section names `CONSTRAINTS_N` and `CONSTRAINTS_C`, which is what `test_csv_loader_accepts_legacy_constraint_sections` pins: older recorded experiments used them, and they predate the `A @ x <= b` naming.

`paper_figure.py` sets `output_dir = "./results"`, relative to the current working directory. This is the known trap described in `AGENTS.md`, and the reason two `results/` directories exist with same-named, differing files. Resolving it against the repository root is the fix.

## 7. Numerical contracts

Normals need not be unit length; `_normalise` scales each row and its offset together before use. Callers may pass whatever the geometry gave them.

Validation is strict and happens once, in `_validate_problem`, which copies its inputs so a caller's arrays are never mutated. `ValueError` is raised for: a `z` that is not a nonempty 1-D point, an `A` that is not 2-D, a `b` that is not 1-D, a row count mismatch between `A` and `b`, a normal whose width is not `z.size`, any nonfinite entry, a zero-norm normal, an explicit `dimensions` that disagrees with `z.size`, a `max_iter` that is not a nonnegative integer, and a `min_error` that is nonfinite or negative. Booleans are rejected where an integer is required. An empty constraint set is legal and returns the point unchanged.

The reference projection is the `quadprog` QP solution, computed once in the constructor. It is exact. Dykstra at finite `max_iter` generally is not. Every error number in this repository and in the paper is a squared distance to that QP solution, not to an idealised limit, and no result should be described as "the projection" without naming the cycle count that produced it.

## 8. Testing strategy

`tests/test_solver_unit.py` covers the geometry primitives and input validation: projection of a feasible point, a nonunit normal, and each rejection path.

`tests/test_solver_regressions.py` covers behaviours that were once wrong. Constraints already satisfied at the start are retained; the dimension is inferred from the point; a mismatched explicit `dimensions` is rejected; zero iterations still records the initial error; a fast-forwarded run matches plain Dykstra at the corresponding cycles, including where a single repeating visit or a converged fixed point once triggered a jump. New bug fixes belong here, test first.

`tests/test_visualiser_regressions.py` covers layout scaling past three constraints and CSV round-tripping, including the legacy section names.

Each file inserts `Python/Working Version` into `sys.path` itself, which is why the suite runs from the repository root with `python -m unittest discover -s tests -v` and needs no package, no `conftest.py` and no pytest. `test_visualiser_regressions.py` calls `matplotlib.use("Agg")` before importing pyplot; the solver tests never import pyplot at all.

## 9. History and dead ends

`Python/Previous Versions/Version 1` through `Version 9` are the development history, one directory per rewrite, each self-contained. Only `Version 3` carries a descriptive suffix, `(path plotter)`.

`Python/Working Version/bin/` holds the superseded procedural implementations that the class hierarchy replaced: `dykstra.py`, `dykstra_MAP_hybrid.py`, `dykstra_functions.py`, `new_dykstra.py`, `plotter.py`.

`Latex/Initial Version/` is the earlier report. `Latex/Current Version/` supersedes it entirely.

## 10. The paper

`Latex/Current Version/` holds `report.tex` (the long report), `paper.tex` (the Elsevier-format paper, class `autart.cls`), `summary.tex`, and `Sections/1-Introduction.tex` through `5-Conclusion.tex`. `mymath.sty` carries the macros and `master_bib_abbrev.bib` the bibliography.

`paper_figure.py` produces the comparison figure: a box with a line through it, projected from `(-2, 1.4)` over 30 cycles, `DykstraProjectionSolver` against `DykstraStallDetectionSolver`. It writes `Dykstra_30_iterations.csv` and `Algorithm (1)_30_iterations.csv`. The other CSVs in `results/`, at 40, 50 and 51 cycles and one named `Fast Forward_51_iterations.csv`, are from earlier runs with different settings; nothing in the repository records which figure each one backs, so check the report before reusing them.

The paper's own plots do not read `results/` at all. `Sections/4-Numerical Example.tex` and `plot.tex` point pgfplots at `data_iterates.csv`, `data_dykstra_original.csv` and `data_dykstra_modified.csv` in `Latex/Current Version/`, which are flat two- and three-column tables (`x,y` and `iteration,error,halfspace`), not the sectioned format `ResultExporter` writes. Nothing in the repository connects the two, so the step from a solver run to a plotted table is undocumented and currently manual. Every other figure is a `.png` in `Figures/`.

`flatten.py` inlines every `\input` of `paper.tex` into `paper_flattened.tex` for submission, rewriting paths that are relative to the repository root rather than to `Latex/Current Version/`. Run it from inside that directory. `agent-config/bin/flatten_tex.py` on the author's machine does the same job; the two are not synchronised.
