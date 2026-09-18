# Dykstra Projection

This repository contains a Python implementation of Dykstra's algorithm for
projecting a point onto the intersection of finitely many half-spaces. It also
includes experimental MAP/Dykstra and stalling-aware variants, 2-D
visualisations, result export utilities, and the accompanying LaTeX report.

## Requirements

- Python 3.10 through 3.13
- The packages pinned in [`requirements.txt`](requirements.txt)
- A graphical Matplotlib backend to display the example plots

The pinned NumPy release supports Python 3.10–3.13; use a matching interpreter
when creating the environment.

## Quick start

From the repository root:

```bash
python3.10 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
cd "Python/Working Version"
python main.py
```

The default example projects `[-2.0, 1.4]` onto the intersection of a unit box
and the line `x / 2 + y = 1`. It prints the finite-iteration result and the
quadratic-programming reference solution, then opens a 2-D visualisation.

## Defining a projection problem

Each row of `A` is a half-space normal and the corresponding entry in `b` is
its offset:

```text
A @ x <= b
```

For example, the following projects a point onto a square:

```python
import numpy as np

from convex_projection_solver import DykstraProjectionSolver

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
print(solver.actual_projection)  # QP reference used for error tracking
```

The solver infers the dimension from `z` and validates that every constraint
has the same dimension. Numerical solving works in any dimension; the supplied
visualisers are for 2-D problems.

All supplied constraints are retained. A constraint that contains the starting
point can still determine the projection, so `delete_spaces=True` is retained
only as a deprecated compatibility option and no longer removes constraints.

## Solvers and outputs

- `DykstraProjectionSolver` is the standard cyclic projection method.
- `DykstraMapHybridSolver` combines MAP and Dykstra updates.
- `DykstraStallDetectionSolver` includes the project's experimental
  stalling-detection and fast-forwarding logic.

`solve()` returns a `ProjectionResult`. Optional fields are populated by the
corresponding flags:

- `track_error=True` records squared error against the QP reference solution.
- `plot_errors=True` records correction vectors for quiver plots.
- `plot_active_halfspaces=True` records constraint activity by cycle.

`Visualiser` uses a horizontal layout and now scales to any number of activity
traces. `VerticalVisualiser` combines the activity traces on one lower axis.
`ComparisonVisualiser` supports side-by-side solver comparisons.

## Examples and exports

Run the comparison used for the paper from the working-code directory:

```bash
cd "Python/Working Version"
python paper_figure.py
```

`ResultExporter` writes solver metadata, constraints, paths, errors, and
activity information to CSV. The loader restores the recorded path and error
arrays to their original cycle-by-constraint shape.

## Tests

After installing the requirements, run the regression suite from the repository
root:

```bash
python -m unittest discover -s tests -v
```

## Repository layout

```text
.
├── README.md
├── AGENTS.md                    # operating guide for coding agents
├── ARCHITECTURE.md              # how the code is put together
├── requirements.txt
├── Latex/
│   ├── Current Version/         # Current report and paper sources
│   │   └── Sections/
│   │       ├── 1-Introduction.tex
│   │       ├── 2-Dykstra Background.tex
│   │       ├── 3-Main Results.tex
│   │       ├── 4-Numerical Example.tex
│   │       └── 5-Conclusion.tex
│   └── Initial Version/         # Earlier report material
├── Python/
│   ├── Working Version/         # Current implementation
│   │   ├── main.py              # Interactive 2-D example
│   │   ├── paper_figure.py      # Comparison figure runner
│   │   ├── convex_projection_solver.py
│   │   ├── visualiser.py
│   │   ├── projection_result.py
│   │   ├── gradient.py
│   │   ├── edge_rounder.py
│   │   └── bin/                 # Legacy helper implementations
│   └── Previous Versions/       # Development history
├── results/                     # Saved experiment output
└── tests/                       # Regression tests
```

## Notes

The code uses `quadprog` to obtain a reference projection for error reporting.
This reference is useful for experiments, but a finite number of Dykstra cycles
is generally an approximation rather than an exact projection.

The mathematical report lives under
[`Latex/Current Version`](Latex/Current%20Version).

The reference papers and the grant-application material are not in this
repository. They are third-party or personal documents that are not ours to
redistribute, so they are kept outside it, alongside the checkout, and both
directory names are gitignored. The report's bibliography is the list of
references.

## Author and funding

Claudio Vestini — University of Oxford

Project funded by Keble Research Grant KSRG118.
