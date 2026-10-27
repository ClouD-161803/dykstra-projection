"""Dykstra's algorithm for projecting a point onto an intersection of half-spaces.

The package holds the core solvers and their experimental variants, the accelerated solvers of
the SIAM paper with ctypes access to their C counterparts, and the scripts that rebuild the
paper's and the supplement's figures and tables. It is research code, run from the repository
root as `python -m dykstra.<module>` and not installed into the environment. Importing the
package imports none of its modules.
"""

from pathlib import Path

# The C libraries are built in C/ and every generated output goes to results/, both beside
# this folder
REPOSITORY_ROOT = Path(__file__).resolve().parent.parent
