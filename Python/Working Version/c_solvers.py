"""ctypes access to the C solvers in the repository's C folder, the accelerated
solver and plain Dykstra. Build the libraries there first. Each call returns the
final point, whether it settled, the cycles accounted for and the multipliers."""

import ctypes
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Callable
import numpy as np
from convex_projection_solver import ConvexProjectionSolver

C_FOLDER = Path(__file__).resolve().parents[2] / "C"

# The C cycle counters are ints that step one past the budget before they stop
MAX_CYCLES = 2**31 - 2

_SUFFIX = {"win32": ".dll", "darwin": ".dylib"}.get(sys.platform, ".so")
_DOUBLES = ctypes.POINTER(ctypes.c_double)


class _AcceleratedInfo(ctypes.Structure):
    """Accelerated solver counters."""
    _fields_ = [("status", ctypes.c_int), ("cycles", ctypes.c_int),
                ("episodes", ctypes.c_int), ("step_episodes", ctypes.c_int),
                ("switches", ctypes.c_int), ("skips", ctypes.c_int),
                ("cycles_skipped", ctypes.c_longlong)]


class _DykstraInfo(ctypes.Structure):
    """Plain Dykstra counters."""
    _fields_ = [("status", ctypes.c_int), ("cycles", ctypes.c_int)]


# Entry point and counter layout of each library
_SOLVERS = {
    "accelerated": ("accelerated_dykstra_solve", _AcceleratedInfo),
    "dykstra": ("dykstra_solve", _DykstraInfo),
}
_loaded = {}


@dataclass
class CSolverResult:
    """Outcome of a C solve."""
    # A settled accelerated result is the projection; a settled Dykstra result is a
    # state that no further cycle changes; otherwise the point is the iterate at
    # the budget. The multipliers refer to the rows of A as they were passed.
    # exact_cycles counts the cycles run as plain Dykstra cycles, the first
    # included, which for plain Dykstra is every cycle
    projection: np.ndarray
    settled: bool
    cycles: int
    multipliers: np.ndarray
    exact_cycles: int = 0
    cycles_skipped: int = 0


def library_path(name: str) -> Path:
    """Where a built library lives."""
    return C_FOLDER / f"{name}{_SUFFIX}"


def libraries_built() -> bool:
    """Both libraries are present."""
    return all(library_path(name).exists() for name in _SOLVERS)


def _entry_point(name: str) -> Callable[..., int]:
    """Load a library once."""
    if name not in _loaded:
        path = library_path(name)
        if not path.exists():
            raise FileNotFoundError(
                f"{path} is missing: build the C solvers in {C_FOLDER} first, "
                "with build.ps1 on Windows or make elsewhere.")
        symbol, info_type = _SOLVERS[name]
        function = getattr(ctypes.CDLL(str(path)), symbol)
        function.restype = ctypes.c_int
        function.argtypes = [ctypes.c_int, ctypes.c_int, _DOUBLES, _DOUBLES, _DOUBLES,
                             ctypes.c_int, _DOUBLES, _DOUBLES, ctypes.POINTER(info_type)]
        _loaded[name] = function
    return _loaded[name]


def _solve(name: str, z: np.ndarray, A: np.ndarray, b: np.ndarray,
           max_iter: int) -> CSolverResult:
    """Run one C solver."""
    # Validate as the Python solvers do; the C code checks neither shapes nor
    # for NaN and infinity, and is handed contiguous copies
    z, A, b, max_iter = ConvexProjectionSolver._validate_problem(z, A, b, max_iter, None)
    if max_iter > MAX_CYCLES:
        raise ValueError(f"max_iter must not exceed {MAX_CYCLES} for the C solvers.")
    n = A.shape[0]

    # Zero cycles leave the point where it is, and the C code needs a budget of one
    if max_iter == 0:
        return CSolverResult(z, False, 0, np.zeros(n))

    function = _entry_point(name)
    info = _SOLVERS[name][1]()
    x = np.zeros(z.size)
    multipliers = np.zeros(max(n, 1))
    code = function(n, z.size, A.ctypes.data_as(_DOUBLES), b.ctypes.data_as(_DOUBLES),
                    z.ctypes.data_as(_DOUBLES), max_iter, x.ctypes.data_as(_DOUBLES),
                    multipliers.ctypes.data_as(_DOUBLES), ctypes.byref(info))
    if code < 0:
        raise RuntimeError(f"The C {name} solver failed with code {code}.")
    # The accelerated solver counts its exact cycles after the first
    accelerated = name == "accelerated"
    return CSolverResult(projection=x, settled=(code == 0), cycles=info.cycles,
                         multipliers=multipliers[:n],
                         exact_cycles=info.switches + 1 if accelerated else info.cycles,
                         cycles_skipped=info.cycles_skipped if accelerated else 0)


def accelerated_c_projection(z: np.ndarray, A: np.ndarray, b: np.ndarray,
                             max_iter: int) -> CSolverResult:
    """Accelerated solver in C."""
    return _solve("accelerated", z, A, b, max_iter)


def dykstra_c_projection(z: np.ndarray, A: np.ndarray, b: np.ndarray,
                         max_iter: int) -> CSolverResult:
    """Plain Dykstra in C."""
    return _solve("dykstra", z, A, b, max_iter)
