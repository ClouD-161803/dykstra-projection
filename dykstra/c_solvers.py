"""ctypes access to the C solvers in the repository's C folder, the accelerated
solver and plain Dykstra. Build the libraries there first. Each call returns the
final point, whether it settled, the cycles accounted for and the multipliers. The
accelerated solver takes the schedule of the KKT test that LTISolver takes."""

import ctypes
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Callable
import numpy as np
from dykstra import REPOSITORY_ROOT
from dykstra.convex_projection_solver import ConvexProjectionSolver
from dykstra.lti_numerics import DEFAULT_SCHEDULE, SCHEDULES

C_FOLDER = REPOSITORY_ROOT / "C"

# The C cycle counters are ints that step one past the budget before they stop
MAX_CYCLES = 2**31 - 2

_SUFFIX = {"win32": ".dll", "darwin": ".dylib"}.get(sys.platform, ".so")
_DOUBLES = ctypes.POINTER(ctypes.c_double)


class _AcceleratedInfo(ctypes.Structure):
    """Accelerated solver counters."""
    _fields_ = [("status", ctypes.c_int), ("cycles", ctypes.c_int),
                ("episodes", ctypes.c_int), ("step_episodes", ctypes.c_int),
                ("switches", ctypes.c_int), ("skips", ctypes.c_int),
                ("cycles_skipped", ctypes.c_longlong), ("kkt_attempts", ctypes.c_int),
                ("entry_tests", ctypes.c_int), ("failed_entry_work", ctypes.c_longlong),
                ("work", ctypes.c_longlong), ("power_iterations", ctypes.c_longlong),
                ("settled_at", ctypes.c_int)]


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
    # included, which for plain Dykstra is every cycle. The other counters are
    # the accelerated solver's and zero for plain Dykstra: the episodes it scanned
    # in closed form, regular or deflated, those it stepped through the cycle map,
    # its exact cycles after the first, and the jumps whose cycles cycles_skipped
    # sums. kkt_attempts counts its KKT tests, entry_tests those at phase entries,
    # and failed_entry_work and work are the work of the failed entry tests and
    # all other work, as LTISolver.kkt_record counts them. power_iterations counts
    # the iterations of the power methods that bound norms of phase matrices.
    # settled_at is the cycle from which a settled accelerated result is the
    # projection, as LTISolver's settled_at, and None for any other result
    projection: np.ndarray
    settled: bool
    cycles: int
    multipliers: np.ndarray
    exact_cycles: int = 0
    cycles_skipped: int = 0
    episodes: int = 0
    step_episodes: int = 0
    switches: int = 0
    skips: int = 0
    kkt_attempts: int = 0
    entry_tests: int = 0
    failed_entry_work: int = 0
    work: int = 0
    power_iterations: int = 0
    settled_at: int | None = None


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
        # the accelerated solver takes the schedule's code after the budget
        settings = [ctypes.c_int, ctypes.c_int] if name == "accelerated" else [ctypes.c_int]
        function.argtypes = ([ctypes.c_int, ctypes.c_int, _DOUBLES, _DOUBLES, _DOUBLES] + settings
                             + [_DOUBLES, _DOUBLES, ctypes.POINTER(info_type)])
        _loaded[name] = function
    return _loaded[name]


def _solve(name: str, z: np.ndarray, A: np.ndarray, b: np.ndarray,
           max_iter: int, schedule: str | None = None) -> CSolverResult:
    """Run one C solver, the accelerated one with a schedule from SCHEDULES."""
    # Validate as the Python solvers do; the C code checks neither shapes nor
    # for NaN and infinity, and is handed contiguous copies
    z, A, b, max_iter = ConvexProjectionSolver._validate_problem(z, A, b, max_iter, None)
    if max_iter > MAX_CYCLES:
        raise ValueError(f"max_iter must not exceed {MAX_CYCLES} for the C solvers.")
    if name == "accelerated" and (not isinstance(schedule, str) or schedule not in SCHEDULES):
        raise ValueError(f"schedule must be one of {', '.join(SCHEDULES)}.")
    n = A.shape[0]

    # Zero cycles leave the point where it is, and the C code needs a budget of one.
    # Without constraints that point is its own projection, which the C code reports as
    # settled after no cycle at any other budget unless the schedule runs no KKT test
    if max_iter == 0:
        settled = n == 0 and schedule != "never"
        return CSolverResult(z, settled, 0, np.zeros(n), settled_at=0 if settled else None)

    function = _entry_point(name)
    info = _SOLVERS[name][1]()
    x = np.zeros(z.size)
    multipliers = np.zeros(max(n, 1))
    settings = (max_iter, SCHEDULES.index(schedule)) if name == "accelerated" else (max_iter,)
    code = function(n, z.size, A.ctypes.data_as(_DOUBLES), b.ctypes.data_as(_DOUBLES),
                    z.ctypes.data_as(_DOUBLES), *settings, x.ctypes.data_as(_DOUBLES),
                    multipliers.ctypes.data_as(_DOUBLES), ctypes.byref(info))
    if code < 0:
        raise RuntimeError(f"The C {name} solver failed with code {code}.")
    # The accelerated solver counts its exact cycles after the first
    accelerated = name == "accelerated"
    counters = ({"cycles_skipped": info.cycles_skipped, "episodes": info.episodes,
                 "step_episodes": info.step_episodes, "switches": info.switches,
                 "skips": info.skips, "kkt_attempts": info.kkt_attempts,
                 "entry_tests": info.entry_tests, "failed_entry_work": info.failed_entry_work,
                 "work": info.work, "power_iterations": info.power_iterations,
                 "settled_at": info.settled_at if code == 0 else None} if accelerated else {})
    return CSolverResult(projection=x, settled=(code == 0), cycles=info.cycles,
                         multipliers=multipliers[:n],
                         exact_cycles=info.switches + 1 if accelerated else info.cycles,
                         **counters)


def accelerated_c_projection(z: np.ndarray, A: np.ndarray, b: np.ndarray,
                             max_iter: int, schedule: str = DEFAULT_SCHEDULE) -> CSolverResult:
    """Accelerated solver in C, trying the KKT test where schedule, one of SCHEDULES, says."""
    return _solve("accelerated", z, A, b, max_iter, schedule)


def dykstra_c_projection(z: np.ndarray, A: np.ndarray, b: np.ndarray,
                         max_iter: int) -> CSolverResult:
    """Plain Dykstra in C."""
    return _solve("dykstra", z, A, b, max_iter)
