"""Records one Dykstra run's active-set schedule and replays it with the oracle."""

import time
import numpy as np
from oracle import record_schedule, oracle_lti_projection
from lti_ver5 import LTIVer5Solver


def run() -> None:
    """Box-and-line example."""
    # Box: -1 <= x <= 1, -1 <= y <= 1
    A_box = np.array([[1., 0.], [-1., 0.], [0., 1.], [0., -1.]])
    b_box = np.array([1., 1., 1., 1.])
    # Line x/2 + y = 1 as two opposite half-spaces
    A_line = np.array([[0.5, 1.], [-0.5, -1.]])
    b_line = np.array([1., -1.])

    z = np.array([-2., 1.4])
    max_iter: int = 200

    A: np.ndarray = np.vstack([A_box, A_line])
    b: np.ndarray = np.hstack([b_box, b_line])

    schedule, dykstra_result = record_schedule(z, A, b, max_iter)
    start = time.perf_counter()
    oracle_result = oracle_lti_projection(z, A, b, schedule)
    oracle_time = time.perf_counter() - start
    accelerated_result = LTIVer5Solver(z, A, b, max_iter).solve()

    print(f"\nSchedule: {len(schedule)} episode(s) over {sum(k for _, k in schedule)} cycle(s)")
    for active, k in schedule:
        print(f"  active half-spaces {[m for m, on in enumerate(active) if on]} for {k} cycle(s)")
    print(f"Dykstra projection:  {dykstra_result.projection}")
    print(f"Oracle replay:       {oracle_result.projection}  ({oracle_time * 1e3:.3f} ms)")
    print(f"LTI Ver5 projection: {accelerated_result.projection}\n")


if __name__ == "__main__":
    run()
