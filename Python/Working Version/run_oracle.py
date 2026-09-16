"""Records one Dykstra run's active-set schedule and replays it with the oracle."""

import time
from lti_ver5 import LTIVer5Solver
from lti_examples import box_line_problem
from oracle import oracle_lti_projection, record_schedule


def run() -> None:
    """Box-and-line example."""
    z, A, b, _ = box_line_problem()
    max_iter: int = 200

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
