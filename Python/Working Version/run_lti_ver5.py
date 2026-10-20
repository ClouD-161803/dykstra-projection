"""Run LTI Ver5 on the shared box-and-line example."""

from lti_examples import run_lti_example
from lti_solver import LTIVer5Solver


def run() -> None:
    """Run Ver5."""
    run_lti_example(LTIVer5Solver)


if __name__ == "__main__":
    run()
