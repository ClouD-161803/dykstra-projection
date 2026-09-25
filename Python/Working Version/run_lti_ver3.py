"""Run LTI Ver3 on the shared box-and-line example."""

from lti_examples import run_lti_example
from lti_solver import LTIVer3Solver


def run() -> None:
    """Run Ver3."""
    run_lti_example(LTIVer3Solver)


if __name__ == "__main__":
    run()
