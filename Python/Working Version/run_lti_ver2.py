"""Run LTI Ver2 on the shared box-and-line example."""

from lti_examples import run_lti_example
from lti_solver import LTIVer2Solver


def run() -> None:
    """Run Ver2."""
    run_lti_example(LTIVer2Solver)


if __name__ == "__main__":
    run()
