"""Run LTI Ver4 on the shared box-and-line example."""

from lti_examples import run_lti_example
from lti_ver4 import LTIVer4Solver


def run() -> None:
    """Run Ver4."""
    run_lti_example(LTIVer4Solver)


if __name__ == "__main__":
    run()
