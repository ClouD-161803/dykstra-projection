"""Run LTI Ver1 on the shared box-and-line example."""

from lti_examples import run_lti_example
from lti_ver1 import LTIVer1Solver


def run() -> None:
    """Run Ver1."""
    run_lti_example(LTIVer1Solver)


if __name__ == "__main__":
    run()
