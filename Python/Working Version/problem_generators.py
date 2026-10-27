"""The two random polyhedron generators of the SIAM paper's supplement, section
Reproducibility: RandomTestPlanes (Generator 1, rejection sampling) and BoundedTestPlanes
(Generator 2, direct construction).

Each returns constraints A @ x <= b with one row of A per half-space. Neither normalises
its rows, because the law of the normals is stated for the rows as drawn, and the last
spanning row of BoundedTestPlanes, -Q 1, has length sqrt(p). The solvers normalise each row
and its offset internally, hence only the polyhedron matters to them.

The entries s tan(theta) of both laws, with theta uniform on [0, pi/2) and s a random sign,
are standard Cauchy variables. A row of independent Cauchy entries points near a coordinate
axis more often than a uniformly random direction does, thus the law does not make every
direction equally likely. tests/test_paper_outputs.py measures the excess at p = 2.

Both generators draw from a random source passed in by the caller. problem() seeds a
legacy RandomState, whose streams numpy keeps fixed across versions, so that a seed names
the same problem on every installation.
"""

import numpy as np


# Generator 2's interior anchor is uniform on [-ANCHOR_HALF_WIDTH, ANCHOR_HALF_WIDTH]^p, and
# each offset clears it by a margin uniform on SLACK_RANGE
ANCHOR_HALF_WIDTH = 2.0
SLACK_RANGE = (0.2, 1.5)


def _matmul(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    """Returns left @ right without the spurious floating-point warnings of matmul.

    numpy 2.1.1 linked to Apple's Accelerate raises divide-by-zero, overflow and invalid-value
    flags from matrix products that are correct. Where convex_projection_solver.py avoids
    matmul, this function keeps it, since a product written without matmul sums in another
    order and would change the problem a seed names in the last bit. The factors are finite
    and far from overflow, hence none of these flags can be genuine.
    """
    with np.errstate(divide="ignore", over="ignore", invalid="ignore"):
        return left @ right


def cauchy_normals(rng, num_planes: int, num_dimensions: int) -> np.ndarray:
    """Rows with entries s tan(theta), theta uniform on [0, pi/2) and s a random sign."""
    angles = rng.uniform(0.0, np.pi / 2, size=(num_planes, num_dimensions))
    signs = rng.choice([-1, 1], size=(num_planes, num_dimensions))
    return signs * np.tan(angles)


class RandomTestPlanes:
    """Generator 1: random half-spaces, redrawn until their intersection is bounded and
    nonempty.

    The normals follow cauchy_normals and the offsets are standard normal. Each draw is
    tested by the 2p linear programs max x_k and min x_k over the polyhedron, solved with
    scipy's HiGHS interface. A finite optimum of every one proves boundedness, and any
    optimum proves nonemptiness. scipy is in the benchmark dependency group.
    """

    def __init__(self, num_planes: int, num_dimensions: int, max_draws: int = 100_000) -> None:
        self.num_planes = num_planes
        self.num_dimensions = num_dimensions
        self.max_draws = max_draws
        # The number of draws the last call to draw() needed, the accepted one included
        self.draws_used = 0

    def draw(self, rng) -> tuple[np.ndarray, np.ndarray]:
        """Draw constraints whose polyhedron is bounded and nonempty.

        Raises RuntimeError when all max_draws draws are rejected.
        """
        for draws in range(1, self.max_draws + 1):
            A = cauchy_normals(rng, self.num_planes, self.num_dimensions)
            b = rng.standard_normal(self.num_planes)
            if self.is_bounded_and_nonempty(A, b):
                self.draws_used = draws
                return A, b
        raise RuntimeError(f"No bounded, nonempty polyhedron in {self.max_draws} draws.")

    @staticmethod
    def is_bounded_and_nonempty(A: np.ndarray, b: np.ndarray) -> bool:
        """Whether the polyhedron is nonempty and bounds every coordinate on both sides."""
        from scipy.optimize import linprog

        num_dimensions = A.shape[1]
        for k in range(num_dimensions):
            for direction in (1.0, -1.0):
                objective = np.zeros(num_dimensions)
                objective[k] = direction
                # Status 0 is an optimum. infeasibility, unboundedness and numerical
                # trouble all reject the draw
                result = linprog(objective, A_ub=A, b_ub=b, bounds=(None, None), method="highs")
                if result.status != 0:
                    return False
        return True


class BoundedTestPlanes:
    """Generator 2: a bounded, nonempty polyhedron built directly.

    The first p + 1 rows are the axis directions and their negated sum carried through a
    Haar-distributed orthogonal matrix Q, i.e. Q e_1, ..., Q e_p and -Q 1. They positively
    span the space and therefore bound the polyhedron whatever the other rows are, which
    follow cauchy_normals. The offsets b_i = a_i^T x_bar + sigma_i, with the interior anchor
    x_bar and the margins sigma_i drawn as the module constants describe, keep x_bar strictly
    inside every half-space. Q is drawn on the whole orthogonal group, reflections included,
    as the Q factor of a Gaussian matrix with the signs of the diagonal of R moved into it.
    """

    def __init__(self, num_planes: int, num_dimensions: int) -> None:
        if num_planes < num_dimensions + 1:
            raise ValueError("BoundedTestPlanes needs at least p + 1 half-spaces.")
        self.num_planes = num_planes
        self.num_dimensions = num_dimensions

    def draw(self, rng) -> tuple[np.ndarray, np.ndarray]:
        """Draw constraints of a bounded polyhedron with a strictly interior point."""
        p = self.num_dimensions
        spanning = np.vstack([np.eye(p), -np.ones((1, p))])
        basis, upper = np.linalg.qr(rng.standard_normal((p, p)))
        A = _matmul(spanning, (basis * np.sign(np.diag(upper))).T)

        extra = self.num_planes - (p + 1)
        if extra > 0:
            A = np.vstack([A, cauchy_normals(rng, extra, p)])

        anchor = rng.uniform(-ANCHOR_HALF_WIDTH, ANCHOR_HALF_WIDTH, size=p)
        return A, _matmul(A, anchor) + rng.uniform(*SLACK_RANGE, size=self.num_planes)


def infeasible_start(A: np.ndarray, b: np.ndarray, rng) -> np.ndarray:
    """A Gaussian point outside the polyhedron, its spread grown until one is found.

    The spread starts at 4 per coordinate and grows by half after each miss. Raises
    RuntimeError after 200 misses.
    """
    scale = 4.0
    for _ in range(200):
        candidate = rng.standard_normal(A.shape[1]) * scale
        if np.any(_matmul(A, candidate) > b):
            return candidate
        scale *= 1.5
    raise RuntimeError("No infeasible start found.")


def problem_seed(num_planes: int, index: int) -> int:
    """The seed of problem number index in the size tier with num_planes half-spaces."""
    return 100000 + 1000 * num_planes + index


def problem(num_planes: int, num_dimensions: int, seed: int) -> tuple:
    """Generator 2's constraints and an infeasible start, drawn from one seeded stream.

    Returns (A, b, z).
    """
    rng = np.random.RandomState(seed)
    A, b = BoundedTestPlanes(num_planes, num_dimensions).draw(rng)
    return A, b, infeasible_start(A, b, rng)
