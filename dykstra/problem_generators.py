"""The random polyhedra of the SIAM paper and its supplement: the two generators of the
supplement's section Reproducibility, RandomTestPlanes (Generator 1, rejection sampling) and
BoundedTestPlanes (Generator 2, direct construction), and the two structured families of the
benchmark, WedgeTestPlanes, which the supplement's angle sweeps in stall_angles.py also draw,
and BoxSlabTestPlanes, on which Dykstra's algorithm has a stalling period.

Each class returns constraints A @ x <= b with one row of A per half-space. None normalises
its rows, because the law of the normals is stated for the rows as drawn, and the last
spanning row of BoundedTestPlanes, -Q 1, has length sqrt(p). The solvers normalise each row
and its offset internally, hence only the polyhedron matters to them.

The entries s tan(theta) of the laws of Generators 1 and 2, with theta uniform on [0, pi/2) and s a random sign,
are standard Cauchy variables. A row of independent Cauchy entries points near a coordinate
axis more often than a uniformly random direction does, thus the law does not make every
direction equally likely. tests/test_paper_outputs.py measures the excess at p = 2.

Every class draws from a random source passed in by the caller. problem() and the problem
method of the two families seed a legacy RandomState, whose streams numpy keeps fixed across
versions, so that a seed names the same problem on every installation.
"""

from collections import namedtuple

import numpy as np


# Generator 2's interior anchor is uniform on [-ANCHOR_HALF_WIDTH, ANCHOR_HALF_WIDTH]^p, and
# each offset clears it by a margin uniform on SLACK_RANGE
ANCHOR_HALF_WIDTH = 2.0
SLACK_RANGE = (0.2, 1.5)

# Chosen here: the geometry of the wedge, of the vertex of stall_angles.py, and of the box and
# slab, in the units of the coordinates. A corner, or the centre of a box, is uniform on
# [-CORNER_HALF_WIDTH, CORNER_HALF_WIDTH]^p, a box has half-width BOX_HALF_WIDTH, and a
# canonical start lies START_DISTANCE from its corner
START_DISTANCE = 1.0
BOX_HALF_WIDTH = 1.0
CORNER_HALF_WIDTH = 1.0

# Chosen here: the slab of BoxSlabTestPlanes. Its near boundary lies at a distance uniform on
# SLAB_GAP_RANGE from the box's vertex, its far boundary SLAB_WIDTH beyond, and the start lies
# beyond the vertex by a distance uniform on START_BEYOND_RANGE along every axis of the box, all
# in the units of the coordinates. The gaps hold the 0.447 of the 2-D example. The stall lasts
# about the start's distance over the gap. With these ranges each of the 60 seeds of
# benchmark.py's block, 20 in each of p = 2, 4, and 8, has 2 to 1029 stalling cycles, and on
# 58 of them the stalled point's squared error exceeds 1e-3
SLAB_GAP_RANGE = (0.05, 0.5)
SLAB_WIDTH = 0.1
START_BEYOND_RANGE = (5.0, 20.0)

# A member of a family with a known corner: its constraints A x <= b, its corner, the number of
# corner rows at the top of A, and the unit directions off the wedge's plane as columns, none
# for a vertex
Geometry = namedtuple("Geometry", "A b corner corner_rows free_directions")


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


def haar_rotation(rng, num_dimensions: int) -> np.ndarray:
    """A Haar-distributed orthogonal matrix, reflections included, drawn as the Q factor of a
    Gaussian matrix with the signs of the diagonal of R moved into it."""
    basis, upper = np.linalg.qr(rng.standard_normal((num_dimensions, num_dimensions)))
    return basis * np.sign(np.diag(upper))


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
    inside every half-space. Q is drawn by haar_rotation.
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
        A = _matmul(spanning, haar_rotation(rng, p).T)

        extra = self.num_planes - (p + 1)
        if extra > 0:
            A = np.vstack([A, cauchy_normals(rng, extra, p)])

        anchor = rng.uniform(-ANCHOR_HALF_WIDTH, ANCHOR_HALF_WIDTH, size=p)
        return A, _matmul(A, anchor) + rng.uniform(*SLACK_RANGE, size=self.num_planes)


class WedgeTestPlanes:
    """The wedge family: two half-spaces whose unit normals (cos(theta/2), +-sin(theta/2), 0,
    ..., 0) meet at angle theta, as in the article's lem:wedge, and the box
    |x_j| <= BOX_HALF_WIDTH on each of the remaining p - 2 coordinates.

    The 2 p - 2 rows hold the two wedge rows first. The box keeps the region bounded off the
    wedge's plane and stays inactive, hence the active normals span only that plane and every
    episode is singular for p >= 3. The family is built with its corner at the origin, then
    turned by a Haar-distributed rotation Q and moved to a corner v. Its start is
    canonical_start's, at START_DISTANCE from the corner, whose projection is known.

    A seed varies the problem through four draws, in the order of the stream: Q, v uniform on
    [-CORNER_HALF_WIDTH, CORNER_HALF_WIDTH]^p, the weights of the start's direction in the
    normal cone, and the start's offset along the box directions. theta and p fix everything
    else, the offsets of A included, which are A v plus BOX_HALF_WIDTH on the box rows.
    """

    def __init__(self, theta: float, num_dimensions: int) -> None:
        if num_dimensions < 2:
            raise ValueError("WedgeTestPlanes needs at least two dimensions.")
        self.theta = theta
        self.num_dimensions = num_dimensions
        self.num_planes = 2 * num_dimensions - 2

    def geometry(self, rng) -> Geometry:
        """Draw a wedge with its corner and the directions off its plane."""
        p, theta = self.num_dimensions, self.theta
        normals = np.zeros((2, p))
        normals[:, 0] = np.cos(theta / 2)
        normals[:, 1] = (np.sin(theta / 2), -np.sin(theta / 2))
        box = np.vstack([np.eye(p)[2:], -np.eye(p)[2:]])
        rotation = haar_rotation(rng, p)
        corner = rng.uniform(-CORNER_HALF_WIDTH, CORNER_HALF_WIDTH, size=p)
        A = _matmul(np.vstack([normals, box]), rotation.T)
        b = _matmul(A, corner) + np.concatenate([np.zeros(2), np.full(len(box), BOX_HALF_WIDTH)])
        return Geometry(A, b, corner, 2, rotation[:, 2:])

    def draw(self, rng) -> tuple[np.ndarray, np.ndarray]:
        """Draw the constraints of a wedge."""
        geometry = self.geometry(rng)
        return geometry.A, geometry.b

    def problem(self, seed: int) -> tuple:
        """A wedge and its canonical start, drawn from one seeded stream. Returns (A, b, z)."""
        rng = np.random.RandomState(seed)
        geometry = self.geometry(rng)
        z, _ = canonical_start(geometry, rng)
        return geometry.A, geometry.b, z


class BoxSlabTestPlanes:
    """The box-and-slab family: the square-and-line example of the article's stalling section,
    dykstra/square_line_example.py, lifted to p dimensions.

    The box |x_k| <= BOX_HALF_WIDTH gives 2 p rows, x_k <= BOX_HALF_WIDTH then
    -x_k <= BOX_HALF_WIDTH for k = 1, ..., p, as the square's right, left, top, and bottom. The
    slab gives two rows with opposite normals, u^T x <= high then -u^T x <= -low, as the
    line's below and above. The slab cuts a vertex v of the box off, whose sign vector sigma
    and the components w = sigma * u of the slab's normal are drawn, the latter standard
    normal and redrawn until the near boundary u^T x = low meets the box. low places that
    boundary at a distance uniform on SLAB_GAP_RANGE from v, and high lies SLAB_WIDTH beyond
    it. The start lies in the cone of points beyond v on every axis, by a distance uniform on
    START_BEYOND_RANGE along each. The whole is turned by a Haar-distributed rotation Q and
    moved to a centre uniform on [-CORNER_HALF_WIDTH, CORNER_HALF_WIDTH]^p.

    Dykstra's algorithm then stalls. The p faces at v are orthogonal and all active, hence
    each cycle lands on v, and the near boundary of the slab returns the point to one place.
    The correction of every face k with w_k < 0 drains by a fixed amount per cycle, and the
    point stays until the first of them is spent. The 2-D example's line is a slab of zero
    width. Lifted at zero width, 4 of 60 draws in p = 2, 4, and 8 were rejected by quadprog as
    inconsistent, which leaves them without a reference projection, hence SLAB_WIDTH > 0.

    A seed varies the problem through six draws, in the order of the stream: sigma, the
    distance of the near boundary from v, w with its redraws, Q, the centre, and the start's
    distances beyond v. p fixes everything else.
    """

    def __init__(self, num_dimensions: int) -> None:
        if num_dimensions < 2:
            raise ValueError("BoxSlabTestPlanes needs at least two dimensions.")
        self.num_dimensions = num_dimensions
        self.num_planes = 2 * num_dimensions + 2

    def _draw(self, rng) -> tuple:
        """The constraints, the vertex v, and the outward axes at v as columns."""
        p = self.num_dimensions
        signs = rng.choice([-1.0, 1.0], size=p)
        gap = rng.uniform(*SLAB_GAP_RANGE)
        # the near boundary meets the box when low is below the largest u^T x over the box,
        # i.e. when gap |w| is below 2 BOX_HALF_WIDTH times the sum of the |w_k| with w_k < 0.
        # the test asks for half of that, which leaves a slice of the box beyond the boundary
        while True:
            components = rng.standard_normal(p)
            spare = BOX_HALF_WIDTH * np.abs(components[components < 0]).sum()
            if gap * np.linalg.norm(components) < spare:
                break
        normal = signs * components
        low = BOX_HALF_WIDTH * components.sum() + gap * np.linalg.norm(normal)
        high = low + SLAB_WIDTH * np.linalg.norm(normal)
        box = np.repeat(np.eye(p), 2, axis=0) * np.tile([1.0, -1.0], p)[:, None]
        rows = np.vstack([box, normal, -normal])
        offsets = np.concatenate([np.full(2 * p, BOX_HALF_WIDTH), [high, -low]])
        rotation = haar_rotation(rng, p)
        centre = rng.uniform(-CORNER_HALF_WIDTH, CORNER_HALF_WIDTH, size=p)
        A = _matmul(rows, rotation.T)
        b = _matmul(A, centre) + offsets
        outward = rotation * signs
        return A, b, centre + BOX_HALF_WIDTH * outward.sum(axis=1), outward

    def draw(self, rng) -> tuple[np.ndarray, np.ndarray]:
        """Draw the constraints of a box and slab."""
        A, b, _, _ = self._draw(rng)
        return A, b

    def problem(self, seed: int) -> tuple:
        """A box and slab and its start, drawn from one seeded stream. Returns (A, b, z)."""
        rng = np.random.RandomState(seed)
        A, b, vertex, outward = self._draw(rng)
        beyond = rng.uniform(*START_BEYOND_RANGE, size=self.num_dimensions)
        return A, b, vertex + _matmul(outward, beyond)


def corner_normals(geometry: Geometry) -> np.ndarray:
    """The unit normals of the corner rows."""
    rows = geometry.A[:geometry.corner_rows]
    return rows / np.linalg.norm(rows, axis=1, keepdims=True)


def canonical_start(geometry: Geometry, rng,
                    distance: float = START_DISTANCE) -> tuple[np.ndarray, np.ndarray]:
    """A canonical start at distance from the corner, in the units of the coordinates, and its
    projection.

    The start is the corner plus distance times the normalised combination of the corner
    normals with weights uniform on the probability simplex, hence the projection is the
    corner. A wedge start also moves off the plane by an offset uniform in half the box, which
    the projection keeps.
    """
    normals = corner_normals(geometry)
    direction = rng.dirichlet(np.ones(len(normals))) @ normals
    free = geometry.free_directions.shape[1]
    reference = geometry.corner + geometry.free_directions @ rng.uniform(
        -BOX_HALF_WIDTH / 2, BOX_HALF_WIDTH / 2, size=free)
    return reference + distance * direction / np.linalg.norm(direction), reference


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
