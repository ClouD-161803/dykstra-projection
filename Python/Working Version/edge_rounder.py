"""This module contains a function to round the corners of a box.
Explanation:

    1. Straight Sides:
       * Defines constraints for the four straight sides of the box using
        outward-facing normal vectors.
       * Calculates the constant offsets `b_straight` based on the position of
        the center and dimensions of the box.

    2. Rounded Corners:
       * Iterates over each of the four corners of the box.
       * For each corner, it further iterates `corner_segments` times to
       define the linear segments approximating the rounded corner.
       * Calculates the angle `angle` for each segment, evenly spaced within
        a quarter-circle.
       * Computes the normal vector `normal` for the segment based on the
        angle and the corner's quadrant.
       * Determines the radius of the rounded corner as the minimum of half
        the width and half the height of the box.
       * Calculates the offset `offset` for the segment based on the position
        of the center, normal vector, and radius.
       * Appends the `normal` and `offset` to lists
        `A_rounded` and `b_rounded`, respectively.

    3. Combination:
       * Converts the lists `A_rounded` and `b_rounded` to NumPy arrays.
       * Vertically stacks the straight side constraints
        (`A_straight`, `b_straight`) and the rounded corner constraints
        (`A_rounded`, `b_rounded`) to form the final `A` and `b`.

    The resulting `A` and `b` can be used in optimization or projection algorithms
    that require half-space constraints to represent the rounded box.
"""


import numpy as np


def rounded_box_constraints(center, width, height, corner_segments=5):
    """
    Generates half-space constraints (A, b) that define a box with rounded corners.

    This function approximates the rounded corners of a box using multiple
    linear segments (half-spaces).
    The more segments used, the smoother the corners appear.

    Args:
        center: A tuple (x, y) representing the coordinates of the box's center.
        width: The width of the box.
        height: The height of the box.
        corner_segments: The number of linear segments used to approximate
        each rounded corner (default is 5).

    Returns:
        A tuple (A, b) where:

        * A: A NumPy array where each row represents the outward-facing normal
            vector of a half-space constraint.
        * b: A NumPy array where each element represents the constant offset
            of a corresponding half-space constraint.
    """

    half_width = width / 2
    half_height = height / 2

    # Constraints for the straight sides (outward-facing normals)
    A_straight = np.array([
        [-1, 0],  # Left side
        [1, 0],   # Right side
        [0, -1],  # Bottom side
        [0, 1]    # Top side
    ])
    b_straight = np.array([
        -center[0] + half_width,
        center[0] + half_width,
        -center[1] + half_height,
        center[1] + half_height
    ])

    # Constraints for the rounded corners
    A_rounded = []
    b_rounded = []
    for corner in [(1, 1), (-1, 1), (-1, -1), (1, -1)]:  # Four corners
        for i in range(corner_segments):
            angle = i / corner_segments * np.pi / 2  # Adjusted angle calculation
            normal = np.array([corner[0] * np.cos(angle), corner[1] * np.sin(angle)])
            # Radius of the rounded corner is the minimum of half_width and half_height
            radius = min(half_width, half_height)
            offset = np.dot(normal, center) + radius
            A_rounded.append(normal)
            b_rounded.append(offset)
    # This inevitably generates an extra copy of the original box constraints

    A_rounded = np.array(A_rounded)
    b_rounded = np.array(b_rounded)

    # Combine all constraints
    A = np.vstack([A_straight, A_rounded])
    b = np.hstack([b_straight, b_rounded])

    return A, b