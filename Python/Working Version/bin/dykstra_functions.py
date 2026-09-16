"""
This module implements Dykstra's algorithm for projecting a point onto the
intersection of convex sets (specifically, half-spaces).

Functions:
- normalise(normal, offset):
    Normalises half space normal and constant offset.
- is_in_half_space(point, unit_normal, constant_offset):
    Checks if a point lies within a single half-space.
- project_onto_half_space(point, normal, offset):
    Projects a given point onto a single half-space.
- delete_inactive_half_spaces(z, A, b):
    Deletes inactive half-spaces and returns updated matrices.
- find_optimal_solution(point, A, b, dimensions: int):
    Computes the projection using QP optimisation (quadprog)
- beta_check(point, A, b):
    Selects a value of beta based on whether the point lies within the
    intersection of half-spaces.
"""


import numpy as np
from gradient import quadprog_solve_qp # needed to track error (V4)


def normalise(normal: np.ndarray, offset: np.ndarray) -> tuple:
    """
    Normalises half space normal and constant offset.

    Args:
        normal: Normal vector of the half space.
        offset: Constant offset of the half space.

    Returns:
        tuple: Unit normal vector and normalised offset.
    """

    # Obtain normal
    norm = np.linalg.norm(normal)
    if norm == 0:  # me and my homies hate division by 0
        raise ValueError("Warning: Zero-norm normal vector encountered.")
    # Normalise normal
    unit_normal = normal / norm
    # Normalise offset
    constant_offset = offset / norm

    return unit_normal, constant_offset


def is_in_half_space(point: np.ndarray, unit_normal: np.ndarray,
                            constant_offset: np.ndarray) -> bool:
    """
    Checks if a point lies within a single half space.

    Args:
        point: Point to check.
        unit_normal: *Unit* normal vector of the half space.
        constant_offset: *Normalised* offset of the half space.

    Returns:
        bool: True if point is within the half space, else False.
    """

    # Find dot product
    dp = np.dot(point, unit_normal)

    # Return boolean
    return dp <= constant_offset


def project_onto_half_space(point: np.ndarray, normal: np.ndarray,
                            offset: np.ndarray) -> np.ndarray:
    """Projects a point onto a single half space 'H_i'.
    A half space is defined by H_i := {x | <x,n_i> <= b_i}, with boundary
    B_i := {x | <x,n_i> = b_i}, and the projection of a point z onto H_i is
    given by: P_H_i(z) = z - (<z,n_i> - b_i)*n_i if z is outside H_i, where:
    point = z, unit_normal  = n_i, constant_offset = b_i

    Args:
        point: Point to project.
        normal: Normal vector of the half space.
        offset: Constant offset of the half space.

    Returns:
        np.ndarray: Projected point.
    """

    # Normalise
    unit_normal, constant_offset = normalise(normal, offset)

    # Check if point is already in the half space
    if is_in_half_space(point, unit_normal, constant_offset):
        return point
    # Project point onto the half space's boundary
    else:
        boundary_projection = (point - (np.dot(point, unit_normal)
                                        - constant_offset) * unit_normal)
        return boundary_projection


def delete_inactive_half_spaces(z: np.ndarray, A: np.ndarray, b: np.ndarray)\
        -> tuple:
    """
    Deletes inactive half spaces.

    Args:
        z: Point used to check for inactive half spaces.
        A: Matrix of normal vectors.
        b: Vector of constant offsets.

    Returns:
        tuple: Updated matrix of normal vectors and vector of constant offsets.
    """

    # Initialise empty removal mask
    indices_to_remove = np.zeros_like(b, dtype=bool)

    # Update mask
    for m, (normal, offset) in enumerate(zip(A, b)):
        # Normalise
        unit_normal, constant_offset = normalise(normal, offset)
        # Check if half space is inactive
        if is_in_half_space(z, unit_normal, constant_offset):
            indices_to_remove[m] = True

    # Remove inactive halfspaces using mask
    new_A = A[~indices_to_remove]
    new_b = b[~indices_to_remove]

    return new_A, new_b


def find_optimal_solution(point: np.ndarray, A: np.ndarray, b: np.ndarray,
                          dimensions: int) -> np.ndarray:
    """
    Solves a quadratic programming problem to find the optimal solution that
    minimises the Euclidean distance between a given point and a target,
    subject to linear constraints.

    The function solves the following optimisation problem:
    min_x ∥x − point∥^2 subject to Gx <= h, where G and h represent
    the constraints.

    Parameters:
    -----------
    point : A 1D array representing the target point in space for the
            optimisation.

    A : A 2D array representing the constraint matrix G in the quadratic
            programming formulation.

    b : A 1D array representing the constraint vector h in the
            quadratic programming formulation.

    dimensions : The dimensionality of the space in which
            the optimisation is being performed.

    Returns:
    --------
    The optimal projection of the point, subject to the constraints,
    which minimises the quadratic objective function.

    Notes:
    ------
    This function uses the `quadprog_solve_qp` method to solve
    the quadratic programming problem.
    The quadratic objective function is reformulated as:
        min_x ∥Ix − point∥^2
    where I is the identity matrix, P = 2 * I^T I, and q = −2 * I^T point.

    References:
    -----------
    The formulation of this quadratic programming problem is based on the following reference:
    https://scaron.info/blog/quadratic-programming-in-python.html
    """

    # Formulate the problem
    identity = np.eye(dimensions)
    # @ command is recommended
    P = 2 * np.matmul(identity.T, identity)
    q = -2 * np.matmul(identity.T, point)
    G = A
    h = b

    # Find projection using quadprog
    actual_projection = quadprog_solve_qp(P, q, G, h)

    return actual_projection

def beta_check(point: np.ndarray, A: np.ndarray, b: np.ndarray):
    """
    Selects a value of beta based on whether the passed point lies
    within the intersection of half-spaces.

    Args:
        point: Point to check.
        A: Matrix of normal vectors.
        b: Vector of constant offsets.

    Returns:
        int: 1 if the point is within the intersection, else 0.
    """

    # Initialise beta
    beta = 1
    not_in_intersection = False  # initialise boolean
    # I encountered numerical error problems
    rounded_point = np.around(point, decimals=10) # round to 10 decimal places
    for _, (normal, offset) in enumerate(zip(A, b)):
        # Normalise
        unit_normal, constant_offset = normalise(normal, offset)
        # Check if we are in the half space to update beta
        if not is_in_half_space(rounded_point, unit_normal, constant_offset):
            not_in_intersection = True
    # Choose Beta = 0 if not in intersection (MAP)
    if not_in_intersection:
        beta = 0
    return beta
