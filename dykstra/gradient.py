"""
This module provides a function to solve quadratic programming (QP)
problems using the quadprog library.

It leverages the `quadprog.solve_qp` function to find the optimal solution
to a QP problem.

Functions:
- quadprog_solve_qp(P, q, G, h, A=None, b=None):
Solves a quadratic program using quadprog.

NOTE: You need to have quadprog installed in order to use this module:
> pip install quadprog

Further documentation at:
https://scaron.info/blog/quadratic-programming-in-python.html
"""


import numpy as np
import quadprog


def quadprog_solve_qp(P: np.ndarray, q: np.ndarray,
                      G: np.ndarray, h: np.ndarray,
                      A: np.ndarray=None, b: np.ndarray=None) -> np.ndarray:
    """
        Solves a quadratic program of the form:

        minimize    (1/2)*x.T*P*x + q.T*x
        subject to  G*x <= h
                    A*x == b

        This function uses the `quadprog.solve_qp` function from the quadprog
        library to solve the QP problem.

        Args:
        P: Quadratic term in the objective function.
        q: Linear term in the objective function.
        G: Inequality constraint matrix.
        h: Inequality constraint vector.
        A: (Optional) Equality constraint matrix.
        b: (Optional) Equality constraint vector.

    Returns:
        The optimal solution 'x' to the quadratic program.
    """

    qp_G = .5 * (P + P.T)   # make sure P is symmetric
    qp_a = -q
    if A is not None:
        qp_C = -np.vstack([A, G]).T
        qp_b = -np.hstack([b, h])
        meq = A.shape[0]
    else:  # no equality constraint
        qp_C = -G.T
        qp_b = -h
        meq = 0
    return quadprog.solve_qp(qp_G, qp_a, qp_C, qp_b, meq)[0]