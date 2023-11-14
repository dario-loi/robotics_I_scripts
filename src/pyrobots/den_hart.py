from enum import Enum
from typing import Optional, Tuple, Type, Union

import numpy as np
from sympy import (
    Matrix,
    cos,
    nsimplify,
    pi,
    pprint,
    simplify,
    sin,
    solve,
    symbols,
    sympify,
    trigsimp,
)


def mat_dh_symbolic(table: np.array):
    """
    Given a table of Denavit-Hartenberg parameters, returns the transformation matrix
    from the base frame to the end effector frame

    Parameters:
    table (numpy.ndarray): a table of Denavit-Hartenberg parameters

    Returns:
    T (numpy.ndarray): the transformation matrix from the base frame to the end effector frame
    """

    # check if table is Nx4
    assert table.shape[1] == 4, "Table is not Nx4"

    # create transformation matrix
    T = np.eye(4)

    # iterate through rows of table
    for i in range(4):
        a, alpha, d, theta = table[i, :]

        def convert(x):
            if x.isnumeric():
                return float(x)
            else:
                try:
                    n = float(sympify(x))
                    # check if n is close to 0
                    if abs(n) < 1e-5:
                        return 0
                    else:
                        return n
                except:
                    return symbols(x)

        a = convert(a)
        alpha = convert(alpha)
        d = convert(d)
        theta = convert(theta)

        """
        if not a.isnumeric():
            a = symbols(a)
        else:
            a = float(a)
        if not alpha.isnumeric():
            alpha = symbols(alpha)
        else:
            alpha = float(alpha)
        if not d.isnumeric():
            d = symbols(d)
        else:
            d = float(d)
        if not theta.isnumeric():
            theta = symbols(theta)
        else:
            theta = float(theta)
       """

        # create transformation matrix for each row
        T_i = Matrix(
            [
                [
                    cos(theta),
                    -sin(theta) * cos(alpha),
                    sin(theta) * sin(alpha),
                    a * cos(theta),
                ],
                [
                    sin(theta),
                    cos(theta) * cos(alpha),
                    -cos(theta) * sin(alpha),
                    a * sin(theta),
                ],
                [0, sin(alpha), cos(alpha), d],
                [0, 0, 0, 1],
            ]
        )

        # multiply transformation matrices
        T = simplify(T) @ simplify(T_i)

    return nsimplify(simplify(trigsimp(T)), tolerance=1e-5)


def mat_dh_numeric(table: np.array):
    # check if table is Nx4
    assert table.shape[1] == 4, "Table is not Nx4"

    # create transformation matrix
    T = np.eye(4)

    # iterate through rows of table
    for i in range(4):
        # create transformation matrix for each row
        a, alpha, d, theta = table[i, :]
        T_i = np.array(
            [
                [
                    np.cos(theta),
                    -np.sin(theta) * np.cos(alpha),
                    np.sin(theta) * np.sin(alpha),
                    a * np.cos(theta),
                ],
                [
                    np.sin(theta),
                    np.cos(theta) * np.cos(alpha),
                    -np.cos(theta) * np.sin(alpha),
                    a * np.sin(theta),
                ],
                [0, np.sin(alpha), np.cos(alpha), d],
                [0, 0, 0, 1],
            ]
        )

        # multiply transformation matrices
        T = T @ T_i

    return T


if __name__ == "__main__":
    dh_table = np.array(
        [[0, 0, "q1", 0], ["N", "-pi/2", 0, "q2"], [0, 0, "q3", 0], [0, 0, 0, "q4"]]
    )

    pprint(mat_dh_symbolic(dh_table))
