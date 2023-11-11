"""
This subpackage contains the rotation classes and functions.

It is helpful to calculate direct and inverse problems for Rotations along a given axis.

===
License

Copyright 2023 Dario Loi

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.

"""

from enum import Enum
from typing import Optional, Tuple, Type, Union

import numpy as np
from sympy import Matrix, cos, sin, symbols


def direct_rot_mat(theta: float, axis: np.ndarray) -> np.ndarray:
    """
    This function returns a rotation matrix for a given angle and axis.

    Parameters
    ----------
    theta : float
        Rotation angle in radians.
    axis : np.ndarray
        Rotation axis.

    Returns
    -------
    np.ndarray
        Rotation matrix.
    """

    axis = axis.squeeze()

    assert axis is not None, "Axis cannot be None"
    assert axis.shape == (3,), f"Axis must be a 3D vector, shape is {axis.shape}"
    assert not np.isclose(np.linalg.norm(axis), 0), "Axis should not be singular"

    axis = axis / np.linalg.norm(axis)
    ux, uy, uz = axis
    c = np.cos(theta)
    s = np.sin(theta)
    rot_mat = np.array(
        [
            [
                c + ux**2 * (1 - c),
                ux * uy * (1 - c) - uz * s,
                ux * uz * (1 - c) + uy * s,
            ],
            [
                uy * ux * (1 - c) + uz * s,
                c + uy**2 * (1 - c),
                uy * uz * (1 - c) - ux * s,
            ],
            [
                uz * ux * (1 - c) - uy * s,
                uz * uy * (1 - c) + ux * s,
                c + uz**2 * (1 - c),
            ],
        ]
    )
    return rot_mat


def inverse_rot_mat(
    R: np.ndarray,
) -> Tuple[Tuple[float, float], Optional[np.ndarray]]:
    """
    This function returns the angle and axis of rotation for a given rotation matrix.

    If the angle is 0, the axis is undefined and None is returned.

    If the angle is pi or -pi, there are two possible solutions for the axis, and both are returned.

    Otherwise, there are two possible solutions for the axis, and both are returned (order matters)

    Parameters
    ----------
    R : np.ndarray
        Rotation matrix.

    Returns
    -------
    Tuple[Tuple[float, float], Optional[np.ndarray]]
        Tuple containing the angle(s) and axis of rotation in radians.
        If the axis is undefined, returns None.
    """

    assert R is not None, "R cannot be None"
    assert R.shape == (3, 3), "R must be a 3x3 matrix"
    assert np.isclose(
        np.linalg.det(R), 1
    ), f"R must be a rotation matrix (orthonomal with determinant 1), determinant is {np.linalg.det(R)}"
    assert np.allclose(
        np.dot(R.T, R), np.eye(3)
    ), f"R must be a rotation matrix (orthonomal with determinant 1), R^T @ R != I"

    s = (
        1
        / 2
        * np.sqrt(
            (R[2, 1] - R[1, 2]) ** 2
            + (R[0, 2] - R[2, 0]) ** 2
            + (R[1, 0] - R[0, 1]) ** 2
        )
    )
    theta1, theta2 = np.arctan2(2 * s, np.trace(R) - 1), np.arctan2(
        -2 * s, np.trace(R) - 1
    )

    assert np.isclose(theta1, theta2) or np.isclose(
        theta1, -theta2
    ), "[BUG] Theta1 and theta2 should be equal or opposite"

    if np.isclose(theta1, 0) and np.isclose(theta2, 0):
        return (0, 0), None
    elif (np.isclose(theta1, np.pi) and np.isclose(theta2, -np.pi)) or (
        np.isclose(theta1, -np.pi) and np.isclose(theta2, np.pi)
    ):
        r1 = np.array(
            [
                np.sqrt((R[0, 0] + 1) / 2),
                np.sqrt((R[1, 1] + 1) / 2),
                np.sqrt((R[2, 2] + 1) / 2),
            ]
        )

        r2 = np.array(
            [
                -np.sqrt((R[0, 0] + 1) / 2),
                -np.sqrt((R[1, 1] + 1) / 2),
                -np.sqrt((R[2, 2] + 1) / 2),
            ]
        )

        return (theta1, theta2), (r1, r2)
    else:  # theta != 0 and theta != pi
        sin1 = np.sin(theta1)
        sin2 = np.sin(theta2)

        r1 = np.array(
            [
                (R[2, 1] - R[1, 2]) / (2 * sin1),
                (R[0, 2] - R[2, 0]) / (2 * sin1),
                (R[1, 0] - R[0, 1]) / (2 * sin1),
            ]
        )

        r2 = np.array(
            [
                (R[2, 1] - R[1, 2]) / (2 * sin2),
                (R[0, 2] - R[2, 0]) / (2 * sin2),
                (R[1, 0] - R[0, 1]) / (2 * sin2),
            ]
        )

        return (theta1, theta2), (r1, r2)


def direct_rpy_separate(
    roll: float, pitch: float, yaw: float
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    This function returns a rotation matrix for a given roll, pitch and yaw.

    Parameters
    ----------
    roll : float
        Roll angle in radians.
    pitch : float
        Pitch angle in radians.
    yaw : float
        Yaw angle in radians.

    Returns
    -------
    np.ndarray
        Rotation matrix.
    """

    assert roll is not None, "Roll cannot be None"
    assert pitch is not None, "Pitch cannot be None"
    assert yaw is not None, "Yaw cannot be None"

    R_roll = direct_rot_mat(roll, np.array([1, 0, 0]))
    R_pitch = direct_rot_mat(pitch, np.array([0, 1, 0]))
    R_yaw = direct_rot_mat(yaw, np.array([0, 0, 1]))
    return R_roll, R_pitch, R_yaw


def direct_rpy(roll: float, pitch: float, yaw: float) -> np.ndarray:
    """
    This function returns a rotation matrix for a given roll, pitch and yaw.

    Parameters
    ----------
    roll : float
        Roll angle in radians.
    pitch : float
        Pitch angle in radians.
    yaw : float
        Yaw angle in radians.

    Returns
    -------
    np.ndarray
        Rotation matrix.
    """

    assert roll is not None, "Roll cannot be None"
    assert pitch is not None, "Pitch cannot be None"
    assert yaw is not None, "Yaw cannot be None"

    mats = direct_rpy_separate(roll, pitch, yaw)

    return mats[2] @ mats[1] @ mats[0]


class RPY_RES(Enum):
    SUM = "phi + psi"
    SUB = "phi - psi"


class AXIS(Enum):
    X = "x"
    Y = "y"
    Z = "z"


def inverse_rpy(
    R: np.ndarray,
) -> Tuple[
    Tuple[float, float, float, Optional[RPY_RES]],
    Tuple[float, float, float, Optional[RPY_RES]],
]:
    """inverse_rpy Calculates the roll, pitch and yaw angles for a given rotation matrix.

    Parameters
    ----------
    R : np.ndarray
        Rotation matrix.

    Returns
    -------
    Tuple[Tuple[float, float, float, Optional[RPY_RES]], Tuple[float, float, float, Optional[RPY_RES]]]
        Two tuples of floats representing the roll, pitch and yaw angles in radians, and the ambiguity resolution.

    """

    assert R is not None, "R cannot be None"
    assert R.shape == (3, 3), "R must be a 3x3 matrix"
    assert np.isclose(
        np.linalg.det(R), 1
    ), f"R must be a rotation matrix (orthonomal with determinant 1), determinant is {np.linalg.det(R)}"
    assert np.allclose(
        np.dot(R.T, R), np.eye(3)
    ), "R must be a rotation matrix (orthonomal with determinant 1), R^T @ R != I"

    ctetsq = np.power(R[2, 1], 2) + np.power(R[2, 2], 2)
    ctet = np.sqrt(ctetsq)

    pitch1 = np.arctan2(-R[2, 0], ctet)
    pitch2 = np.arctan2(-R[2, 0], -ctet)

    def get_pitch_sol(mat, pitch):
        if np.isclose(pitch, np.pi / 2):
            yawminroll = np.arctan2(mat[1, 2], mat[0, 2])

            # return two arbitrary solutions that satisfy the constraint roll - yaw = rollminyaw
            yaw = yawminroll + 0.0
            roll = 0.0

            return roll, pitch, yaw, RPY_RES.SUB
        elif np.isclose(pitch, -np.pi / 2):
            yawplusroll = np.arctan2(-mat[1, 2], mat[1, 1])

            yaw = yawplusroll - 0.0
            roll = 0.0

            return roll, pitch, yaw, RPY_RES.SUM
        else:
            assert False, "Singularity case unhandled"

    if not np.isclose(ctetsq, 0, atol=1e-12):
        cos1 = np.cos(pitch1)
        cos2 = np.cos(pitch2)

        roll1 = np.arctan2(R[2, 1] / ctet, R[2, 2] / cos1)
        roll2 = np.arctan2(R[2, 1] / ctet, R[2, 2] / cos2)

        yaw1 = np.arctan2(R[1, 0] / ctet, R[0, 0] / cos1)
        yaw2 = np.arctan2(R[1, 0] / ctet, R[0, 0] / cos2)

        return (roll1, pitch1, yaw1, None), (roll2, pitch2, yaw2, None)
    else:
        return get_pitch_sol(R, pitch1), get_pitch_sol(R, pitch2)


def gen_roll(symbol):
    return Matrix(
        [[1, 0, 0], [0, cos(symbol), -sin(symbol)], [0, sin(symbol), cos(symbol)]]
    )


def gen_pitch(symbol):
    return Matrix(
        [[cos(symbol), 0, sin(symbol)], [0, 1, 0], [-sin(symbol), 0, cos(symbol)]]
    )


def gen_yaw(symbol):
    return Matrix(
        [[cos(symbol), -sin(symbol), 0], [sin(symbol), cos(symbol), 0], [0, 0, 1]]
    )


def direct_symbolic(
    axes: Tuple[AXIS, AXIS, AXIS], use_greek_symbols: bool = True
) -> Matrix:
    """direct_symbolic Generates a symbolic rotation matrix for a given sequence of rotations along three generic axes.

    Parameters
    ----------
    axes : Tuple[AXIS, AXIS, AXIS]
        A triple of Enums representing the axes of rotation.
    use_greek_symbols : bool, optional
        Whether to use phi, theta, psi for the rotations or to state them explicityl as
        roll, pitch, and yaw, by default True

    Returns
    -------
    Matrix
        A symbolic rotation matrix.
    """

    from sympy import init_printing

    init_printing()

    def gen_mat(axis: AXIS, i: int = 0):
        if AXIS(axis) == AXIS.X:
            if use_greek_symbols:
                return gen_roll(f"phi_{i}")
            else:
                return gen_roll(f"roll_{i}")

        elif AXIS(axis) == AXIS.Y:
            if use_greek_symbols:
                return gen_pitch(f"theta_{i}")
            else:
                return gen_pitch(f"pitch_{i}")
        elif AXIS(axis) == AXIS.Z:
            if use_greek_symbols:
                return gen_yaw(f"psi_{i}")
            else:
                return gen_yaw(f"yaw_{i}")
        else:
            assert False, "Invalid axis"

    mats = [gen_mat(axis, i) for i, axis in enumerate(axes)]

    R = mats[2] @ mats[1] @ mats[0]

    return R


def get_symbols(
    axes: Tuple[AXIS, AXIS, AXIS], use_greek_symbols: bool = True
) -> Tuple[Type[symbols], Type[symbols], Type[symbols]]:
    symbols = []

    for i, ax in enumerate(axes):
        match ax:
            case AXIS.X:
                if use_greek_symbols:
                    symbols.append(f"phi_{i}")
                else:
                    symbols.append(f"roll_{i}")
            case AXIS.Y:
                if use_greek_symbols:
                    symbols.append(f"theta_{i}")
                else:
                    symbols.append(f"pitch_{i}")
            case AXIS.Z:
                if use_greek_symbols:
                    symbols.append(f"psi_{i}")
                else:
                    symbols.append(f"yaw_{i}")
            case _:
                assert False, "Invalid axis"
    return symbols


def direct_generic(
    axes: Tuple[AXIS, AXIS, AXIS],
    angles: Tuple[float, float, float],
    get_symbolic: bool = False,
) -> Tuple[np.ndarray, Optional[Matrix]]:
    R_sym = direct_symbolic(axes)
    symbols = get_symbols(axes)

    R = np.array(
        [
            [R_sym[i, j].subs({sym: a for sym, a in zip(symbols, angles)})]
            for i in range(3)
            for j in range(3)
        ]
    )

    if get_symbolic:
        return R, R_sym
    else:
        return R


def inverse_generic(
    axes: Tuple[AXIS, AXIS, AXIS], R: np.ndarray
) -> Optional[Tuple[float, float, float]]:
    """inverse_generic Invert a rotation matrix obtained from the composition of a rotation
    along three generic axes.

    Parameters
    ----------
    axes : Tuple[AXIS, AXIS, AXIS]
        A triple of Enums representing the axes of rotation.

    Returns
    -------
    Optional(Tuple[float, float, float])
        A tuple of floats representing the roll, pitch and yaw angles in radians.
        If the matrix is singular, returns None.
    """
    R_sym = direct_symbolic(axes)

    equations = []
    for i in range(3):
        for j in range(3):
            equations.append(R_sym[i, j] - R[i, j])

    try:
        solutions = nsolve(
            equations, (roll, pitch, yaw), (0, 0, 0), dict=True
        )  # Solve the equations
    except:
        return None  # Singularity!

    assert solutions is not None, "[BUG] Solutions should not be None"
    assert solutions != [], "[BUG] Solutions should not be empty"

    return solutions


if __name__ == "__main__":
    from rich import print
    from sympy import pprint

    print(inverse_rpy(direct_rpy(0.1, 0.2, 0.3)))

    pprint(
        direct_generic((AXIS.X, AXIS.Y, AXIS.Z), (0.1, 0.2, 0.3), get_symbolic=True)[1]
    )
