""" 
This subpackage contains the rotation classes and functions.

It is helpful to calculate direct and inverse problems for Rotations along a given axis.
"""

import numpy as np
from typing import Type, Tuple, Optional, Union


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
) -> Tuple[Union[float, Tuple[float, float]], Optional[np.ndarray]]:
    """
    This function returns the angle and axis of rotation for a given rotation matrix.

    Parameters
    ----------
    R : np.ndarray
        Rotation matrix.

    Returns
    -------
    Tuple[Union[float, Tuple[float, float]], Optional[Vec3]]
        Rotation angle in radians and rotation axis, the first can have either one or two values, the latter is None (undefined) if the angle is 0.
    """

    s = (
        1
        / 2
        * np.sqrt(
            (R[2, 1] - R[1, 2]) ** 2
            + (R[0, 2] - R[2, 0]) ** 2
            + (R[1, 0] - R[0, 1]) ** 2
        )
    )
    theta = np.arctan2(2 * s, np.trace(R) - 1)

    if np.isclose(theta, 0):  # undefined axis
        return 0, None
    elif np.isclose(np.abs(theta), np.pi):
        r = np.array(
            [
                np.sqrt((R[0, 0] + 1) / 2),
                np.sqrt((R[1, 1] + 1) / 2),
                np.sqrt((R[2, 2] + 1) / 2),
            ]
        )

        return (-theta, theta), r
    else:  # theta != 0 and theta != pi
        r = np.array(
            [
                (R[2, 1] - R[1, 2]) / (2 * s),
                (R[0, 2] - R[2, 0]) / (2 * s),
                (R[1, 0] - R[0, 1]) / (2 * s),
            ]
        )
        return theta, r


if __name__ == "__main__":
    inverse_rot_mat(
        np.array(
            [
                [0.70710678, -0.70710678, 0.0],
                [0.70710678, 0.70710678, 0.0],
                [0.0, 0.0, 1.0],
            ]
        )
    )
