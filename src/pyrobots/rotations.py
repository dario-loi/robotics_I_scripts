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

from typing import Optional, Tuple, Type, Union

import numpy as np


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

    assert axis is not None, "Axis cannot be None"
    assert axis.shape == (3,), "Axis must be a 3D vector"
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

    assert R is not None, "R cannot be None"
    assert R.shape == (3, 3), "R must be a 3x3 matrix"
    assert np.isclose(
        np.linalg.det(R), 1
    ), "R must be a rotation matrix (orthonomal with determinant 1)"
    assert (
        np.dot(R, R.T).all() == np.eye(3).all()
    ), "R must be a rotation matrix (orthonomal with determinant 1)"

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

    return (
        direct_rot_mat(roll, np.array([1, 0, 0]))
        @ direct_rot_mat(pitch, np.array([0, 1, 0]))
        @ direct_rot_mat(yaw, np.array([0, 0, 1]))
    )


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
