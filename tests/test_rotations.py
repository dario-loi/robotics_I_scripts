"""
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
import pytest
from rich import print

from pyrobots import rotations as rot
from pyrobots.rotations import RPY_RES


def test_direct() -> None:
    assert (
        rot.direct_rot_mat(0, np.array([1, 0, 0])).all() == np.eye(3).all()
    ), "Rotation matrix for 0 radians about x-axis should be identity"
    assert (
        rot.direct_rot_mat(np.pi, np.array([1, 0, 0])).all()
        == np.array([[1, 0, 0], [0, -1, 0], [0, 0, -1]]).all()
    ), "Rotation matrix for pi radians about x-axis should be correct"
    assert (
        rot.direct_rot_mat(np.pi / 2, np.array([1, 0, 0])).all()
        == np.array([[1, 0, 0], [0, 0, -1], [0, 1, 0]]).all()
    ), "Rotation matrix for pi/2 radians about x-axis should be correct"


def test_inverse() -> None:
    assert rot.inverse_rot_mat(np.eye(3))[0] == 0
    assert rot.inverse_rot_mat(np.eye(3))[1] == None

    # Accessing the first element of the tuple should return the angle, which should have two values
    # we check that the second value is pi
    theta, axis = rot.inverse_rot_mat(np.array([[1, 0, 0], [0, -1, 0], [0, 0, -1]]))

    assert isinstance(theta, tuple) and len(theta) == 2
    t1, t2 = theta

    assert axis is not None, "axis should not be None"
    assert np.isclose(t1, -np.pi), "t1 should be -pi"
    assert np.isclose(t2, np.pi), "t2 should be pi"
    assert axis.all() == np.array([1, 0, 0]).all(), "axis should be [1, 0, 0]"
    assert t1 == -t2, "t1 and t2 should be negatives of each other"


def test_compose_should_be_identity() -> None:
    # General case

    R = rot.direct_rot_mat(np.pi / 4, np.array([0, 1, 0]))
    theta, axis = rot.inverse_rot_mat(R)

    assert np.isclose(theta, np.pi / 4), "theta should be pi/4"
    assert axis is not None, "axis should not be None"
    assert axis.all() == np.array(np.array([0, 1, 0])).all(), "axis should be [0, 1, 0]"

    # Special case (theta = +- pi)

    R = rot.direct_rot_mat(-np.pi, np.array([1, 0, 0]))
    theta, axis = rot.inverse_rot_mat(R)

    assert np.isclose(
        theta, np.array([-np.pi, np.pi])
    ).all(), "theta should be -pi or pi"
    assert axis is not None, "axis should not be None"
    assert axis.all() == np.array([1, 0, 0]).all(), "axis should be [1, 0, 0]"

    # Special case (theta = 0)

    R = rot.direct_rot_mat(0, np.array([0, 0, 1]))
    theta, axis = rot.inverse_rot_mat(R)

    assert np.isclose(theta, 0), "theta should be 0"
    assert axis == None, "axis should be None"


if __name__ == "__main__":
    test_direct()
    test_inverse()
    test_compose_should_be_identity()

    print("All tests passed.")


def test_axis_scale_invariant() -> None:
    R_1 = rot.direct_rot_mat(np.sqrt(2) * np.pi, np.array([1, 0, 0.5]))
    R_2 = rot.direct_rot_mat(np.sqrt(2) * np.pi, np.array([3, 0, 1.5]))

    assert np.allclose(R_1, R_2), "Rotation matrices should be equal"


def test_huge_axis():
    axis = np.array([1e15, 1e15, -1e15])

    R = rot.direct_rot_mat(2 * np.pi, axis)

    assert np.allclose(R, np.eye(3)), "Rotation matrix should be identity"


def test_huge_angle():
    # angles break down close to 1e8
    R = rot.direct_rot_mat(2 * np.pi * 1e7, np.array([0, 0, 1]))

    assert np.allclose(R, np.eye(3)), "Rotation matrix should be identity"


def test_huge_angle_and_axis() -> None:
    axis = np.array([1e15, 1e15, -1e15])

    R = rot.direct_rot_mat(2 * np.pi * 1e7, axis)

    assert np.allclose(R, np.eye(3)), "Rotation matrix should be identity"


@pytest.mark.xfail
def test_singular_axis() -> None:
    R = rot.direct_rot_mat(np.pi, np.array([0, 0, 0]))

    assert np.allclose(R, np.eye(3)), "Rotation matrix should be identity"


def test_roll_pitch_yaw_identity() -> None:
    R = rot.direct_rpy(0, 0, 0)
    assert np.allclose(R, np.eye(3)), "Rotation matrix should be identity"


def test_roll_pitch_yaw_single() -> None:
    R = rot.direct_rpy(np.pi, 0, 0)
    R_expected = rot.direct_rot_mat(np.pi, np.array([1, 0, 0]))

    assert np.allclose(R, R_expected), "Rotation matrix should be correct"


def test_roll_pitch_yaw_multiple() -> None:
    R = rot.direct_rpy(np.pi, np.pi, np.pi)
    R_expected = (
        rot.direct_rot_mat(np.pi, np.array([0, 0, 1]))
        @ rot.direct_rot_mat(np.pi, np.array([0, 1, 0]))
        @ rot.direct_rot_mat(np.pi, np.array([1, 0, 0]))
    )

    assert np.allclose(R, R_expected), "Rotation matrix should be correct"


def test_roll_pitch_yaw_separate() -> None:
    R_roll, R_pitch, R_yaw = rot.direct_rpy_separate(np.pi, np.pi, np.pi)

    R_expected_roll = rot.direct_rot_mat(np.pi, np.array([1, 0, 0]))
    R_expected_pitch = rot.direct_rot_mat(np.pi, np.array([0, 1, 0]))
    R_expected_yaw = rot.direct_rot_mat(np.pi, np.array([0, 0, 1]))

    assert np.allclose(R_roll, R_expected_roll), "Rotation matrix should be correct"
    assert np.allclose(R_pitch, R_expected_pitch), "Rotation matrix should be correct"
    assert np.allclose(R_yaw, R_expected_yaw), "Rotation matrix should be correct"


def test_rpy_separate_equal_to_direct() -> None:
    R1 = rot.direct_rot_mat(np.pi / 2, np.array([1, 0, 0]))

    R2_roll, R2_pitch, R2_yaw = rot.direct_rpy_separate(np.pi / 2, 0, 0)
    R2 = R2_yaw @ R2_pitch @ R2_roll

    assert np.allclose(
        R1, R2
    ), "Rotation matrices should be equal when computed through different methods"

    angle, axis = rot.inverse_rot_mat(R1)
    roll, pitch, yaw, ambiguity = rot.inverse_rpy(R1)

    assert axis is not None, "Axis should not be None"
    assert isinstance(
        angle, float
    ), f"Angle should be a single float, not {type(angle)}"
    assert np.isclose(angle, np.pi / 2), "Angle should be pi/2"
    assert np.allclose(axis, np.array([1, 0, 0])), "Axis should be [1, 0, 0]"

    assert np.isclose(roll, np.pi / 2), "Roll should be pi/2"
    assert np.isclose(pitch, 0), "Pitch should be 0"
    assert np.isclose(yaw, 0), "Yaw should be 0"

    assert ambiguity == None, "There should be no ambiguity"
    assert np.allclose(roll, angle), "Roll should be equal to angle"


def test_rpy_singularity() -> None:
    R = rot.direct_rpy(1, np.pi / 2, 1)
    roll, pitch, yaw, ambiguity = rot.inverse_rpy(R)

    assert ambiguity == RPY_RES.SUB, f"There should be a subtractive ambiguity"
    assert np.isclose(roll - yaw, 0), f"Roll - Yaw should be 0, not {roll - yaw}"
    assert np.isclose(pitch, np.pi / 2), f"Pitch should be pi/2, not {pitch}"

    R = rot.direct_rpy(1, -np.pi / 2, 1)

    roll, pitch, yaw, ambiguity = rot.inverse_rpy(R)

    assert ambiguity == RPY_RES.SUM, f"There should be an additive ambiguity"
    assert np.isclose(roll + yaw, 2), f"Roll + Yaw should be 2, not {roll + yaw}"
    assert np.isclose(pitch, -np.pi / 2), f"Pitch should be -pi/2, not {pitch}"


def test_roll_pitch_yaw_inverse() -> None:
    R = rot.direct_rpy(np.pi / 4, np.pi / 4, np.pi / 4)
    roll, pitch, yaw, ambiguity = rot.inverse_rpy(R)

    assert ambiguity == None, "There should be no ambiguity"

    print(roll, pitch, yaw)

    assert np.isclose(roll, np.pi / 4), "Roll should be pi/4"
    assert np.isclose(pitch, np.pi / 4), "Pitch should be pi/4"
    assert np.isclose(yaw, np.pi / 4), "Yaw should be pi/4"


def test_roll_pitch_yaw_inverse_ambiguity_sub() -> None:
    R = rot.direct_rpy(np.pi, np.pi / 2, np.pi)
    roll, pitch, yaw, ambiguity = rot.inverse_rpy(R)

    assert (
        ambiguity == RPY_RES.SUB
    ), "There should be an ambiguity with subtractive constraints"
    assert np.isclose(pitch, np.pi / 2), "Pitch should be pi/2"


def test_roll_pitch_yaw_inverse_ambiguity_sum() -> None:
    R = rot.direct_rpy(np.pi, -np.pi / 2, np.pi)
    roll, pitch, yaw, ambiguity = rot.inverse_rpy(R)

    assert (
        ambiguity == RPY_RES.SUM
    ), "There should be an ambiguity with additive constraints"
    assert np.isclose(pitch, -np.pi / 2), "Pitch should be -pi/2"
