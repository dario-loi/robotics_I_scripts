import numpy as np
import pytest
from rich import print

from pyrobots import rotations as rot


def test_direct() -> None:
    assert rot.direct_rot_mat(0, np.array([1, 0, 0])).all() == np.eye(3).all(), "Rotation matrix for 0 radians about x-axis should be identity"
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
    
    assert (isinstance(theta, tuple) and len(theta) == 2)
    t1, t2 = theta
    
    assert axis is not None, "axis should not be None"
    assert np.isclose(t1, -np.pi), "t1 should be -pi"
    assert np.isclose(t2, np.pi), "t2 should be pi"
    assert axis.all() == np.array([1, 0, 0]).all(), "axis should be [1, 0, 0]"
    assert (t1 == -t2), "t1 and t2 should be negatives of each other"

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

    assert np.isclose(theta, np.array([-np.pi, np.pi])).all(), "theta should be -pi or pi"
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
        rot.direct_rot_mat(np.pi, np.array([1, 0, 0]))
        @ rot.direct_rot_mat(np.pi, np.array([0, 1, 0]))
        @ rot.direct_rot_mat(np.pi, np.array([0, 0, 1]))
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
