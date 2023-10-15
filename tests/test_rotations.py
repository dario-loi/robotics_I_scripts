from pyrobots import rotations as rot

import numpy as np

import pytest

from rich import print

def test_direct():
    assert rot.direct_rot_mat(0, np.array([1, 0, 0])).all() == np.eye(3).all()
    assert (
        rot.direct_rot_mat(np.pi, np.array([1, 0, 0])).all()
        == np.array([[1, 0, 0], [0, -1, 0], [0, 0, -1]]).all()
    )
    assert (
        rot.direct_rot_mat(np.pi / 2, np.array([1, 0, 0])).all()
        == np.array([[1, 0, 0], [0, 0, -1], [0, 1, 0]]).all()
    )


def test_inverse():
    assert rot.inverse_rot_mat(np.eye(3))[0] == 0
    assert rot.inverse_rot_mat(np.eye(3))[1] == None

    # Accessing the first element of the tuple should return the angle, which should have two values
    # we check that the second value is pi
    assert (
        rot.inverse_rot_mat(np.array([[1, 0, 0], [0, -1, 0], [0, 0, -1]]))[0][1]
        == np.pi
    )


def test_compose_should_be_identity():
    # General case

    R = rot.direct_rot_mat(np.pi / 4, np.array([0, 1, 0]))
    theta, axis = rot.inverse_rot_mat(R)

    assert np.isclose(theta, np.pi / 4)
    assert axis.all() == np.array(np.array([0, 1, 0])).all()

    # Special case (theta = +- pi)

    R = rot.direct_rot_mat(-np.pi,np.array([1, 0, 0]))
    theta, axis = rot.inverse_rot_mat(R)

    assert np.isclose(theta, np.array([-np.pi, np.pi])).all()
    assert axis.all() == np.array([1, 0, 0]).all()

    # Special case (theta = 0)

    R = rot.direct_rot_mat(0, np.array([0, 0, 1]))
    theta, axis = rot.inverse_rot_mat(R)

    assert np.isclose(theta, 0)
    assert axis == None


if __name__ == "__main__":
    test_direct()
    test_inverse()
    test_compose_should_be_identity()

    print("All tests passed.")


def test_axis_scale_invariant():
    R_1 = rot.direct_rot_mat(np.sqrt(2) * np.pi, np.array([1, 0, 0.5]))
    R_2 = rot.direct_rot_mat(np.sqrt(2) * np.pi, np.array([3, 0, 1.5]))

    assert np.allclose(R_1, R_2)

def test_huge_axis():
    axis = np.array([1e15, 1e15, -1e15])
    
    R = rot.direct_rot_mat(2 * np.pi, axis)

    assert np.allclose(R, np.eye(3))
    
def test_huge_angle():
    # angles break down close to 1e8
    R = rot.direct_rot_mat(2 * np.pi * 1e7, np.array([0, 0, 1]))

    assert np.allclose(R, np.eye(3))
    
def test_huge_angle_and_axis():
    axis = np.array([1e15, 1e15, -1e15])
    
    R = rot.direct_rot_mat(2 * np.pi * 1e7, axis)

    assert np.allclose(R, np.eye(3))
    


@pytest.mark.xfail
def test_singular_axis():
    R = rot.direct_rot_mat(np.pi, np.array([0, 0, 0]))

    assert np.allclose(R, np.eye(3))
    
@pytest.mark.xfail
def test_huge_matrix():
    
    mat = np.array([
        [32, 44, 55],
        [12, 23, 34],
        [0, 0, 0]
    ])
    
    R = rot.direct_rot_mat(np.pi, mat)

    assert np.allclose(R, np.eye(3))
    