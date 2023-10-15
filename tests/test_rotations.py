from pyrobots import rotations as rot

import numpy as np


def test_direct():
    assert rot.direct_rot_mat(0, [1, 0, 0]).all() == np.eye(3).all()
    assert (
        rot.direct_rot_mat(np.pi, [1, 0, 0]).all()
        == np.array([[1, 0, 0], [0, -1, 0], [0, 0, -1]]).all()
    )
    assert (
        rot.direct_rot_mat(np.pi / 2, [1, 0, 0]).all()
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

    R = rot.direct_rot_mat(np.pi / 4, [0, 1, 0])
    theta, axis = rot.inverse_rot_mat(R)

    assert np.isclose(theta, np.pi / 4)
    assert axis.all() == np.array([0, 1, 0]).all()

    # Special case (theta = +- pi)

    R = rot.direct_rot_mat(-np.pi, [1, 0, 0])
    theta, axis = rot.inverse_rot_mat(R)

    assert np.isclose(theta, np.array([-np.pi, np.pi])).all()
    assert axis.all() == np.array([1, 0, 0]).all()

    # Special case (theta = 0)

    R = rot.direct_rot_mat(0, [0, 0, 1])
    theta, axis = rot.inverse_rot_mat(R)

    assert np.isclose(theta, 0)
    assert axis == None


if __name__ == "__main__":
    test_direct()
    test_inverse()
    test_compose_should_be_identity()

    print("All tests passed.")


def test_axis_scale_invariant():
    R_1 = rot.direct_rot_mat(np.sqrt(2) * np.pi, [1, 0, 0.5])
    R_2 = rot.direct_rot_mat(np.sqrt(2) * np.pi, [3, 0, 1.5])

    assert np.isclose(R_1, R_2).all()
