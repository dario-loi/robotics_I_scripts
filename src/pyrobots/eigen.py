from typing import Tuple

import numpy as np


def eigenvalues(matrix: np.ndarray) -> np.ndarray:
    """
    Given a matrix, returns the eigenvalues of that matrix

    Parameters:
    matrix (numpy.ndarray): a square matrix
    """

    assert matrix.shape[0] == matrix.shape[1], "Matrix is not square"

    eigvals = np.linalg.eigvals(matrix)
    return eigvals


def eigenvectors(matrix: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Given a matrix, returns only the real eigenvectors of that matrix

    Parameters:
    matrix (numpy.ndarray): a square matrix

    Returns:
    eigvals (numpy.ndarray): eigenvalues of the matrix
    eigvecs (numpy.ndarray): eigenvectors of the matrix
    """
    # check if matrix is square

    assert matrix.shape[0] == matrix.shape[1], "Matrix is not square"

    if matrix.shape[0] != matrix.shape[1]:
        print("Matrix is not square")
        return

    # get eigenvalues
    eigvals, eigvecs = np.linalg.eig(matrix)

    for i in range(eigvecs.shape[1]):
        eigvecs[:, i] = eigvecs[:, i] / np.linalg.norm(eigvecs[:, i])

    # sort eigenvectors with real eigenvalues
    reals_mask = np.isreal(eigvals)

    eigvecs = eigvecs[:, reals_mask]
    eigvals = eigvals[reals_mask]

    # turn underlying data type to float
    import warnings

    # I swear I know what I'm doing
    warnings.filterwarnings(action="ignore", category=np.ComplexWarning)

    eigvecs = eigvecs.astype(np.float32)
    eigvals = eigvals.astype(np.float32)

    warnings.resetwarnings()

    return eigvals, eigvecs


# matrix = np.array([[0,1/np.sqrt(2),-1/np.sqrt(2)],[-1/np.sqrt(2),-0.5,-0.5],[-1/np.sqrt(2),0.5,0.5]])
# print(eigenvectors(matrix))
