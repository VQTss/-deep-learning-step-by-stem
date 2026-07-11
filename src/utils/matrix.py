import numpy as np

def matrix_transpose(A):
    """
    Returns: ndarray, the transpose of A.
    """
    A = np.asarray(A,dtype=float)
    return A.T


def matix_trace(A):
    """
    Returns: float, the trace (sum of diagonal elements) of A.
    """
    A = np.asarray(A,dtype=float)
    return np.trace(A)


def matrix_vector_mutiply(A,x):
    """
    Returns: 1-D float64 array, the product A @ x.
    """
    A = np.asarray(A,dtype=float)
    x = np.asarray(x,dtype=float)
    return A @ x

def hadamard_product(A,B):
    """
    Returns: ndarray, the element-wise product A * B.
    """
    A = np.asarray(A,dtype=float)
    B = np.asarray(B,dtype=float)
    return np.multiply(A,B)

def matrix_multiply(A,B):
    """
    Returns: 2-D float64 array, the matrix product A @ B.
    """
    A = np.asarray(A,dtype=float)
    B = np.asarray(B,dtype=float)
    return A @ B

def matrix_determinant(A):
    """
    Returns: float, the determinant of square matrix A.
    """
    A = np.asarray(A,dtype=float)

    return float(np.linalg.det(A))


def matrix_rank(A):
    """
    Returns: int, the rank of matrix A.
    """
    A = np.asarray(A,dtype=float)
    return np.linalg.matrix_rank(A)