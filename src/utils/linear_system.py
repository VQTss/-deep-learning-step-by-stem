import numpy as np

def vector_projection(u,v):
    """
    Returns: float64 array, the projection of u onto v.
    """

    u = np.asarray(u,dtype=float)
    v = np.asarray(v,dtype=float)
    scalar = np.dot(u,v) / np.dot(v,v)
    return scalar *  v

def gram_schmidt(vectors):
    """
    Returns: float64 array of shape (k, n), orthonormal basis spanning the input space.
    """
    A = np.array(vectors,dtype=float)
    Q = np.zeros_like(A)
    for i in range(len(A)):
        v = A[i].copy()
        for j in range(i):
            v -= np.dot(v,Q[j]) * Q[j]
        Q[i] = v / np.linalg.norm(v)
    return Q

def lu_decomposition(A):
    """Returns: tuple (L, U) where A = L @ U."""
    A = np.array(A, dtype=float)
    n = len(A)
    L = np.eye(n)
    U = np.zeros((n, n))
    for k in range(n):
        for j in range(k, n):
            U[k, j] = A[k, j] - sum(L[k, m] * U[m, j] for m in range(k))
        for i in range(k + 1, n):
            L[i, k] = (A[i, k] - sum(L[i, m] * U[m, k] for m in range(k))) / U[k, k]
    return L, U