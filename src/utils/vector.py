from unittest import result
import numpy as np

def dot_product(x,y):
    """
    Compute the dot product of two 1D arrays x and y.
    Must return a float.
    """
    return np.dot(x,y)

def eculidean_distance(x,y):
    """
    Returns: float, the Euclidean distance between x and y.
    """
    x = np.asarray(x,dtype=float)
    y = np.asarray(y,dtype=float)
    return float(np.linalg.norm(x-y))

def consine_similarity(a,b):
    """
    Returns: float in [-1, 1], cosine similarity between a and b.
    """
    a = np.asarray(a, dtype=float)
    b = np.asarray(b,dtype=float)
    dotProduct = dot_product(a,b)
    normA = np.linalg.norm(a)
    normB = np.linalg.norm(b)
    return float(dotProduct / (normA * normB))

def vector_norms(v):
    """
    Returns: float64 array of shape (3,) containing [L1, L2, L-inf] norms.
    """
    v = np.asarray(v,dtype=float)
    l1 = np.linalg.norm(v,ord=1)
    l2 = np.linalg.norm(v,ord=2)
    linf = np.linalg.norm(v,ord=np.inf)
    return np.array([l1,l2,linf])

def outer_product(u,v):
    """
    Returns: float64 matrix of shape (m, n), the outer product u v^T.
    """
    u = np.asarray(u,dtype=float)
    v = np.asarray(v,dtype=float)
    return np.outer(u,v)


def projection(u, v):
    """
    Returns: 1D array, the vector projection of u onto v.
    """
    u = np.asarray(u, dtype=float)
    v = np.asarray(v, dtype=float)
    return (dot_product(u, v) / dot_product(v, v)) * v


def angle_between(u, v):
    """
    Returns: float, the angle in radians between vectors u and v.
    """
    u = np.asarray(u, dtype=float)
    v = np.asarray(v, dtype=float)
    cos_theta = consine_similarity(u, v)
    return float(np.arccos(np.clip(cos_theta, -1.0, 1.0)))


a =  [-3, 4, -2]
b = [3,4]

result = vector_norms(a)
print(result)