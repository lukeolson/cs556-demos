import numpy
import scipy.sparse.linalg as la

from numpy.linalg import norm


def arnoldi(A, k, v0=None, reorthog=False):
    """Arnoldi iteration
    Compute H, V such that H = V.T A V

    Parameters
    ----------
    A : sparse matrix
        n x n, linear system to orthogonalize
    x0 : {array, matrix}
        initial guess, default is a random vector
    """

    A = la.aslinearoperator(A)

    n = A.shape[0]
    if v0 is None:
        v = numpy.random.rand(n,)
    else:
        v = v0

    V = numpy.zeros((n, k))
    H = numpy.zeros((k, k))
    v = (1.0 / norm(v)) * v
    V[:, 0] = v

    for j in range(0, k):
        w = A * V[:, j]
        for i in range(0, j + 1):
            H[i, j] = numpy.dot(w, V[:, i])
            w -= H[i, j] * V[:, i]

        if reorthog:
            # reorthogonlize
            d = V[:, :(j + 1)].T.dot(w)
            w -= V[:, :(j + 1)].dot(d)
            H[:(j + 1), j] += d

        newh = norm(w)
        if j < (k - 1):
            H[j + 1, j] = newh
            V[:, j + 1] = (1 / newh) * w

    return (V, H)
