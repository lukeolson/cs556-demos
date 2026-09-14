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
    k : int
        number of Arnoldi steps (size of the Krylov basis)
    v0 : array
        starting vector, default is a random vector
    reorthog : bool
        if True, perform a second Gram-Schmidt pass each step

    Returns
    -------
    V, H : n x m and m x m arrays, where m = k unless the iteration
        breaks down (invariant subspace found), in which case m < k
    """

    A = la.aslinearoperator(A)

    n = A.shape[0]
    k = min(k, n)  # at most n orthonormal vectors in R^n
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
        normAv = norm(w)
        for i in range(0, j + 1):
            H[i, j] = numpy.dot(w, V[:, i])
            w -= H[i, j] * V[:, i]

        if reorthog:
            # reorthogonlize
            d = V[:, :(j + 1)].T.dot(w)
            w -= V[:, :(j + 1)].dot(d)
            H[:(j + 1), j] += d

        if j < (k - 1):
            newh = norm(w)
            if newh <= 1e-10 * normAv:
                # breakdown: the Krylov space is invariant under A
                return (V[:, :(j + 1)], H[:(j + 1), :(j + 1)])
            H[j + 1, j] = newh
            V[:, j + 1] = (1 / newh) * w

    return (V, H)
