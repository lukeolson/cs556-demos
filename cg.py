import numpy as np
from scipy.sparse.linalg.isolve.utils import make_system
from pyamg.util.linalg import norm


def cg(A, b, x0=None, tol=1e-5, maxiter=None,
       callback=None, residuals=None, errs=None):
    '''Conjugate Gradient algorithm

    Solves the linear system Ax = b. Left preconditioning is supported.

    Parameters
    ----------
    A : {array, matrix, sparse matrix, LinearOperator}
        n x n, linear system to solve
    b : {array, matrix}
        right hand side, shape is (n,) or (n,1)
    x0 : {array, matrix}
        initial guess, default is a vector of zeros
    tol : float
        relative convergence tolerance, i.e. tol is scaled by ||b||
    maxiter : int
        maximum number of allowed iterations
    callback : function
        User-supplied funtion is called after each iteration as
        callback(xk), where xk is the current solution vector
    residuals : list
        residuals has the residual norm history,
        including the initial residual, appended to it
    errs : list
        list of x.T A X

    '''
    A, M, x, b, postprocess = make_system(A, None, x0, b)

    # Determine maxiter
    if maxiter is None:
        maxiter = int(1.3*len(b)) + 2
    elif maxiter < 1:
        raise ValueError('Number of iterations must be positive')

    # Scale tol by normb
    normb = norm(b)
    if normb != 0:
        tol = tol*normb

    # setup method
    r = b - A*x
    p = r.copy()
    rr = np.inner(np.conjugate(r), r)

    normr = norm(r)

    if residuals is not None:
        residuals[:] = [normr]  # initial residual
    if errs is not None:
        errs[:] = [np.sqrt(np.dot(A*x, x))]

    if normr < tol:
        return (postprocess(x), 0)

    iter = 0

    while True:
        Ap = A*p

        rr_old = rr

        alpha = rr/np.inner(np.conjugate(Ap), p)
        x += alpha * p
        r -= alpha * Ap
        rr = np.inner(np.conjugate(r), r)
        beta = rr / rr_old
        p *= beta
        p += r

        iter += 1

        normr = norm(r)

        if residuals is not None:
            residuals.append(normr)

        if errs is not None:
            errs.append(np.sqrt(np.dot(A*x, x)))

        if callback is not None:
            callback(x)

        if normr < tol:
            return (postprocess(x), 0)

        if iter == maxiter:
            return (postprocess(x), iter)

if __name__ == '__main__':
    from pyamg.gallery import poisson
    A = poisson((10, 10))
    b = np.ones((A.shape[0],))
    res = []
    (x, flag) = cg(A, b, maxiter=8, tol=1e-8, residuals=res)
    print(np.linalg.norm(b - A*x))
    print(res)
