import numpy as np


def fom(A, b, x0=None, maxiter=None, residuals=None, errs=None):
    """Full orthogonalization method

    Parameters
    ----------
    A : {array, matrix, sparse matrix, LinearOperator}
        n x n, linear system to solve
    b : {array, matrix}
        right hand side, shape is (n,) or (n,1)
    x0 : {array, matrix}
        initial guess, default is a vector of zeros
    maxiter : int
        maximum number of allowed iterations
    residuals : list
        residuals has the residual norm history,
        including the initial residual, appended to it
    errs : list of errors returned through (Ax,x), so test the errors on Ax=0
    """
    n = len(b)
    if maxiter is None:
        maxiter = n
    if x0 is None:
        x = np.ones((n,))
    else:
        x = x0.copy()

    r = b - A * x
    beta = np.linalg.norm(r)

    if residuals is not None:
        residuals[:] = [beta]  # initial residual
    if errs is not None:
        errs[:] = [np.sqrt(np.dot(A * x, x))]

    V = np.zeros((n, maxiter))
    H = np.zeros((maxiter, maxiter))
    V[:, 0] = (1 / beta) * r

    for j in range(0, maxiter):
        w = A * V[:, j]
        for i in range(0, j + 1):
            H[i, j] = np.dot(w, V[:, i])
            w += -H[i, j] * V[:, i]
        newh = np.linalg.norm(w)
        if abs(newh) < 1e-13:
            break
        elif j < (maxiter - 1):
            H[j + 1, j] = newh
            V[:, j + 1] = (1 / newh) * w

        # do some work to check the residual
        #
        if residuals is not None:
            e1 = np.zeros((j + 1, 1))
            e1[0] = beta
            y = np.linalg.solve(H[0:j + 1, 0:j + 1], e1)
            z = np.dot(V[:, 0:j + 1], y)
            x = x0 + z.ravel()
            residuals.append(abs(newh * y[j][0]))
        if errs is not None:
            e1 = np.zeros((j + 1, 1))
            e1[0] = beta
            y = np.linalg.solve(H[0:j + 1, 0:j + 1], e1)
            z = np.dot(V[:, 0:j + 1], y)
            x = x0 + z.ravel()
            errs.append(np.sqrt(np.dot(A * x, x)))

    e1 = np.zeros((j + 1, 1))
    e1[0] = beta
    y = np.linalg.solve(H[0:j + 1, 0:j + 1], e1)
    z = np.dot(V[:, 0:j + 1], y)
    x = x0 + z.ravel()

    return (x, newh)

if __name__ == "__main__":
    from pyamg.gallery import poisson
    import onedprojection
    import matplotlib.pyplot as plt
    import time

    its = 40

    n = 20
    A = poisson((n, n)).tocsr()

    b = np.zeros((n * n,))
    x0 = np.random.rand(n * n)
    res = []
    err = []
    res2 = []
    err2 = []

    t1 = time.time()
    x = onedprojection.onedprojection(
        A, b, x0=x0, tol=1e-8, maxiter=its, residuals=res,
        method='SD', errs=err)
    t2 = time.time()

    t1 = time.time()
    x2 = fom(A, b, x0=x0, maxiter=its, residuals=res2, errs=err2)
    t2 = time.time()

    plt.figure()
    plt.semilogy(res, label='SD residuals')
    plt.hold(True)
    plt.semilogy(err, label='SD errors')
    plt.semilogy(res2, label='FOM residuals')
    plt.semilogy(err2, label='FOM errors')
    plt.xlabel("iteration")
    plt.ylabel("||.||")
    plt.legend()
    plt.show()
