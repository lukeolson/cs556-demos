import numpy as np
from scipy.linalg import norm
import scipy.sparse.linalg as sla
from warnings import warn


def apply_givens(Q, v, k):
    '''
    Apply the first k Given's rotations in Q to v

    Parameters
    ----------
    Q : {list}
        list of consecutive 2x2 Given's rotations
    v : {array}
        vector to apply the rotations to
    k : {int}
        number of rotations to apply.

    Returns
    -------
    v is changed in place

    Notes
    -----
    This routine is specialized for GMRES.  It assumes that the first Given's
    rotation is for dofs 0 and 1, the second Given's rotation is for dofs 1, 2,
    and so on.
    '''

    for j in range(k):
        Qloc = Q[j]
        v[j:j+2] = np.dot(Qloc, v[j:j+2])


def gmres_mgs(A, b, x0=None, tol=1e-5, restrt=None, maxiter=None,
              residuals=None):
    '''
    Generalized Minimum Residual Method (GMRES)
        GMRES iteratively refines the initial guess to the system Ax = b
        Modified Gram-Schmidt version

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
    restrt : {None, int}
        - if int, restrt is max number of inner iterations
          and maxiter is the max number of outer iterations
        - if None, do not restart GMRES, and
          max number of inner iterations is maxiter
    maxiter : {None, int}
        - if restrt is None, maxiter is the max number of inner iterations
          and GMRES does not restart
        - if restrt is int, maxiter is the max number of outer iterations,
          and restrt is the max number of inner iterations
    residuals : list
        residuals has the residual norm history,
        including the initial residual, appended to it

    Returns
    -------
    (xNew, info)
    xNew : an updated guess to the solution of Ax = b
    info : halting status of gmres

            ==  =============================================
            0   successful exit
            >0  convergence to tolerance not achieved,
                return iteration count instead.  This value
                is precisely the order of the Krylov space.
            <0  numerical breakdown, or illegal input
            ==  =============================================

    Notes
    -----
        - The LinearOperator class is in scipy.sparse.linalg.interface.
          Use this class if you prefer to define A or M as a mat-vec routine
          as opposed to explicitly constructing the matrix.  A.psolve(..) is
          still supported as a legacy.
        - For robustness, modified Gram-Schmidt is used to orthogonalize
          the Krylov Space.
          Givens Rotations are used to provide the residual norm
          each iteration

    References
    ----------
    .. [1] Yousef Saad, "Iterative Methods for Sparse Linear Systems,
       Second Edition", SIAM, pp. 151-172, pp. 272-275, 2003
       http://www-users.cs.umn.edu/~saad/books.html

    .. [2] C. T. Kelley, http://www4.ncsu.edu/~ctk/matlab_roots.html
    '''

    A = sla.aslinearoperator(A)

    x = x0.copy()
    dimen = A.shape[0]

    # Should norm(r) be kept
    if residuals == []:
        keep_r = True
    else:
        keep_r = False

    # Set number of outer and inner iterations
    if restrt:
        if maxiter:
            max_outer = maxiter
        else:
            max_outer = 1
        if restrt > dimen:
            warn('Setting number of inner iterations (restrt) to maximimum '
                 'allowed, which is A.shape[0] ')
            restrt = dimen
        max_inner = restrt
    else:
        max_outer = 1
        
        if maxiter is None:
            maxiter = min(dimen, 40)
        
        if maxiter > dimen:
            warn('Setting number of inner iterations (maxiter) to maximimum '
                 'allowed, which is A.shape[0] ')
            maxiter = dimen

        max_inner = maxiter

    # Scale tol by normb
    normb = norm(b)
    if normb == 0:
        pass
    else:
        tol = tol*normb

    # Is this a one dimensional matrix?
    if dimen == 1:
        entry = np.ravel(A*np.array([1.0]))
        return (b/entry, 0)

    # Prep for method
    r = b - np.ravel(A*x)
    normr = norm(r)
    if keep_r:
        residuals.append(normr)

    # Is initial guess sufficient?
    if normr <= tol:
        return (x, 0)

    normr = norm(r)

    # Use separate variable to track iterations.  If convergence fails, we
    # cannot simply report niter = (outer-1)*max_outer + inner.  Numerical
    # error could cause the inner loop to halt while the actual ||r|| > tol.
    niter = 0

    # Begin GMRES
    for outer in range(max_outer):

        # Preallocate for Given's Rotations, Hessenberg matrix and Krylov Space
        # Space required is O(dimen*max_inner).  NOTE:  We are dealing with
        # row-major matrices, so we traverse in a row-major fashion, i.e., H
        # and V's transpose is what we store.
        Q = []                                 # Given's Rotations
        H = np.zeros((max_inner+1, max_inner+1), dtype=A.dtype)  # Upper Hessenberg matrix is
        V = np.zeros((max_inner+1, dimen), dtype=A.dtype)        # Krylov Space
        vs = []                                # vs are pointers to col of V

        # v = r/normr
        V[0, :] = (1.0/normr)*r
        vs.append(V[0, :])

        # This is the RHS vector for the problem in the Krylov Space
        g = np.zeros((dimen,))
        g[0] = normr

        for inner in range(max_inner):

            # New Search Direction
            v = V[inner+1, :]
            v[:] = np.ravel(A*vs[-1])
            vs.append(v)

            #  Modified Gram Schmidt
            for k in range(inner+1):
                vk = vs[k]
                alpha = np.dot(vk, v)
                H[inner, k] = alpha
                v += -alpha*vk

            normv = norm(v)
            H[inner, inner+1] = normv

            # Check for breakdown
            if H[inner, inner+1] != 0.0:
                v[:] = (1.0/H[inner, inner+1])*v

            # Apply previous Given's rotations to H
            if inner > 0:
                apply_givens(Q, H[inner, :], inner)

            # Calculate and apply next complex-valued Given's Rotation ==> Note
            # that if max_inner = dimen, then this is unnecessary for the last
            # inner iteration, when inner = dimen-1.
            if inner != dimen-1:
                if H[inner, inner+1] != 0:
                    h1 = H[inner, inner]
                    h2 = H[inner, inner+1]
                    h1_mag = abs(h1)
                    h2_mag = abs(h2)
                    if h1_mag < h2_mag:
                        mu = h1/h2
                        tau = np.conjugate(mu)/abs(mu)
                    else:
                        mu = h2/h1
                        tau = mu/abs(mu)

                    denom = np.sqrt(h1_mag**2 + h2_mag**2)
                    c = h1_mag/denom
                    s = h2_mag*tau/denom
                    Qblock = np.array([[c, np.conjugate(s)], [-s, c]])
                    Q.append(Qblock)

                    # Apply Given's Rotation to g,
                    #   the RHS for the linear system in the Krylov Subspace.
                    g[inner:inner+2] = np.dot(Qblock, g[inner:inner+2])

                    # Apply effect of Given's Rotation to H
                    H[inner, inner] = \
                        np.dot(Qblock[0, :], H[inner, inner:inner+2])
                    H[inner, inner+1] = 0.0

            # Don't update normr if last inner iteration, because
            # normr is calculated directly after this loop ends.
            if inner < max_inner-1:
                normr = abs(g[inner+1])
                if normr < tol:
                    break

                # Allow user access to residual
                if keep_r:
                    residuals.append(normr)

            niter += 1

        # end inner loop, back to outer loop

        # Find best update to x in Krylov Space, V.  Solve inner x inner system
        y = sla.spsolve(H[0:inner+1, 0:inner+1].T, g[0:inner+1])
        update = np.ravel(np.mat(V[:inner+1, :]).T*y.reshape(-1, 1))
        x = x + update
        r = b - np.ravel(A*x)

        normr = norm(r)

        # Allow user access to residual
        if keep_r:
            residuals.append(normr)

        # Has GMRES stagnated?
        indices = (x != 0)
        if indices.any():
            change = max(abs(update[indices] / x[indices]))
            if change < 1e-12:
                # No change, halt
                return (x, -1, H)

        # test for convergence
        if normr < tol:
            return (x, 0, H)

    # end outer loop

    return (x, niter, H)

if __name__ == '__main__':
    import scipy.sparse as sparse

    n = 10
    d = np.arange(1, n+1, dtype=float)
    d[0] = 1.0
    A = sparse.spdiags(d, [0], n, n).tocsr()

    b = np.zeros((n,))
    x0 = np.random.rand(n)

    res = []
    (x, flag) = gmres_mgs(A, b, x0, tol=1e-8, maxiter=n-2, residuals=res)
    res = np.array(res)

    import matplotlib.pyplot as plt
    plt.interactive(True)
    plt.figure()
    plt.semilogy(res, label='residuals')

    plt.figure()
    plt.semilogy(res[1:]/res[:-1], label='residual factors')
