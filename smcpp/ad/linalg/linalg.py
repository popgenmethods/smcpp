"""
This sub-module allows for the usage of several linear algebra routines that
are otherwise unavailable for use for automatic differentiation. Not all 
numpy.linalg routines have an equivalent here, but we're working on it.

Decompositions
    a. chol (Cholesky)
    b. qr (QR Factorization
    c. lu (LU Decomposition)

Solving equations and inverting matrices
    a. solve (Solve a linear matrix equation, or system of linear scalar eqs)
    b. lstsq (Solve a linear least squares problem)
    c. inv (Compute the (multiplicative) inverse of a matrix)
    
(c) 2013 by Abraham Lee <tisimst@gmail.com>.
Please send feature requests, bug reports, or feedback to this address.

This software is released under a dual license.  (1) The BSD license.
(2) Any other license, as long as it is obtained from the original
author.

"""

from __future__ import division
import numpy as np

__all__ = ['np', 'chol', 'qr', 'lu', 'solve', 'lstsq', 'inv']

###############################################################################

def chol(A, side='lower'):
    """
    Cholesky decomposition of a symmetric, positive-definite matrix, defined
    by A = L*L.T = U.T*U, where L is a lower triangular matrix and U is an
    upper triangular matrix.
    
    Parameters
    ----------
    A : 2d-array
        The input matrix
    
    Optional
    --------
    side : str
        If 'lower' (default), then the lower triangular form of the
        decompostion is returned. If 'upper', then the upper triangular
        form of the decomposition is returned (the transpose of 'lower')
    
    Returns
    -------
    L : 2d-array
        The lower (or upper) triangular matrix that helps define the
        decomposition.
    
    Example
    -------
    Example 1::
    
        >>> A = [[25, 15, -5], 
        ...      [15, 18,  0], 
        ...      [-5,  0, 11]]
        ...
        >>> L = chol(A)
        >>> L
        array([[ 5.,  0.,  0.],
               [ 3.,  3.,  0.],
               [-1.,  1.,  3.]])
        >>> U = chol(A, 'upper')
        >>> U
        array([[ 5.,  3., -1.],
               [ 0.,  3.,  1.],
               [ 0.,  0.,  3.]])
        
    Example 2::
    
        >>> A = [[18, 22,  54,  42], 
        ...      [22, 70,  86,  62], 
        ...      [54, 86, 174, 134], 
        ...      [42, 62, 134, 106]]
        ...
        >>> L = chol(A)
        >>> L
        array([[  4.24264069,   0.        ,   0.        ,   0.        ],
               [  5.18544973,   6.5659052 ,   0.        ,   0.        ],
               [ 12.72792206,   3.0460385 ,   1.64974225,   0.        ],
               [  9.89949494,   1.62455386,   1.84971101,   1.39262125]])
           
    """
    A = np.array(A)
    assert A.shape[0]==A.shape[1], 'Input matrix must be square'
    
    L = [[0.0] * len(A) for _ in xrange(len(A))]
    for i in xrange(len(A)):
        for j in xrange(i+1):
            s = sum(L[i][k] * L[j][k] for k in xrange(j))
            L[i][j] = (A[i][i] - s)**0.5 if (i == j) else \
                      (1.0 / L[j][j] * (A[i][j] - s))
                          
    if side=='lower':
        return np.array(L)
    elif side=='upper':
        return np.array(L).T

###############################################################################

def qr(A):
    """
    QR Decomposition
    
    Parameters
    ----------
    A : 2d-array
        The input matrix (need not be square).
    
    Returns
    -------
    qm : 2d-array
        The orthogonal Q matrix from the decomposition
    rm : 2d-array
        The R matrix from the decomposition
    
    Example
    -------
    A square input matrix::
    
        >>> A = [[12, -51,   4], 
        ...      [ 6, 167, -68], 
        ...      [-4,  24, -41]]
        ...
        >>> q, r = qr(A)
        >>> q
        array([[-0.85714286,  0.39428571,  0.33142857],
               [-0.42857143, -0.90285714, -0.03428571],
               [ 0.28571429, -0.17142857,  0.94285714]])
        >>> r
        array([[ -1.40000000e+01,  -2.10000000e+01,   1.40000000e+01],
               [  5.97812398e-18,  -1.75000000e+02,   7.00000000e+01],
               [  4.47505281e-16,   0.00000000e+00,  -3.50000000e+01]])

    A non-square input matrix::
    
        >>> A = [[12, -51,   4], 
        ...      [ 6, 167, -68], 
        ...      [-4,  24, -41], 
        ...      [-1,   1,   0], 
        ...      [ 2,   0,   3]]
        ...
        >>> q, r = qr(A)
        >>> q
        array([[-0.84641474,  0.39129081, -0.34312406,  0.06613742, -0.09146206],
               [-0.42320737, -0.90408727,  0.02927016,  0.01737854, -0.04861045],
               [ 0.28213825, -0.17042055, -0.93285599, -0.02194202,  0.14371187],
               [ 0.07053456, -0.01404065,  0.00109937,  0.99740066,  0.00429488],
               [-0.14106912,  0.01665551,  0.10577161,  0.00585613,  0.98417487]])
        >>> r
        array([[ -1.41774469e+01,  -2.06666265e+01,   1.34015667e+01],
               [  3.31666807e-16,  -1.75042539e+02,   7.00803066e+01],
               [ -3.36067949e-16,   2.87087579e-15,   3.52015430e+01],
               [  9.46898347e-17,   5.05117109e-17,  -9.49761103e-17],
               [ -1.74918720e-16,  -3.80190411e-16,   8.88178420e-16]])
        >>> import numpy as np
        >>> np.all(np.dot(q, r) - A<1e-12)
        True
        
    """
    A = np.atleast_2d(A)
    m, n = A.shape
    qm = np.eye(m)
    rm = A.copy()
    for i in xrange(n-1 if m==n else n):
        x = getSubmatrix(rm, i, i, m, i)
        h = np.eye(m)
        h = setSubmatrix(h, i, i, householder(x))
        qm = np.dot(qm, h)
        rm = np.dot(h, rm)
    return qm, rm
    
###############################################################################

def lu(A):
    """
    Decomposes a nxn matrix A by PA=LU and returns L, U and P.
    
    Parameters
    ----------
    A : 2d-array
        The input matrix
   
    Returns
    -------
    L : 2d-array
        The lower-triangular matrix of the decomposition
    U : 2d-array
        The upper-triangular matrix of the decomposition
    P : 2d-array
        The pivoting matrix used in the decomposition
    
    Examples
    --------
    Example 1::
    
        >>> A = [[1, 3, 5],
        ...      [2, 4, 7],
        ...      [1, 1, 0]]
        ...
        >>> L, U, P = lu(A)
        >>> L
        array([[ 1. ,  0. ,  0. ],
               [ 0.5,  1. ,  0. ],
               [ 0.5, -1. ,  1. ]])
        >>> U
        array([[ 2. ,  4. ,  7. ],
               [ 0. ,  1. ,  1.5],
               [ 0. ,  0. , -2. ]])
        >>> P
        array([[ 0.,  1.,  0.],
               [ 1.,  0.,  0.],
               [ 0.,  0.,  1.]])
    
    Example 2::

        >>> A = [[11,  9, 24, 2], 
        ...      [ 1,  5,  2, 6], 
        ...      [ 3, 17, 18, 1], 
        ...      [ 2,  5,  7, 1]]
        ...
        >>> L, U, P = lu(A)
        >>> L
        array([[ 1.        ,  0.        ,  0.        ,  0.        ],
               [ 0.27272727,  1.        ,  0.        ,  0.        ],
               [ 0.09090909,  0.2875    ,  1.        ,  0.        ],
               [ 0.18181818,  0.23125   ,  0.00359712,  1.        ]])
        >>> U
        array([[ 11.        ,   9.        ,  24.        ,   2.        ],
               [  0.        ,  14.54545455,  11.45454545,   0.45454545],
               [  0.        ,   0.        ,  -3.475     ,   5.6875    ],
               [  0.        ,   0.        ,   0.        ,   0.51079137]])
        >>> P
        array([[ 1.,  0.,  0.,  0.],
               [ 0.,  0.,  1.,  0.],
               [ 0.,  1.,  0.,  0.],
               [ 0.,  0.,  0.,  1.]])
       
    """
    A = np.array(A)
    assert A.shape[0]==A.shape[1], 'Input matrix must be square'
    
    n = len(A)
    L = [[0.0]*n for i in xrange(n)]
    U = [[0.0]*n for i in xrange(n)]
    
    # Create the pivoting matrix for A
    P = [[float(i == j) for i in xrange(n)] for j in xrange(n)]
    for j in xrange(n):
        row = max(xrange(j, n), key=lambda i: A[i][j])
        if j != row:
            P[j], P[row] = P[row], P[j]
    
    A2 = np.dot(P, A)
    for j in xrange(n):
        L[j][j] = 1.0
        for i in xrange(j+1):
            s1 = sum(U[k][j] * L[i][k] for k in xrange(i))
            U[i][j] = A2[i][j] - s1
        for i in xrange(j, n):
            s2 = sum(U[k][j] * L[i][k] for k in xrange(j))
            L[i][j] = (A2[i][j] - s2) / U[j][j]
    return (np.array(L), np.array(U), np.array(P))
    
    
###############################################################################

def solve(A, b):
    """
    Solve a system of equations Ax = b by Gaussian elimination
    
    Parameters
    ----------
    A : 2d-array
        The LHS of the system of equations. The number of rows must not be less
        than the number of columns, but can be more.
    b : array-like
        The RHS of the system of equations (must have the same number of rows
        as ``A``. If more than one column is given, a solution is generated for
        each column
        
    Returns
    -------
    x : array-like
        The solution that satisifies the equality ``Ax==b``. If multiple ``b``
        are given, a solution column will be given satisfying each set.
    
    Example
    -------
    ::
    
        >>> A = [[1, 2, 1], [4, 6, 3], [9, 8, 2]]
        >>> b = [3, 2, 1]
        >>> solve(A, b)
        array([ -7.,  11., -12.])
        
    """
    try:
        A = np.matrix(A)
    except Exception:
        raise Exception('A must be a 2-dimensional array')
    assert A.shape[0]>=A.shape[1], 'A must not have less rows than columns'
    b = np.array(b)
    numout = 1 if b.ndim==1 else b.shape[1]
    assert A.shape[0]==b.shape[0], 'b must have the same number of rows as A'
    
    # Make the LHS array square if it isn't already and adjust RHS if needed
    if A.shape[0]>A.shape[1]:
        b = A.T*b
        A = A.T*A
        
    eqs = np.column_stack((A, b))
    n, m = eqs.shape
    
    # Forward substitution
    for k in range(n-1):
        # Pivot test
        p = k
        piv_el = abs(eqs[k, k])
        
        for i in range(k + 1, n):
            tmp = abs(eqs[i, k])
            if tmp>piv_el:
                piv_el = tmp
                p = i
        
        # Swap the kth and pth rows if a pivot element was found
        if p!=k:
            rtmp = np.array(eqs[p, :])
            eqs[p, :] = eqs[k, :]
            eqs[k, :] = rtmp
        
        for i in range(k + 1, n):
            f = 1.*eqs[i, k]/eqs[k, k]
            for j in range(k, m):
                eqs[i, j] -= f*eqs[k, j]
    
    # Backward substitution
    for k in range(1, n)[::-1]:
        for i in range(k)[::-1]:
            f = 1.*eqs[i, k]/eqs[k, k]
            for j in range(m)[::-1]:
                eqs[i, j] -= f*eqs[k, j]
    
    # Normalize
    for i in range(n):
        x = 1.*eqs[i, i]
        for j in range(m):
            eqs[i, j] = eqs[i, j]/x
    
    if numout==1:
        return np.array(eqs[:, -(m-n):]).ravel()
    else:
        return np.array(eqs[:, -(m-n):])

###############################################################################

def lstsq(A, b):
    """
    Solve Ax = b with the linear least squares method.
    
    Parameters
    ----------
    A : 2d-array
        The linear system matrix
    b : array
        The right-hand-side of the equation
    
    Returns
    -------
    x : array
        The solution to the system of equations
    
    Example
    -------
    (Taken from the NumPy lstsq example)::
    
        >>> x = np.array([0, 1, 2, 3])
        >>> y = np.array([-1, 0.2, 0.9, 2.1])
        >>> A = np.vstack([x, np.ones(len(x))]).T
        >>> A
        array([[ 0.,  1.],
               [ 1.,  1.],
               [ 2.,  1.],
               [ 3.,  1.]])
        >>> print lstsq(A, y)
        [ 1.   -0.95]
               
    """
    q, r = qr(A)
    n = r.shape[1]
    x = solveUpperTriangular(getSubmatrix(r, 0, 0, n - 1, n - 1), 
        np.dot(q.T, b))
    return x.ravel()

###############################################################################

def inv(A):
    """
    Calculate the multiplicative inverse of a matrix.
    
    Parameters
    ----------
    A : 2d-array
        The input matrix
        
    Returns
    -------
    Ainv : 2d-array
        The inverse of A
    
    Example
    -------
    ::

        >>> A = [[25, 15, -5], 
        ...      [15, 18,  0], 
        ...      [-5,  0, 11]]
        ...
        >>> Ainv = inv(A)
        >>> Ainv
        array([[ 0.09777778, -0.08148148,  0.04444444],
               [-0.08148148,  0.12345679, -0.03703704],
               [ 0.04444444, -0.03703704,  0.11111111]])
        >>> np.dot(Ainv, A)
        array([[  1.00000000e+00,   0.00000000e+00,   0.00000000e+00],
               [  2.77555756e-16,   1.00000000e+00,   0.00000000e+00],
               [  0.00000000e+00,   1.11022302e-16,   1.00000000e+00]])
       
    """
    shape = np.atleast_2d(A).shape
    assert shape[0]==shape[1], 'Matrix must be square'
    return solve(A, np.eye(shape[0]))
    
###############################################################################
#
#
#    All functions below are utility functions for some of the above
#    linear algebra functions. They should NOT be used directly.
#
#
###############################################################################

def ematrix(m, n, x, i, j):
    a = np.empty((m, n))
    a[:, :] = 0.0
    a[i, j] = x
    return a
    
def unitVector(n):
    return ematrix(n, 1, 1, 0, 0)

def signValue(r):
    if r<0:
        return -1
    elif r==0:
        return 0
    else:
        return 1

def householder(A):
    m = len(A)
    #u = A + np.sqrt(np.dot(A.T, A)[0, 0])*unitVector(m)*signValue(A[0, 0])
    u = A - np.sqrt(np.dot(A.T, A)[0, 0])*unitVector(m)#*signValue(A[0, 0])
    v = u#/u[0, 0]
    beta = 2/np.dot(v.T, v)
    return np.eye(m) - beta*(np.dot(v, v.T).T)

def getSubmatrix(obj, i1, j1, i2, j2):
    """
    i1, j1, i2, j2 are inclusive indices
    """
    return obj[i1:i2 + 1, j1:j2 + 1]

def setSubmatrix(obj, i1, j1, subobj):
    """
    i1, j1 are the top left corner indices where subobj will be inserted
    """
    m, n = np.atleast_2d(subobj).shape
    obj[i1:i1+m, j1:j1+n] = subobj
    return obj

def solveUpperTriangular(r, b):
    r = np.atleast_2d(r)
    n = r.shape[1]
    
    x = np.zeros((n, 1))
    for k in xrange(n - 1, -1, -1):
        idx = min(n - 1, k)
        x[k, 0] = (b[k] - np.dot(getSubmatrix(r, k, idx, k, n - 1),
            getSubmatrix(x, idx, 0, n - 1, 0)))/r[k, k]
    return x

def polyfit(x, y, n):
    """
    Fit a polynomial of order n to data arrays x and y
    
    Parameters
    ----------
    x : array
        A data array that defines the first dimension coordinates.
    y : array
        A data array that defines the second dimension coordinates.
    n : int
        The order of the polynomial (e.g., 1 for linear, 2 for quadratic, etc.)
    
    Returns
    -------
    b : array
        The polynomial coefficients, in ascending order.
    
    Example
    -------
    ::
    
        >>> x = range(11)
        >>> y = [1, 6, 17, 34, 57, 86, 121, 162, 209, 262, 321]
        >>> polyfit(x, y, 2)
        array([ 1.,  2.,  3.])
        
    """
    a = np.empty((len(x), n + 1))
    for i in xrange(a.shape[0]):
        for j in xrange(a.shape[1]):
            a[i, j] = 1 if j==0 else x[i]**j
    return lstsq(a, y)
