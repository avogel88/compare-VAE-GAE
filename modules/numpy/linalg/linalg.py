import numpy as np


__all__ = ['matrix_append', 'isdiag', 'gauss_jordan', 'bool_solve', 'bool_rank', 'bool_inverse']


def matrix_append(A, b):
    '''Extend matrix A with vector b.'''
    if b.ndim == 1:
        return np.concatenate((A, b[np.newaxis].T), axis=1)
    return np.concatenate((A, b), axis=1)


def isdiag(A):
    '''Is matrix A a diagonal matrix?'''
    return not np.any(A - np.diag(np.diag(A)))


def gauss_jordan(A, pivot=None):
    '''Gauss Jordan algorithm for solving linear systems - boolean version.'''
    A = A%2
    rng = min(A.shape) if pivot in (None, -1) else pivot
    nz = A.shape[0] # nr not-zero-rows
    for i in range(rng):
        # sort by number of coefficients
        sl = slice(None, pivot)
        ind = np.argsort(np.sum(A[i:nz, sl], axis=1)) + i
        A[i:nz] = A[ind]
        # push zero-rows down
        zrows = np.where(~A[:,sl].any(axis=1))[0]
        nzrows = np.where(A[:,sl].any(axis=1))[0]
        nz -= zrows.size
        A = A[np.concatenate((nzrows,zrows))]
        # sort by order of columns
        ind = np.argsort(np.argmax(A[i:], axis=1)) + i
        A[i:] = A[ind]
        # subtract row (xor)
        ind = np.argwhere(A[i+1:, i]) + i + 1
        A[ind] = np.logical_xor(A[ind], A[i]) * 1
    # remove upper triangular
    for i in reversed(range(1, rng)):
        ind = np.argwhere(A[:i, i])
        A[ind] = np.logical_xor(A[ind], A[i]) * 1
    return A


def bool_solve(A, b=None):
    '''Linear system solver for boolean values.'''
    # prepare extended coefficient matrix
    if b is None:
        Ab = A
    else:
        Ab = matrix_append(A, b)
    Ab = gauss_jordan(Ab, pivot=-1)
    return Ab[:A.shape[1], -1]


def bool_inverse(A):
    m = min(A.shape)
    n = max(A.shape)
    D = np.diag(np.ones(n))
    AD = np.concatenate((A, D), axis=1)
    return gauss_jordan(AD, pivot=m)[:m, m:]


def bool_rank(A):
    '''Matrix rank for boolean matrices.'''
    G = gauss_jordan(A)
    loc = np.argwhere(G)
    rows = np.unique(loc[:, 0])
    cols = np.unique(loc[:, 1])
    return min(rows.size, cols.size)