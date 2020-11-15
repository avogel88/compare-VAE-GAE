import numpy as np
from numpy.testing import (assert_array_equal, assert_almost_equal,
                           assert_array_almost_equal, assert_equal, assert_)

from modules.numpy.cov import *
from numpy import diag, ones, trace
from numpy.linalg import eigvals


class TestCov:
    def psd(self, X):
        PSD = all(eigvals(X) >= 0)
        assert_equal(PSD, True)

    def test_diag(self):
        var = varroll(0,(3,7),(10,.1))
        n = len(var)
        V = [var, ones(n)]
        C = [var2cov(var),
             randcorr(n, n)]
        for v, c in zip(V, C):
            assert_array_almost_equal(diag(c), v)
            assert_almost_equal(trace(c), sum(v))

    def test_PSD(self):
        """Positive semi-definite?"""
        var = varroll(0,(3,7),(10,.1))
        self.psd(randcorr(10, 10))
        self.psd(var2cov(var))

    def test_symmetry(self):
        var = varroll(0,(3,7),(10,.1))
        P = randcorr(10, 10)
        assert_array_almost_equal(P, P.T)
        C = var2cov(var)
        assert_array_almost_equal(C, C.T)

    def test_mix(self):
        check = np.array([[1, 1, 0, 0, 0, 0],
                          [0, 0, 1, 1, 0, 0],
                          [0, 0, 0, 0, 1, 1]])
        var = varroll((0,1,2),(2,4),(1,0))
        cov1 = covmix(var)
        cov2 = var2cov(var, len(var))
        assert_equal(var, check)
        assert_equal(cov1.shape, (3, 6, 6))
        assert_equal(cov2.shape, (3, 6, 6))
