import numpy as np
from numpy.random import randint
from numpy.testing import (assert_array_equal, assert_almost_equal,
                           assert_array_almost_equal, assert_equal, assert_)


from modules.scipy.special import *


class TestRBIG:
    def test_rbig2(self):
        # prepare distribution
        Ns = 10000
        x = np.zeros((Ns, 2))
        x[:, 0] = np.abs(2 * np.random.randn(1, Ns))
        x[:, 1] = np.sin(x[:, 0]) + 0.25 * np.random.randn(1, Ns)

        # rbig
        G = rbig(x)
        G.fit(epochs=50)
        y = inv(G)(G(x))
        assert_array_almost_equal(x, y)

        G = rbig(x, epochs=50)
        y = inv(G)(G.x)
        assert_array_almost_equal(x, y)

        G = rbig(x, seeds=randint(2 * 32 - 1, size=50, dtype=np.int64))
        y = inv(G)(G.x)
        assert_array_almost_equal(x, y)

        G = rbig(x, epochs=30, seeds=randint(2 * 32 - 1, size=50, dtype=np.int64))
        y = inv(G)(G.x)
        assert_array_almost_equal(x, y)

        G = rbig(x, epochs=70, seeds=randint(2 * 32 - 1, size=50, dtype=np.int64))
        y = inv(G)(G.x)
        assert_array_almost_equal(x, y)

    def test_rbig(self):
        # prepare distribution
        Ns = 10000
        x = np.zeros((Ns, 2))
        x[:, 0] = np.abs(2*np.random.randn(1, Ns))
        x[:, 1] = np.sin(x[:, 0]) + 0.25*np.random.randn(1, Ns)

        # rbig
        G = RBIG_file(x, epochs=50, file='tmp/tmp.rbig')
        y = inv(G)(G.x)
        G.rem
        assert_array_almost_equal(x, y)

    def test_empirical_pdf(self):
        xk = [0, 3, 2, 3, 4, 5, 6, 7, 8, 9]
        pr = np.array([0.1, 0.2, 0.1, 0.2, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1])
        emp = empiric(xk)
        assert_array_equal(emp.pdf(xk), pr)

    def test_empirical_reconstruct(self):
        xk = [0, 1, 2, 3, 4]
        pk = np.array([.2, .3, .2, .1, .2])
        emp = empiric(xk, pk)
        assert_array_equal(emp.pdf(xk), pk)

    def test_empirical_ppf_cdf(self):
        xk = np.array([0, 3, 2, 3, 4, 5, 6, 7, 8, 9])
        emp = empiric(xk)
        assert_array_equal(emp.ppf(emp.cdf(xk)), xk)

    def test_empirical_isf_sf(self):
        xk = np.array([0, 3, 2, 3, 4, 5, 6, 7, 8, 9])
        emp = empiric(xk)
        assert_array_equal(emp.isf(emp.sf(xk)), xk)

    def test_empirical_gaussianization(self):
        xk = np.array([0, 3, 2, 3, 4, 5, 6, 7, 8, 9])
        gss = gaussianization(xk)
        xr = inv(gss)(gss(xk))
        assert_array_almost_equal(xr, xk)

    def test_empirical_domains(self):
        probs = [[-1, 2],   # domain
                 [0, .1],   # sum to 1
                 [.1, 0],   # domain of cdf
                 [.3, .3],
                 [0, 0],    # zero sum
                 [-2, -1]]  # order
        for prob in probs:
            emp = empiric([1, 2], prob)
            # assert sum of pk == 1
            # also tests normalize
            assert_equal(sum(emp.pk), 1.)
            # assert domains
            # also tests normalize & normalize_cdf
            assert_equal(all(emp.pk >= 0) and all(emp.pk <= 1), True)
            assert_equal(all(emp.ck > 0) and all(emp.ck < 1), True)
            assert_equal(all(emp.sk > 0) and all(emp.sk < 1), True)
