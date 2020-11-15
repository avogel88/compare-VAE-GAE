import numpy as np
from numpy.testing import (assert_array_equal, assert_almost_equal,
                           assert_array_almost_equal, assert_equal, assert_raises)


from modules.scipy.stats import *


class TestDiscrete:
    def test_cum_probs(self):
        xk = [0, 3, 2, 3, 4, 5, 6]
        pk = (0.1, 0.2, 0.3, 0.1, 0.1, 0.0, 0.2)
        xr = np.array([0, 2, 3, 4, 5, 6])
        pr = np.array([0.1, 0.3, 0.3, 0.1, 0.0, 0.2])
        assert_array_almost_equal(cum_probs((xk, pk)), np.stack((xr, pr)))

    def test_discrete_pmf(self):
        xk = [0, 3, 2, 3, 4, 5, 6]
        pk = (0.1, 0.2, 0.3, 0.1, 0.1, 0.0, 0.2)
        pr = np.array([0.1, 0.3, 0.3, 0.3, 0.1, 0., 0.2])
        disc = discrete((xk, pk))
        assert_array_almost_equal(disc.pmf(xk), pr)

    def test_discrete_ppf_cdf(self):
        xk = [0, 3, 2, 3, 4, 5, 6]
        pk = (0.1, 0.2, 0.3, 0.1, 0.1, 0.0, 0.2)
        xr = np.array([0, 3, 2, 3, 4, 4, 6])
        disc = discrete((xk, pk))
        assert_array_equal(disc.ppf(disc.cdf(xk)), xr)

    def test_disc_gen_pmf(self):
        xk = [0, 3, 2, 3, 4, 5, 6]
        pk = (0.1, 0.2, 0.3, 0.1, 0.1, 0.0, 0.2)
        pr = np.array([0.1, 0.3, 0.3, 0.3, 0.1, 0., 0.2])
        disc = disc_gen(values=(xk, pk))
        assert_array_almost_equal(disc.pmf(xk), pr)

    def test_disc_gen_ppf_cdf(self):
        xk = [0, 3, 2, 3, 4, 5, 6]
        pk = (0.1, 0.2, 0.3, 0.1, 0.1, 0.0, 0.2)
        xr = np.array([0, 3, 2, 3, 4, 4, 6])
        disc = disc_gen(values=(xk, pk))
        assert_array_equal(disc.ppf(disc.cdf(xk)), xr)


class TestEmpirical:
    def test_empirical_pdf(self):
        xk = [0, 3, 2, 3, 4, 5, 6, 7, 8, 9]
        pr = np.array([0.1, 0.2, 0.1, 0.2, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1])
        emp = empiric(values=xk)
        assert_array_equal(emp.pdf(xk), pr)

    def test_empirical_reconstruct(self):
        xk = [0, 1, 2, 3, 4]
        pk = np.array([.2, .3, .2, .1, .2])
        emp = empiric(values=(xk, pk), reconstruct=True)
        assert_array_equal(emp.pdf(xk), pk)

    def test_empirical_ppf_cdf(self):
        xk = np.array([0, 3, 2, 3, 4, 5, 6, 7, 8, 9])
        emp = empiric(values=xk)
        assert_array_equal(emp.ppf(emp.cdf(xk)), xk)

    def test_empirical_isf_sf(self):
        xk = np.array([0, 3, 2, 3, 4, 5, 6, 7, 8, 9])
        emp = empiric(values=xk)
        assert_array_equal(emp.isf(emp.sf(xk)), xk)

    def test_empirical_uniformization(self):
        xk = np.array([0, 3, 2, 3, 4, 5, 6, 7, 8, 9])
        emp = empiric(values=xk)
        yk = emp.uniformization(xk)
        xr = emp.uniformization(yk, inv=True)
        assert_array_equal(xr, xk)

    def test_empirical_gaussianization(self):
        xk = np.array([0, 3, 2, 3, 4, 5, 6, 7, 8, 9])
        emp = empiric(values=xk)
        yk = emp.gaussianization(xk)
        xr = emp.gaussianization(yk, inv=True)
        assert_array_almost_equal(xr, xk)


class TestMixofGaussians:
    def test_init(self):
        '''Test initialization - data types, values and shapes.'''
        # weights must be iterable
        with assert_raises(TypeError): gaussian_mixture(weights=None)
        with assert_raises(TypeError): gaussian_mixture(weights=1)
        with assert_raises(TypeError): gaussian_mixture(weights=[1])

        # weights must be in range (0, 1) and sum to 1
        with assert_raises(ValueError): gaussian_mixture(weights=[-1])
        with assert_raises(ValueError): gaussian_mixture(weights=[2])
        with assert_raises(ValueError): gaussian_mixture(weights=[.5])

        # cov must be numpy array/not None
        with assert_raises(TypeError): gaussian_mixture(weights=[1], cov=None)
        with assert_raises(TypeError): gaussian_mixture(weights=[1], cov=1)

        # cov must be a list of w_dim square matrices
        with assert_raises(IndexError): gaussian_mixture(weights=[1], cov=np.array([1]))

        # valid call; throws if None
        assert gaussian_mixture(weights=[1], mean=None, cov=np.array([[[1,1],[1,1]]]))

        # mean must be numpy array
        with assert_raises(TypeError): gaussian_mixture(weights=[1], mean=1, cov=np.array([[[1,1],[1,1]]]))
        with assert_raises(TypeError): gaussian_mixture(weights=[1], mean=[1], cov=np.array([[[1,1],[1,1]]]))

        # mean must be of shape (w_dim, dim)
        with assert_raises(IndexError): gaussian_mixture(weights=[1], mean=np.array([1]), cov=np.array([[[1,1],[1,1]]]))

        # valid call; throws if None
        assert gaussian_mixture(weights=[1], mean=np.array([[1,1]]), cov=np.array([[[1,0],[0,1]]]))

    def test_pdf(self):
        '''Test pdf of the mix.'''
        m = gaussian_mixture(weights=[1], mean=np.array([[1,1]]), cov=np.array([[[1,0],[0,1]]]))
        # x must be numpy array
        with assert_raises(TypeError): m.pdf(None)
        # x must be of shape (n, dim) or (dim,)
        with assert_raises(IndexError): m.pdf(np.array(1))
        # valid calls; throw if None
        assert m.pdf(np.array([1,1]))
        assert m.pdf(m.rvs())

    def test_logpdf(self):
        '''Test logpdf of the mix.'''
        m = gaussian_mixture(weights=[1], mean=np.array([[1,1]]), cov=np.array([[[1,0],[0,1]]]))
        # x must be numpy array
        with assert_raises(TypeError): m.logpdf(None)
        # x must be of shape (n, dim) or (dim,)
        with assert_raises(IndexError): m.logpdf(np.array(1))
        # valid calls; throw if None
        assert m.logpdf(np.array([1,1]))
        assert m.logpdf(m.rvs())

    def test_pdf_w(self):
        '''Test pdf from known distribution.'''
        m = gaussian_mixture(weights=[1], mean=np.array([[1,1]]), cov=np.array([[[1,0],[0,1]]]))
        # x must be numpy array
        with assert_raises(TypeError): m.pdf_w(None, None)
        # x must be of shape (n, dim) or (dim,)
        with assert_raises(IndexError): m.pdf_w(np.array(1), None)
        # k must be scalar
        with assert_raises(TypeError): m.pdf_w(np.array([1,1]), None)
        # k must be in range (0, w_dim)
        with assert_raises(ValueError): m.pdf_w(np.array([1,1]), 1)
        with assert_raises(ValueError): m.pdf_w(np.array([1,1]), -1)
        # valid call; throws if None
        assert m.pdf_w(np.array([1,1]), 0)

    def test_logpdf_w(self):
        '''Test logpdf from known distribution.'''
        m = gaussian_mixture(weights=[1], mean=np.array([[1,1]]), cov=np.array([[[1,0],[0,1]]]))
        # x must be numpy array
        with assert_raises(TypeError): m.logpdf_w(None, None)
        # x must be of shape (n, dim) or (dim,)
        with assert_raises(IndexError): m.logpdf_w(np.array(1), None)
        # k must be scalar
        with assert_raises(TypeError): m.logpdf_w(np.array([1,1]), None)
        # k must be in range (0, w_dim)
        with assert_raises(ValueError): m.logpdf_w(np.array([1,1]), 1)
        with assert_raises(ValueError): m.logpdf_w(np.array([1,1]), -1)
        # valid call; throws if None
        assert m.logpdf_w(np.array([1,1]), 0)

    def test_rvs(self):
        '''Test random variates.'''
        m = gaussian_mixture(weights=[1], mean=np.array([[1,1]]), cov=np.array([[[1,0],[0,1]]]))
        # sample 10
        assert_equal(m.rvs(10).shape, (10,2))
        # sample 10 and return index of used normal distribution
        r, i = m.rvs(10, with_index=True)
        assert_equal(r.shape, (10,2))
        assert_array_equal(i, np.zeros(10))

    def test_logpdfrelevant(self):
        """Test most relevant log-likelihoods."""
        m = gaussian_mixture(weights=[1], mean=np.array([[1,1]]), cov=np.array([[[1,0],[0,1]]]))
        # x must be numpy array
        with assert_raises(TypeError): m.logpdfrelevant(None)
        # x must be of shape (n, dim) or (dim,)
        with assert_raises(IndexError): m.logpdfrelevant(np.array(1))
        # valid calls; throw if None
        assert m.logpdfrelevant(np.array([1,1]))
        assert m.logpdfrelevant(m.rvs())
        