from collections import namedtuple
from collections.abc import Iterable
from scipy.stats import rv_discrete, rv_continuous, multivariate_normal, norm
from scipy.stats._distn_infrastructure import rv_sample
from numpy import interp
from os.path import dirname
import numpy as np
import pickle
import os


__all__ = ['cum_probs', 'disc_gen', 'discrete', 'empiric', 'gaussian_mixture']


# for pickling distributions
Gaussian_Mixture = namedtuple('Gaussian_Mixture', 'weights mu cov')


def cum_probs(values) -> np.ndarray:
    """Unique values & cumulated probabilities.

    example:
    val: [ 1,  2,  1] -> [ 1,  2]
    p:   [.2, .5, .3] -> [.5, .5]
    """
    x, px = np.reshape(values, [2, -1])
    y = np.unique(x)
    py = np.zeros(y.shape)
    for i in range(py.size):
        py[i] = px[np.where(x == y[i])].sum()
    return np.stack((y, py))


# https://github.com/scipy/scipy/blob/b225fd7a650c5beee18f26d98bd08d358f23a5d9/scipy/stats/_distn_infrastructure.py#L2675
class disc_gen(rv_sample):
    """Discrete distribution with reduced values and probabilities.

    example:
    val: [ 1,  2,  1] -> [ 1,  2]
    p:   [.2, .5, .3] -> [.5, .5]
    """
    def __new__(cls, *args, **kwds):
        return super(disc_gen, cls).__new__(cls)

    def __init__(self, **kwargs):
        kwargs['values'] = cum_probs(kwargs['values'])
        super().__init__(**kwargs)


def discrete(values) -> np.ndarray:
    """Discrete distribution with reduced values and probabilities.

    example:
    val: [ 1,  2,  1] -> [ 1,  2]
    p:   [.2, .5, .3] -> [.5, .5]
    """
    return rv_discrete(name='discrete', values=cum_probs(values))


class empiric(rv_sample):
    """Empiric distribution with linear interpolation."""

    def __new__(cls, *args, **kwds):
        return super(empiric, cls).__new__(cls)
    
    def __init__(self, reconstruct=False, **kwargs):
        if not reconstruct:
            xk, pk = np.unique(kwargs['values'], return_counts=True)
            kwargs['values'] = np.stack((xk, pk/pk.sum()))
        super().__init__(**kwargs)
        self.ck = np.cumsum(self.pk)
        self.ck[-1] = self.ck[-1] - min(0.000001, self.ck[0]/2)

    def pdf(self, x):
        return interp(x, self.xk, self.pk)
    
    def logpdf(self, x):
        return np.log(self.pdf(x))

    def cdf(self, x):
        return interp(x, self.xk, self.ck)
    
    def logcdf(self, x):
        return np.log(self.cdf(x))

    def sf(self, x):
        sf = 1-self.ck
        sf[-1] = 0
        return interp(x, self.xk, sf)#1.-self.ck)
    
    def logsf(self, x):
        return np.log(self.sf(x))
    
    def isf(self, x):
        sf = 1-self.ck
        sf[-1] = 0
        return interp(x, sf[::-1], self.xk[::-1])

    def ppf(self, x):
        return interp(x, self.ck, self.xk)

    def likelihood(self, x):
        return self.pdf(x).prod()

    def log_likelihood(self, x):
        return self.logpdf(x).sum()

    def uniformization(self, x, inv=False):
        return self.ppf(x) if inv else self.cdf(x)

    def gaussianization(self, x, inv=False):
        a, b = (self, norm) if inv else (norm, self)
        return a.ppf(b.cdf(x))


class gaussian_mixture(rv_continuous):
    """Gaussian mixture distribution."""

    def __new__(cls, *args, **kwds):
        return super(gaussian_mixture, cls).__new__(cls)

    def __init__(self, weights, mean=None, cov=None, **kwargs):
        """
        weights - mixing coefficients
        """
        self.weights, self.mean, self.cov = self.__prep(weights, mean, cov)
        super().__init__(**kwargs)

    def __prep(self, weights, mean, cov):
        """Handling data types, values and shapes.

        shapes:
        weights - w_dim
        mean - w_dim x dim
        cov - w_dim x dim x dim
        """
        if not isinstance(weights, Iterable):
            raise TypeError('weights must be iterable')
        if np.any(np.array(weights) < 0) or np.any(np.array(weights) > 1) or np.sum(weights) != 1:
            raise ValueError('weights must be in range (0, 1) and sum to 1')
        self.w_dim = len(weights)
        
        if cov is None or not isinstance(cov, np.ndarray):
            raise TypeError('cov must be numpy array/not None')
        if cov.ndim != 3 or cov.shape[1] != cov.shape[2] or cov.shape[0] != self.w_dim:
            raise IndexError('cov must be a list of %d square matrices' % (self.w_dim))
        self.dim = cov.shape[1]
        
        if mean is None:
            mean = np.zeros((self.w_dim, self.dim))
        if not isinstance(mean, np.ndarray):
            raise TypeError('mean must be numpy array')
        if mean.shape != (self.w_dim, self.dim):
            raise IndexError('mean must be of shape (%d, %d)' % (self.w_dim, self.dim))

        return weights, mean, cov

    def __prep_x(self, x):
        """Handling data types, values and shapes of x.

        shapes:
        x - (n, dim) or (dim)
        """
        if not isinstance(x, np.ndarray):
            raise TypeError('x must be numpy array')
        if x.ndim not in (1,2) or x.shape[-1] != self.dim:
            raise IndexError('x must be of shape (n, %d) or (%d,)' % (self.dim, self.dim))

    def __prep_k(self, x, k):
        """Handling data types, values and shapes of k.

        shapes:
        x - (n, dim) or (dim)
        k - (n) or scalar
        """
        self.__prep_x(x)
        if x.ndim == 2 and not isinstance(k, Iterable):
            raise TypeError('k must be of shape (%d,)' % (x.shape[0]))
        if x.ndim == 1 and (np.ndim(k) != 0 or k is None):
            raise TypeError('k must be scalar')
        if np.any(k < 0) or np.any(k >= self.w_dim):
            raise ValueError('k must be in range (0, %d)' % (self.w_dim))

    def pdf(self, x):
        """pdf from mix."""
        self.__prep_x(x)
        
        p = 0
        if x.ndim > 1:
            D = len(x)
            p = np.zeros(D)

        for k in range(self.w_dim):
            p += self.weights[k] * \
                multivariate_normal.pdf(x, mean=self.mean[k], cov=self.cov[k])
        return p

    def logpdf(self, x):
        """logpdf from mix."""
        return np.log(self.pdf(x))

    def pdf_w(self, x, k):
        """pdf from known distribution.

        k specifies which gaussian is used
        """
        self.__prep_k(x, k)
        if np.ndim(k) == 0:
            return multivariate_normal.pdf(x, mean=self.mean[k], cov=self.cov[k])

        D = len(k)
        p = np.zeros(D)

        for i, k_ in enumerate(k):
            p[i] = multivariate_normal.pdf(x[i], mean=self.mean[k_], cov=self.cov[k_])
        return p

    def logpdf_w(self, x, k):
        """logpdf from known distribution.

        k specifies which gaussian is used
        """
        self.__prep_k(x, k)
        if np.ndim(k) == 0:
            return multivariate_normal.logpdf(x, mean=self.mean[k], cov=self.cov[k])

        D = len(k)
        p = np.zeros(D)

        for i, k_ in enumerate(k):
            p[i] = multivariate_normal.logpdf(x[i], mean=self.mean[k_], cov=self.cov[k_])
        return p

    def rvs(self,
            size: int = 1,
            with_index: bool = False):
        """
        Draw random samples.
        ToDo: numpy.linalg.LinAlgError: SVD did not converge
        """
        # sample-wise weighted randomization
        r = discrete(values=(range(self.w_dim), self.weights)).rvs(size=size)
        
        # draw K samples and choose 1
        samples = np.zeros((self.w_dim, size, self.dim))
        for k in range(self.w_dim):
            samples[k] = multivariate_normal.rvs(
                mean=self.mean[k], cov=self.cov[k], size=size)

        # return with index of sampled distribution
        if with_index:
            return samples[r, range(size)], r
        # return without index
        return samples[r, range(size)]
    
    def logpdfrelevant(self, x,
                       ndim: int = 10):
        """Likelihood of most relevant features."""
        self.__prep_x(x)

        if ndim > self.dim:
            ndim = self.dim

        nmix = self.w_dim

        # split first nmix x ndim dims in nmix parts
        if np.ndim(x) == 1:
            # shape: nmix x ndims
            mix = x[:nmix*ndim].reshape(nmix, ndim)
        else:
            # shape: nmix x nsamples x ndims
            mix = x[:, :nmix*ndim].reshape(nmix, -1, ndim)
        # return index of mix with highest variance
        var = np.var(mix, axis=-1)
        k = np.argmax(var, axis=0)

        # range
        l, r = k*ndim, (k+1)*ndim

        # relevant features
        y = mix[k] if np.ndim(x) == 1 else mix[k, range(len(k))]

        def cov(k, l, r):
            """Retrieve relevant variances."""
            return np.diag(np.diag(self.cov[k])[l:r])

        mv = multivariate_normal
        if np.ndim(x) == 1:
            return mv.logpdf(y, cov=cov(k, l, r))
        return np.array([mv.logpdf(Y, cov=cov(K, L, R)) for Y, K, L, R in zip(y, k, l, r)])

    def save(self, file):
        os.makedirs(dirname(file), exist_ok=True)
        with open(file, 'wb') as f:
            pickle.dump(Gaussian_Mixture(*[self.weights, self.mean, self.cov]), f)

    def load(file):
        with open(file, 'rb') as f:
            return gaussian_mixture(*pickle.load(f))
