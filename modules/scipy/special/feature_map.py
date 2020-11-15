from itertools import combinations_with_replacement as combinations
import numpy as np
import scipy
from scipy.special import binom
from scipy.stats import norm
from modules.numpy.linalg import bool_solve, bool_rank, bool_inverse


__all__ = ['VeroneseMap', 'VeroneseMapWithIdentity']


# make numpy array from generator
def _np(x): return np.array(list(x))


def crange(start, stop=None, step=1):
    '''Range for chars.'''
    if(stop is None):
        start, stop = 'a', start
    if(isinstance(stop, int)):
        start, stop = 'a', chr(ord('a')+stop-1)
    for c in range(ord(start), ord(stop)+1, step):
        yield chr(c)


def check_stats(stat, ind):
    if isinstance(stat, int):
        return stat
    elif stat.ndim == 1:
        return stat[ind]
    return stat[:, ind]


class VeroneseMap():
    # variables:
    # map
    # literals
    def __init__(self, shape, shuffle=False):
        # shape: input & target dimensions
        m, n = self.shape = shape

        # calculate required degree of monomials
        d = 1
        while binom(m+d-1, d) < n:
            d = d + 1

        # combinations of input variates/indizees with degree d
        V = _np(combinations(range(m), d))

        if shuffle:
            np.random.shuffle(V)

        # reduce to target dimension
        self.map = V[:n]

    def __literals(self):
        '''map with literals'''
        L = _np(crange(self.shape[0]))
        return np.array([''.join(l) for l in L[self.map]])

    def __get__(self, instance, owner):
        return self.map

    def __set__(self, instance, value):
        self.map = value

    def __getitem__(self, key):
        return self.map[key]

    def __setitem__(self, key, value):
        self.map[key] = value

    def __repr__(self):
        return self.__str__()

    def __str__(self):
        return np.array2string(self.map)

    def __call__(self, x, inverse=False):
        # is x generator, list, tuple?
        if not isinstance(x, np.ndarray):
            x = _np(x) if isinstance(x, range) else np.array(x)
        if x.ndim > 1:
            return np.apply_along_axis(self.__apply, 1, x, inverse=inverse)
        return self.__apply(x, inverse)

    def __apply(self, x, inverse=False):
        '''Apply Veronese map on data.'''
        m = self.map

        if inverse:
            # solve linear system A*x = y
            # A - expanded sparse map
            A, ind = self.coefficient_matrix(full_rank=True)

            # log space: exponents as coefficients
            y = np.log(np.abs(x[ind]))

            # correct sign
            b = (np.sign(-x[ind])+1)/2
            s = bool_solve(A % 2, b)

            # solve linear system
            return np.exp(np.linalg.solve(A, y)) * (-2*s+1)

        return np.prod(x[m], axis=1)

    def coefficient_matrix(self, full_rank=False):
        '''Converts numeric coeffients into coefficient matrix.'''
        dim_x, dim_z = self.shape
        if full_rank:
            A = np.zeros((dim_x, dim_x))
            # assert full rank
            while(np.linalg.matrix_rank(A) != dim_x or bool_rank(A) != dim_x):
                # random indizees
                ind = np.random.permutation(self.map.shape[0])[:dim_x]
                A = np.zeros((dim_x, dim_x))
                for i, j in enumerate(ind):
                    # row-wise expansion
                    b, c = np.unique(self.map[j], return_counts=True)
                    A[i, b] = c
            return A, ind
        A = np.zeros((dim_z, dim_x))
        for j in range(dim_z):
            # row-wise expansion
            b, c = np.unique(self.map[j], return_counts=True)
            A[j, b] = c
        return A

    def l(self, x, dist, loc, scale):
        '''Calulate likelihood.'''
        lx = np.sum(dist.logpdf(x, loc, scale), axis=1)
        return lx

    literals = property(__literals)


class VeroneseMapWithIdentity():
    '''Feature map - degree * combinations of input variates.

    x ∈ ℝ^(samples ⨯ dim_x)  - start dstribution
    y ∈ ℝ^(dim_z   ⨯ degree) - feature map
    z ∈ ℝ^(samples ⨯ dim_z)  - target distribution
    '''

    def __init__(self, shape):
        '''Prepare feature map.

        [Identity + monomials]
        
        shape: input- & target distributions (dim_x, dim_z)
        degree: required degree of monomials, calculated from dim_x & dim_z
        
        y.shape = (dim_z-dim_x, degree)
        '''
        dim_x, dim_z = self.shape = shape
        degree = 1
        while binom(dim_x+degree-1, degree) < dim_z:
            degree = degree + 1

        # combinations of input variates/indizees
        y = _np(combinations(range(dim_x), degree))

        # reduce to target dimension
        self.y = y[:dim_z-dim_x]

    def __call__(self, x, inverse=False):
        '''Apply feature map.

        shape = (samples, dim_z)
        inverse:
        shape = (samples, dim_x)
        '''
        dim_x, dim_z = self.shape
        if inverse:
            # inverse feature map == identity
            return x[:, :dim_x]
        z = np.prod(x[:, self.y], axis=2)
        return np.concatenate((x, z), axis=1)

    def __getitem__(self, key):
        return self.y[key]
    
    def __repr__(self):
        return self.__str__()

    def __str__(self):
        return np.array2string(self.y)
    
    def __literals(self):
        '''map with literals'''
        dim_x, dim_z = self.shape
        L = _np(crange(self.shape[0]))
        Ly = np.array([''.join(l) for l in L[self.y]])
        return np.concatenate((L, Ly), axis=0)
    
    def l(self, x, dist=norm, loc=0, scale=1):
        '''Calulate likelihood.'''
        lx = np.sum(dist.logpdf(x, loc, scale),
                    axis=1)
        return lx

    def l_compare(lx, lz):
        '''Compare two sets of log-likelihoods.'''
        # https://stephens999.github.io/fiveMinuteStats/likelihood_ratio_simple_models.html
        return scipy.spatial.distance.euclidean(lx, lz)
        return scipy.spatial.distance.euclidean(np.exp(lx), np.exp(lz))

    literals = property(__literals)