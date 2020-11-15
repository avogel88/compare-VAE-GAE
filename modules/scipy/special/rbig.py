from collections import namedtuple
import matplotlib.pyplot as plt
from numpy import *
from numpy.linalg import qr
from numpy.random import randint, randn, seed as randseed
from os import makedirs, remove
from os.path import isfile, basename, dirname
from pandas import Series
import pickle
from scipy.stats import norm, kstest
from typing import Any, Callable, Generator, List, Optional, Sequence, Union


__all__ = ['RBIG_file', 'RBIG', 'inv', 'empiric', 'gaussianization']


Empiric = namedtuple('Empiric', ['xk', 'pk'])


def inv(obj: Any) -> Callable:
    """Switches inv & call methods."""
    return obj.__inv__
    obj.__class__.__call__, obj.__inv__ = obj.__inv__, obj.__call__
    return obj


def normalize(x):
    """Domain: [0, 1]"""
    x = array(x)
    # assert minimal value >= 0
    x[x<0] = 0
    # assert sum = 1
    if sum(x) == 0:
        x = ones_like(x)
    return x/sum(x)


def normalize_cdf(x, tol=1e-8):
    """This prevents norm.ppf({0, 1}) = +-inf by shifting the values of CDF into (0, 1)."""
    x[x==0.] = tol
    x[x==1.] = 1. - tol
    return x


class empiric:
    """Empiric distribution with connected support.

    domains:
    xk: real numbers
    pk: [0,1]
    ck, sk: (0,1)
    to prevent norm.ppf from returning {-inf, +inf},
    which in turn would break the rotations of RBIG
    - anynumber * inf = nan
    - matrix multiplications propagate nans
    ToDo: Warnings:
        RuntimeWarning: invalid value encountered in greater_equal
        cond2 = (x >= np.asarray(_b)) & cond0
        RuntimeWarning: invalid value encountered in less_equal
        cond2 = cond0 & (x <= _a)
        RuntimeWarning: invalid value encountered in greater
        cond1 = (0 < q) & (q < 1)
        RuntimeWarning: invalid value encountered in less
        cond1 = (0 < q) & (q < 1)
    """

    def __init__(self, x, p=None):
        if ndim(x) == 2:
            x, p = x[0], x[1]
        y, py = unique(x, return_counts=True)
        if p is not None:
            py = p
            if len(x) != len(y):
                py = zeros(y.shape)
                for i in range(y.size):
                    py[i] = p[where(x == y[i])].sum()
        # normalize probabilities
        p = normalize(py)
        self.xk = y
        self.pk = p
        self.ck = cumsum(p)
        # correct domains of ck & sk
        # many values may be 0 or 1!
        self.ck = normalize_cdf(self.ck, 1e-8)
        self.sk = 1. - self.ck

    def pdf(self, x): return interp(x, self.xk, self.pk)
    def cdf(self, x): return interp(x, self.xk, self.ck)
    def ppf(self, x): return interp(x, self.ck, self.xk)
    def  sf(self, x): return interp(x, self.xk, self.sk)
    # reversed ordering - sf is decreasing
    def isf(self, x): return interp(x, self.sk[::-1], self.xk[::-1])
    params = property(lambda self: Empiric(self.xk, self.pk))


class gaussianization(empiric):
    def __init__(self, x, p=None): super().__init__(x, p)
    def __call__(self, x): return norm.ppf(self.cdf(x))
    def __inv__(self, x): return self.ppf(norm.cdf(x))


class Gaussianize_Matrix:
    """Marginally gaussianize each feature of design matrix."""

    def __call__(self, x, dists):
        for i in range(len(x)):
            x[i] = gaussianization(dists[i])(x[i])
        return x

    def __inv__(self, x, dists):
        for i in range(len(x)):
            x[i] = inv(gaussianization(dists[i]))(x[i])
        return x
gaussianize_matrix = Gaussianize_Matrix()


def dists_of_matrix(x: ndarray) -> List[ndarray]:
    """Feature wise empirical distributions."""
    dists = []
    for feature in x:
        dists += [empiric(feature).params]
    # dists = self.dim * [0]
    # for i, feature in enumerate(self._x):
        # dists[i] = empiric(feature).params
    return dists


def file_exists(file: str = 'tmp.rbig') -> str:
    path, base = dirname(file), basename(file)
    root, ext = base.split('.', 1)
    i, sep = 0, len(root) + len(path) + 1
    while isfile(file):
        file = '{}-{:d}.{}'.format(file[:sep], i, ext)
        i += 1
    return file


class RBIG_file:
    """Rotation based iterative gaussianization (arXiv:1602.00229).
    
    Design matrix is transposed for efficiency reasons.
    All transformations are saved to a file."""

    def __init__(
            self,
            x: ndarray,
            epochs: int,
            seed: Optional[int] = None,
            file: str = '../tmp/tmp.rbig'):
        self._x = copy(x.T)
        self.dim = x.shape[1]
        self.epoch_nr = 0  # !
        self.file = file_exists(file)
        self.fit(epochs, seed)

    def fit(self, epochs: int, seed: Optional[int] = None):
        randseed(seed)
        for epoch in range(epochs):
            dists = dists_of_matrix(self._x)
            rot = qr(randn(self.dim, self.dim))[0]
            self._x = gaussianize_matrix(self._x, dists)
            self._x = rot @ self._x
            ks = mean([kstest(y, 'norm')[0] for y in self._x])
            self._save([self.epoch_nr, ks, dists, rot])
            self.epoch_nr += 1

    def __call__(self, x): return self.encode(x)
    def __inv__(self, x): return self.decode(x)

    def encode(self, x: ndarray) -> ndarray:
        self._x = copy(x.T)
        for epoch, ks, dists, rot in self._load():
            self._x = gaussianize_matrix(self._x, dists)
            self._x = rot @ self._x
        return self.x

    def decode(self, x: ndarray) -> ndarray:
        self._x = copy(x.T)
        for epoch, ks, dists, rot in reversed(list(self._load())):
            self._x = rot.T @ self._x
            self._x = inv(gaussianize_matrix)(self._x, dists)
        return self.x

    def kstest(self,
               show: bool = False,
               save: bool = False,
               path: str = 'KS.png') -> Series:
        ks = []
        for epoch, k, dists, rot in self._load():
            ks += [k]
        ks = Series(ks, name='KS-test')
        ks.index.name = 'Epochen'
        ks.index += 1

        if show or save:
            ks.plot(title='Kolmogorow-Smirnow-Test', loglog=True, legend=False, color='k', figsize=(3, 2))
            #plt.ylabel('KS Test Statistic')
            if save:
                plt.savefig(path, bbox_inches='tight')
            plt.show() if show else plt.close()
        return ks

    def _save(self, params: list):
        makedirs(dirname(self.file), exist_ok=True)
        with open(self.file, 'ab') as file:
            pickle.dump(params, file, pickle.HIGHEST_PROTOCOL)

    def _load(self) -> Generator[List, None, None]:
        with open(self.file, "rb") as f:
            while True:
                try:
                    yield pickle.load(f)
                except EOFError:
                    break

    @property
    def rem(self):
        if isfile(self.file):
            remove(self.file)
    x = property(lambda self: self._x.T)


def inspect_seeds(
        epochs: Optional[int] = None,
        seeds: Optional[Sequence[int]] = None) -> (int, List[int]):
    """
    Assert number of epochs matches number of seeds and that seeds is a list.
    :param epochs: Number of epochs for training.
    :param seeds: Seeds for random number generation.
    :return: Inspected epochs and seeds.
    """
    # Assert that seeds is of type list.
    if isinstance(seeds, int) or seeds is None:
        randseed(seeds)
        seeds = []
    else:
        seeds = list(seeds)
    # Optionally determine number of epochs from number of seeds.
    if epochs is None:
        epochs = len(seeds)
    # Match number of epochs with number of seeds.
    if epochs > len(seeds):
        # Extend random values to seeds.
        seeds += list(randint(
            2 ** 32 - 1,  # Max accepted range for numpy seed generator.
            size=epochs - len(seeds),
            dtype=int64))
    elif epochs < len(seeds):
        seeds = seeds[:epochs]
    return epochs, seeds


class RBIG:
    """
    Rotation based iterative gaussianization (arXiv:1602.00229).
    :member _x: Transposed design matrix with shape (features, samples).
    :member features: Number of features of _x.
    :member samples: Number of samples of _x.
    :member epoch_nr: Current epoch number of training.
    :member seeds: Seeds for epoch wise generation of random rotation matrices.
    :member dists: Collection of feature wise empirical distributions of _x with shape (epochs, features, samples).
    :member ks: Epoch wise Kolmogorow-Smirnow-Test-statistics.
    """

    def __init__(
            self,
            x: ndarray,
            epochs: Optional[int] = None,
            seeds: Optional[Sequence[int]] = None,
            eps: float = 1e-7):
        """
        Initializes RBIG.
        :param x: Design matrix with shape (samples, features).
        :param epochs: Number of epochs for initial training.
        :param seeds: Seeds for epoch wise generation of random rotation matrices for initial training.
        :param eps: Precision parameter for marginal gaussianization.
        """
        self._x = copy(x.T)
        self.features = self._x.shape[0]
        self.samples = self._x.shape[1]
        self.epoch_nr = 0
        self.dists = None
        self.seeds = []
        self.ks = []
        self.eps = eps
        epochs, seeds = inspect_seeds(epochs, seeds)
        # initial gaussianization
        self.fit(epochs, seeds)

    def _empirical_distributions(self):
        """
        Determines feature wise cumulated empirical distributions at recent epoch.
        Correct support at the edges by a small tolerance value.
        """
        dists = zeros((2,) + self._x.shape)
        # Feature wise sorting and counting.
        for feature in range(self.features):
            vals, cnts = unique(self._x[feature], return_counts=True)
            dists[0, feature, :len(vals)] = vals
            dists[1, feature, :len(vals)] = cnts
            # correct length of distributions, otherwise interp breaks!
            dists[0, feature, len(vals):] = nan
        # Cumulate counts feature wise.
        dists[1] = cumsum(dists[1], axis=1)
        # Apply feature wise [0, 1]-normalization.
        dists[1] /= expand_dims(dists[1, :, -1], axis=1)
        # Clip empirical cdf into open interval (0, 1).
        dists[1] = clip(dists[1], self.eps, 1. - self.eps)
        self.dists[self.epoch_nr] = dists

    def _marginalcdf(self, epoch: int):
        xk, ck = self.dists[epoch]
        for feature in range(self.features):
            self._x[feature] = interp(self._x[feature], xk[feature], ck[feature])

    def _marginalppf(self, epoch: int):
        xk, ck = self.dists[epoch]
        for feature in range(self.features):
            self._x[feature] = interp(self._x[feature], ck[feature], xk[feature])

    def _mgauss(self, epoch: int):
        self._marginalcdf(epoch)
        self._x = norm.ppf(self._x)

    def _igauss(self, epoch: int):
        self._x = norm.cdf(self._x)
        self._marginalppf(epoch)

    def fit(
            self,
            epochs: Optional[int] = None,
            seeds: Optional[Sequence[int]] = None):
        """
        Fit gaussianization transform.
        :param epochs: number of epochs to train; inferred from seeds if None
        :param seeds: sequence of seeds for the random rotation matrices; random if None
        :return:
        """
        epochs, seeds = inspect_seeds(epochs, seeds)
        if epochs == 0:
            return
        # Expand empirical distributions by incoming epochs.
        dists = zeros((epochs, 2, self.features, self.samples))
        if self.dists is None:
            self.dists = dists
        else:
            self.dists = concatenate((self.dists, dists))

        for epoch, seed in enumerate(seeds):
            self._empirical_distributions()
            randseed(seed)
            rot = qr(randn(self.features, self.features))[0]

            # ToDo: rot is all NaN in first epoch.
            """
            f.e. prepare_gauss, plot_anomalies, plot_comparison and plot_time_series do work in terminal
            but break if called from main.
            Warnings:
                RuntimeWarning: invalid value encountered in greater_equal
                cond2 = (x >= np.asarray(_b)) & cond0
                RuntimeWarning: invalid value encountered in less_equal
                cond2 = cond0 & (x <= _a)
                RuntimeWarning: invalid value encountered in greater
                cond1 = (0 < q) & (q < 1)
                RuntimeWarning: invalid value encountered in less
                cond1 = (0 < q) & (q < 1)
            """
            if any(isnan(rot)):
                raise ValueError('Rotation matrix contains NaN in epoch %d from seed %d.' % (epoch, seed))

            self._mgauss(epoch)
            self._x = rot @ self._x
            # Update members.
            self.ks += [mean([kstest(feature, 'norm')[0] for feature in self._x])]
            self.epoch_nr += 1
            self.seeds += [seed]

    def __call__(self, x):
        return self.encode(x)

    def __inv__(self, x):
        return self.decode(x)

    def encode(self, x: ndarray) -> ndarray:
        self._x = copy(x.T)
        for epoch, seed in enumerate(self.seeds):
            randseed(seed)
            rot = qr(randn(self.features, self.features))[0]
            self._mgauss(epoch)
            self._x = rot @ self._x
        return self.x

    def decode(self, x: ndarray) -> ndarray:
        self._x = copy(x.T)
        for epoch in reversed(range(self.epoch_nr)):
            randseed(self.seeds[epoch])
            rot = qr(randn(self.features, self.features))[0]
            self._x = rot.T @ self._x
            self._igauss(epoch)
        return self.x

    def kstest(
            self,
            show: bool = False,
            save: bool = False,
            path: str = 'KS.png') -> Series:
        """
        Feature wise KS-tests.
        Optionally plotting and saving.
        """
        ks = Series(self.ks, name='KS-test')
        ks.index.name = 'Epochen'
        ks.index += 1

        if show or save:
            ks.plot(title='Kolmogorow-Smirnow-Test', loglog=True, legend=False, color='k', figsize=(3, 2))
            if save:
                plt.savefig(path, bbox_inches='tight')
            plt.show() if show else plt.close()
        return ks

    x = property(lambda self: self._x.T)
