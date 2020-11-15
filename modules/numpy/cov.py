import numpy as np
from typing import Sequence, Union


__all__ = ['randcorr', 'var2cov', 'varroll', 'covmix']


def randcorr(d: int, k: int) -> np.ndarray:
    """Generate random correlation matrix.

    https://stats.stackexchange.com/questions/124538/how-to-generate-a-large-full-rank-random-correlation-matrix-with-some-strong-cor#answer-125020
    d - number of dimensions
    k - number of factors
    """
    from numpy import diag, sqrt
    from numpy.random import rand, randn
    W = randn(d, k)
    S = W @ W.T + diag(rand(1, d))
    S = diag(1./sqrt(diag(S))) @ S @ diag(1./sqrt(diag(S)))
    return S


def var2cov(σ2: Sequence[float],
            k: int = 10) -> np.ndarray:
    """Construct covariance matrix with random correlation matrix and given marginal variances on its diagonal.

    k ∈ ℝ
        factors for correlation matrix
    σ2 ∈ ℝ^n
        marginal variances
    D ∈ ℝ^n,n
        standard deviations on diagonal
        D = diag(σ)
    P ∈ ℝ^n,n
        correlation matrix - linear dependence structure
        symmetric & positive semi-definite
        P = D^-1 Σ D^-1
        diag(P) = 1
    Σ ∈ ℝ^n,n
        covariance matrix
        symmetric & positive semi-definite
        Σ = D P D
        diag(Σ) = σ2

    symmetry: A = A.T
    positive semi-definiteness: ew(A) >= 0
    """
    if np.ndim(σ2) == 1:
        from numpy import diag, sqrt
        n = len(σ2)
        P = randcorr(n, k)
        D = diag(sqrt(σ2))
        Σ = D @ P @ D
        return Σ
    # mix
    return np.array([var2cov(var, len(var)) for var in σ2])


def varroll(pos: Union[int, Sequence[int]],
            quant: Sequence[int],
            var: Sequence[float]) -> np.ndarray:
    '''Generate rolled sequence/s of variances.'''
    if np.ndim(pos) == 0:
        lvar = []
        for q, v in zip(quant, var):
            lvar += q*[v]
        return np.roll(lvar, pos * quant[0])
    # mix
    return np.array([varroll(i, quant, var) for i in pos])


def covmix(varmix: np.ndarray) -> np.ndarray:
    '''Covariance matrices with given variances and random correlations.'''
    return var2cov(varmix, len(varmix))
