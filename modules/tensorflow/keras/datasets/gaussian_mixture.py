import numpy as np
import os
from os.path import dirname, join
from modules.numpy import covmix, varroll
from modules.pandas import DesignMatrix
from modules.scipy.stats import gaussian_mixture


def gaussian_mixture_generate(file, train=60000, test=60000, validate=10000):
    dim_x, dim_z = 784, 10

    # distributions
    π = [.2, .3, .5]
    K, N, D = len(π), dim_z, dim_x
    µ = np.zeros((K, D))
    Σ = covmix(varroll(range(K), (N, D - N), (10, .1)))
    x_dist = gaussian_mixture(weights=π, mean=µ, cov=Σ)

    # sampling
    x = DesignMatrix(x_dist.rvs(train))
    y = DesignMatrix(x_dist.rvs(test))
    z = DesignMatrix(x_dist.rvs(validate))

    # save distribution
    os.makedirs(dirname(file), exist_ok=True)
    x_dist.save(file)

    # save
    x.to_csv(join(dirname(file), 'train.csv'))
    y.to_csv(join(dirname(file), 'test.csv'))
    z.to_csv(join(dirname(file), 'validate.csv'))


def gaussian_mixture_load(path):
    x_dist = gaussian_mixture.load(path)
    x = DesignMatrix.read_csv(join(dirname(path), 'train.csv'))
    y = DesignMatrix.read_csv(join(dirname(path), 'test.csv'))
    z = DesignMatrix.read_csv(join(dirname(path), 'validate.csv'))
    return x_dist, x, y, z
