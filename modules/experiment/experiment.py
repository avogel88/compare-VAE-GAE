import os
import numpy as np
from configparser import ConfigParser, ExtendedInterpolation
from itertools import chain, product
from numpy.random import randn
from os.path import dirname
from .texfig import pgf


__all__ = ['experiments', 'Experiment', 'config', 'prepare_z']


def Experiment(
        dataset: str = '',
        model: str = '',
        run: int = 0):
    """Organize experiment.

    Using dtype for accessing fields by names.
    Using record array for accessing fields as members.
    E.g. a['dataset'] or a.dataset
    """
    dtype = [('dataset', 'U30'), ('model', 'U30'), ('run', 'int')]
    return np.rec.array((dataset, model, run), dtype=dtype)


def experiments(datasets, models, runs):
    """Prepare the experiments.

    input:
    datasets:
        [dataset: str, ]
        names of datasets
    runs:
        [run: int, ]
        number of runs of corresponding dataset
    models:
        [[basemodel: str, model: str], ]
        names of models

    output:
    experiments: [((dataset: str, run: int), [basemodel: str, model: str]), ]
    """
    ds = [[dataset] * run for dataset, run in zip(datasets, runs)]
    run = chain(*[range(run) for run in runs])
    dsr = zip(chain(*ds), run)
    return product(dsr, models)


def config(section, path='experiments.ini', **kwargs):
    conf = ConfigParser(interpolation=ExtendedInterpolation())
    conf.read(path)
    # config.sections()
    for key, val in kwargs.items():
        kwargs[key] = str(val)
    conf['VARS'].update(kwargs)
    return conf[section]


def prepare_z(file, samples=60000, dim=10):
    z = randn(samples, dim).astype(np.float32)
    os.makedirs(dirname(file), exist_ok=True)
    with open(file, 'wb') as f:
        np.save(f, z)


if config('GENERAL')['img'] == 'pgf':
    pgf()
