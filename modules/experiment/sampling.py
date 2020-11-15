import matplotlib.pyplot as plt
import numpy as np
import os

from os.path import dirname
from scipy.stats import multivariate_normal
from sklearn.metrics import auc, roc_curve
from tensorflow.python.keras.layers import Dense

from modules.experiment import config, layers, Autoencoder, GAE, Experiment
from modules.experiment.plot import plot_strip, strip
from modules.tensorflow.keras.datasets import gaussian_mixture_load, import_mnist
from modules.tensorflow.keras.layers import Variational

__all__ = ['anomaly_detection', 'generate_samples', 'plot_anomalies', 'plot_comparison', 'plot_time_series']

"""Generate plots: comparison of reconstructed and generated content, time series, ROC for anomalies."""


def generate_samples(experiment, dim_z, imgs=20, epoch=None, randseed=None):
    ex = {
        'dataset': experiment.dataset,
        'model'  : experiment.model,
        'run'    : experiment.run
    }
    if experiment.model == 'gae':
        ex['model'] = 'ae'

    # load dataset
    seed = np.load(config('DATA')['normal_samples'])
    if experiment.dataset == 'gauss':
        x_dist, train, test, validate = gaussian_mixture_load(config('DATA')['gaussian_mixture'])
        train, test, validate = train.S, test.S, validate.S
        z_dist = multivariate_normal(cov=np.eye(dim_z))
        dists = [x_dist, z_dist]
    else:
        (train, _), (test, _) = import_mnist(experiment.dataset)
        validate = test
    gauss = test
    test = test[:imgs]
    seed = seed[:imgs]

    # models
    if experiment.model in ('ae', 'gae'):
        enc, dec = layers(Dense, dim_z)
        model = Autoencoder(
            enc, dec, dim_z,
            name='Autoencoder',
            short_name='AE',
            loss='mse')
        latest = model.load_checkpoint(config('PATHS', **ex)['ckpt'], epoch)
        if experiment.model == 'gae':
            model = GAE(model)
            model.gauss(gauss, seeds=randseed)
    else:
        enc, dec = layers(Variational, dim_z)
        model = Autoencoder(
            enc, dec, dim_z,
            name='VariationalAutoencoder',
            short_name='VAE',
            loss=Variational.MSE)
        latest = model.load_checkpoint(config('PATHS', **ex)['ckpt'], epoch)

    # recoding
    encoded = model.encode(test)
    reconstructed = model.decode(encoded)
    generated = model.decode(seed)

    return reconstructed, generated, encoded


def plot_comparison(seed=None):
    """Compare reconstructions and generated samples."""
    kwargs = {'dim_z': 10, 'imgs': 20, 'randseed': seed}
    mnist_ae = generate_samples(Experiment('mnist', 'ae', 0), **kwargs)
    fashion_ae = generate_samples(Experiment('fashion', 'ae', 0), **kwargs)
    mnist_vae = generate_samples(Experiment('mnist', 'vae', 0), **kwargs)
    fashion_vae = generate_samples(Experiment('fashion', 'vae', 0), **kwargs)
    mnist_gae = generate_samples(Experiment('mnist', 'gae', 0), **kwargs)
    fashion_gae = generate_samples(Experiment('fashion', 'gae', 0), **kwargs)
    (_, _), (mnist_test, _) = import_mnist('mnist')
    (_, _), (fashion_test, _) = import_mnist('fashion')

    # saving
    plot_strip(
        images=[mnist_test[:20], mnist_ae[0], mnist_vae[0], mnist_gae[0], mnist_ae[1], mnist_vae[1], mnist_gae[1]],
        n=20, path=config('PATHS')['mnist_samples'])
    plot_strip(
        images=[fashion_test[:20], fashion_ae[0], fashion_vae[0], fashion_gae[0], fashion_ae[1], fashion_vae[1], fashion_gae[1]],
        n=20, path=config('PATHS')['fashion_samples'])


def plot_time_series(time_series=(10, 20, 100), seed=None):
    path = config('PATHS')['timeline']
    os.makedirs(dirname(path), exist_ok=True)
    # np.random.seed(seed)
    # seed = np.random.randint(
    #         2 ** 32 - 1,  # Max accepted range for numpy seed generator.
    #         size=100,
    #         dtype=np.int64)
    datasets = ['mnist', 'fashion']
    models = ['vae', 'gae']
    num_imgs = 2
    # shape: models, time_series, images, dimy, dimx, channels
    mnist_time_series = np.zeros((len(models), len(time_series), num_imgs, 28, 28, 1))
    fashion_time_series = np.zeros_like(mnist_time_series)
    for i, epoch in enumerate(time_series):
        for j, model in enumerate(models):
            # images.shape = (images, dimy, dimx, channels)
            kwargs = {'dim_z': 10, 'imgs': num_imgs, 'epoch': epoch, 'randseed': seed}
            mnist_time_series[j, i, :] = generate_samples(Experiment('mnist', model, 0), **kwargs)[1]
            fashion_time_series[j, i, :] = generate_samples(Experiment('fashion', model, 0), **kwargs)[1]
    # shape: models, time_series, dimy, images*dimx, channels
    mnist_time_series_r = np.array([strip(*model) for model in mnist_time_series])
    fashion_time_series_r = np.array([strip(*model) for model in fashion_time_series])
    # ylabels = ['%s %s' % names for names in product(datasets, models)]
    ylabels = 2*[model.upper() for model in models]
    plot_strip(
        [i for i in mnist_time_series_r] + [i for i in fashion_time_series_r],
        n=num_imgs*len(time_series),
        xlabels=time_series,
        ylabels=ylabels,
        path=path)


def anomaly_detection(experiment, dataset, dim_z, imgs=20, epoch=None, randseed=None):
    """Encode anomalies and calculate their likelihoods / MSE."""
    ex = experiment
    if experiment.model == 'gae':
        ex = Experiment(experiment.dataset, 'ae', experiment.run)

    (train, _), (test, _) = import_mnist(experiment.dataset)
    gauss = test
    (_, _), (anomaly, _) = import_mnist(dataset)
    anomaly = anomaly[:imgs]

    # models
    if experiment.model in ('ae', 'gae'):
        kwargs = {'name': 'Autoencoder', 'short_name': 'AE', 'loss': 'mse'}
        enc, dec = layers(Dense, dim_z)
    else:
        kwargs = {'name': 'VariationalAutoencoder', 'short_name': 'VAE', 'loss': Variational.MSE}
        enc, dec = layers(Variational, dim_z)
    model = Autoencoder(enc, dec, dim_z, **kwargs)
    ex = {
        'dataset': ex.dataset,
        'model': ex.model,
        'run': ex.run
    }
    latest = model.load_checkpoint(config('PATHS', **ex)['ckpt'], epoch)
    if experiment.model == 'gae':
        model = GAE(model)
        model.gauss(gauss, seeds=randseed)

    an_enc = model.anomaly(anomaly, mode='enc')
    an_rec = model.anomaly(anomaly, mode='rec')
    return an_enc, an_rec


def plot_anomalies(models, orig, anomaly, path, num=10000, seed=None):
    """Plots ROC curves."""
    fig, axes = plt.subplots(1, 2, figsize=(6, 3))
    for model in models:
        kwargs = {'dataset': orig, 'dim_z': 10, 'imgs': num, 'randseed': seed}
        enc, rec = anomaly_detection(Experiment(orig, model, 0), **kwargs)
        enc_anomaly, rec_anomaly = anomaly_detection(Experiment(anomaly, model, 0), **kwargs)
        y_true = num * [0] + num * [1]
        y_enc_score = np.concatenate((enc_anomaly, enc))
        y_rec_score = np.concatenate((rec_anomaly, rec))
        fpr_enc, tpr_enc, thresh_enc = roc_curve(y_true, y_enc_score)
        fpr_rec, tpr_rec, thresh_rec = roc_curve(y_true, y_rec_score)
        axes[0].plot(tpr_enc, fpr_enc, lw=1, label='%s (AUC = %0.2f)' % (model.upper(), auc(tpr_enc, fpr_enc)))
        axes[0].set_title('Encodings')
        axes[1].plot(tpr_rec, fpr_rec, lw=1, label='%s (AUC = %0.2f)' % (model.upper(), auc(tpr_rec, fpr_rec)))
        axes[1].set_title('Rekonstruktionen')
    for ax in axes.flat:
        ax.set_xlabel('Falsch-Positiv-Rate')
        ax.set_ylabel('Richtig-Positiv-Rate')
        ax.legend(frameon=False)  # , loc="lower right")
        ax.label_outer()
    # fig.suptitle('ROC-Kurven für %s' % orig.upper(), fontsize=12)
    fig.tight_layout()
    os.makedirs(dirname(path), exist_ok=True)
    fig.savefig(path)
