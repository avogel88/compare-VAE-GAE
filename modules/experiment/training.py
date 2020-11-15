from datetime import datetime as dt
from itertools import product

import numpy as np
from scipy.stats import multivariate_normal
from tensorflow.keras.callbacks import ModelCheckpoint, TensorBoard
from tensorflow.keras.layers import Dense

from . import Autoencoder, layers, GAE, config
from ..tensorflow.keras.callbacks import RBIGCallback, ImageCallback, StatsCallback
from ..tensorflow.keras.datasets import gaussian_mixture_load, import_mnist
from ..tensorflow.keras.layers import Variational

__all__ = ['continued_training']


def continued_training(section):
    """
    Continues training from last state.
    Load paths and parameters from ini-file.
    :param section: QUAL or QUANT
    :return:
    """
    def ini_params(runs, epochs, models, datasets, **kwargs):
        return int(runs), int(epochs), models.split(','), datasets.split(',')

    def prepare_models(model):
        if model == 'vae':
            enc, dec = layers(Variational, dim_z)
            model = Autoencoder(enc, dec, dim_z,
                                name='VariationalAutoencoder',
                                short_name='VAE',
                                loss=Variational.MSE)
            return model, None
        else:
            enc, dec = layers(Dense, dim_z)
            model = Autoencoder(enc, dec, dim_z,
                                name='Autoencoder',
                                short_name='AE',
                                loss='mse')
            gae = GAE(model)
            return model, gae

    def load_data(dataset, dim_z):
        seed = np.load(config('DATA')['normal_samples'])
        if dataset == 'gauss':
            x_dist, train, test, validate = gaussian_mixture_load(config('DATA')['gaussian_mixture'])
            # reshape data to (-1, 28, 28, 1)
            train, test, validate = train.S, test.S, validate.S
            z_dist = multivariate_normal(cov=np.eye(dim_z))
            seed = seed[:len(test)]
            return train, test, validate, seed, x_dist, z_dist
        else:
            (train, _), (test, _) = import_mnist(dataset)
            validate = test
            seed = seed[:len(test)]
            return train, test, validate, seed, None, None

    def prepare_callbacks(dataset, test, seed, latentseeds, x_dist, ae, gae, experiment):
        # prepare callbacks for training
        # handling the application of RBIG in AE
        # saving checkpoints
        # handling tensorboard and previews
        label = ae.short_name
        dim_z, period, imgs = [int(config('GENERAL')[s]) for s in ['dim_z', 'save_period', 'nr_imgs']]

        # different config handles for different models
        conf = config('PATHS', **experiment)
        ex = experiment.copy()
        ex['model'] = 'gae'
        confgae = config('PATHS', **ex)

        callbacks = []
        if gae is not None:
            # RBIG needs to be first
            rb_callback = RBIGCallback(test, gae, confgae['kstest'], period, latentseeds)
            img_callback = ImageCallback(test, seed, gae, imgs, 'GAE', confgae['imgs'])()
            callbacks += [rb_callback, img_callback]
        cp_callback = ModelCheckpoint(filepath=conf['ckpt'],
                                      save_weights_only=True,
                                      verbose=0,
                                      period=period)
        tb_callback = TensorBoard(log_dir=conf['root'], profile_batch=0, histogram_freq=1)
        im_callback = ImageCallback(test, seed, ae, imgs, label, conf['imgs'])()
        callbacks += [cp_callback, tb_callback, im_callback]
        if dataset == 'gauss':
            # callbacks for calculating r²
            st_callback = StatsCallback(test, seed, x_dist, ae, label,
                                        datadir=conf['lhood'],
                                        traindir=conf['root'],
                                        period=period)
            callbacks += [st_callback]
            if gae is not None:
                # additional callback for GAE
                stg_callback = StatsCallback(test, seed, x_dist, gae, 'GAE',
                                             datadir=confgae['lhood'],
                                             traindir=confgae['root'],
                                             period=period)
                callbacks += [stg_callback]
        return callbacks

    # load from ini-file
    dim_z, period, imgs = [int(config('GENERAL')[s]) for s in ['dim_z', 'save_period', 'nr_imgs']]
    runs, epochs, models, datasets = ini_params(**config(section))

    # prepare experiments
    for dataset in datasets:
        train, test, validate, seed, x_dist, z_dist = load_data(dataset, dim_z)
        for model, run in product(models, range(runs)):
            ex = {
                'dataset': dataset,
                'model': model,
                'run': run,
            }
            ae, gae = prepare_models(model)
            # continue training from last checkpoint
            latest = ae.load_checkpoint(config('PATHS', **ex)['ckpt'])
            # prevent empty logs from finished / aborted training
            # ToDo: prevent empty logs from manual abort
            if epochs > latest:
                if dataset == 'gauss':
                    callbacks = prepare_callbacks(dataset, test, seed, 12345, x_dist.logpdfrelevant, ae, gae, ex)
                else:
                    callbacks = prepare_callbacks(dataset, test, seed, 12345, None, ae, gae, ex)

                print('%s: %s, epochs: [%d - %d]' % (dt.now(), ex, latest, epochs - 1))
                hist = ae.fit(train, train, 256, epochs, 0,
                              validation_data=(validate, validate),
                              initial_epoch=latest,
                              callbacks=callbacks)
