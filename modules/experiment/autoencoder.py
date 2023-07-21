import numpy as np
import os
import tensorflow as tf

from numpy.random import randn
from os.path import basename, dirname, join, isfile, isdir, splitext
from parse import parse
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, Layer, Reshape, Conv2D, Conv2DTranspose
from tensorflow.keras.losses import MSE
from tensorflow.keras.optimizers import Optimizer
from tensorflow.keras.optimizers import Adam
from ..scipy.special import RBIG
from ..tensorflow.stats import corrcoef, lognormpdf
from typing import List, Optional, Sequence, Union


__all__ = ['file_basename', 'list_ckpts', 'ckpt_nr', 'layers', 'Autoencoder', 'GAE']


def file_basename(path):
    """Reduce path to filename without (multi-) extensions."""
    path = basename(path)
    if '.' in path:
        separator_index = path.index('.')
        true_basename = path[:separator_index]
        return true_basename
    return path


def list_ckpts(path: str) -> List[str]:
    """List all checkpoints in path."""
    return [join(path, f) for f in os.listdir(path) if isfile(
        join(path, f)) and '.index' in f]


def ckpt_nr(path: (str, Sequence[str]),
            pattern: str = '{}{epoch:05d}') -> Union[int, List[int]]:
    """Return the epoch/s in the filename/s of the path/s.

    Error on ckpt_nr('ckpt_100')  # without '.index'
    """
    # one path?
    if np.ndim(path) == 0:
        # file?
        if isfile(path):
            return abs(parse(pattern, file_basename(path))['epoch'])
        # folder?
        path = list_ckpts(path)
    return sorted([ckpt_nr(file, pattern) for file in path])


def layers(latent_layer: Layer, latent_dim: int):
    kwargs = {
        'kernel_size': 3,
        'strides': (2, 2),
        'activation': 'relu',
    }
    encoder_layers = [
        #InputLayer(input_shape=(28, 28, 1)),
        Conv2D(filters=32, **kwargs,
               input_shape=(28, 28, 1)),
        Conv2D(filters=64, **kwargs),
        Conv2D(filters=64, **kwargs),
        Flatten(),
        latent_layer(latent_dim),
    ]

    decoder_layers = [
        Dense(units=3 * 3 * 32, activation='relu',
              input_shape=(latent_dim,)),
        Reshape(target_shape=(3, 3, 32)),
        Conv2DTranspose(
            filters=64,
            **kwargs),
        Conv2DTranspose(
            filters=64,
            padding="SAME",
            **kwargs),
        Conv2DTranspose(
            filters=32,
            padding="SAME",
            **kwargs),
        Conv2DTranspose(
            filters=1,
            kernel_size=3,
            strides=(1, 1),
            padding="SAME"),
    ]
    return encoder_layers, decoder_layers


class Autoencoder(Sequential):
    def __init__(self,
                 encoder: list = None,
                 decoder: list = None,
                 latent_dim: int = None,
                 name: str = 'Autoencoder',
                 short_name: str = 'AE',
                 optimizer: Optimizer = Adam,
                 learning_rate: int = 1e-5,
                 loss: str = 'mean_squared_error'):
        super(Autoencoder, self).__init__(encoder + decoder, name)
        self.latent_dim = latent_dim
        self.__name__ = name or self.__class__.__name__
        self.short_name = short_name

        self.encoder = Sequential(encoder, 'Encoder')
        self.decoder = Sequential(decoder, 'Decoder')

        # is optimizer initialized?
        if not isinstance(optimizer, Optimizer):
            optimizer = optimizer(learning_rate)
        self.compile(optimizer, loss)

    '''
    def __call__(self, x):
        if type(x) == int:
            return self.sample(x)
        return self.encode(x)
    def __inv__(self, x):
        if type(x) == int:
            return self.sample(x)
        return self.decode(x)
    '''

    def sample(self, num: int) -> np.ndarray:
        z = randn(num, self.latent_dim).astype(np.float32)
        return self.decode(z)

    def encode(self, x: np.ndarray) -> np.ndarray:
        return self.encoder.predict(x)

    def decode(self, z: np.ndarray) -> np.ndarray:
        return self.decoder.predict(z)

    def anomaly(self, x: np.ndarray, thresh=None, mode='normal') -> np.ndarray:
        """
        Detect anomalies.
        Normal mode determines the mse between reconstructions and inputs.
        Latent mode determines the likelihood of the encodings.
        Applies a threshold if specified.
        """
        if mode in ['normal', 'decoding', 'reconstruction', 'rec', 'x']:
            y = self.predict(x)
            l = MSE(x.reshape(len(x), -1), y.reshape(len(y), -1))
        if mode in ['latent', 'encoding', 'enc', 'z']:
            z = self.encode(x)
            l = lognormpdf(z)
        return l if thresh is None else l > thresh

    def corrcoef(self, x, z, logxpdf, logzpdf=lognormpdf):
        """
        Coefficients of determination for gaussianizing autoencoders.
        :param x: inputs; for encoding and reconstruction
        :param z: latent samples; for generating samples
        :param logxpdf: logpdf for input space
        :param logzpdf: logpdf for latent space
        :return: coefficients of determination for reconstructions, encodings and generated samples
        """
        z_x = self.encode(x)
        x_z_x = self.predict(x).reshape(len(x), -1)
        x_z = self.decode(z).reshape(len(x), -1)

        logpx = logxpdf(x.reshape(len(x), -1))  # original
        logqz_x = logzpdf(z_x)                  # encoded
        logpx_z_x = logxpdf(x_z_x)              # reconstructed
        logpz = logzpdf(z)                      # latent samples
        logpx_z = logxpdf(x_z)                  # generated samples

        rsq_rec = corrcoef(logpx, logpx_z_x)[0, 1]
        rsq_enc = corrcoef(logpx, logqz_x)[0, 1]
        rsq_gen = corrcoef(logpz, logpx_z)[0, 1]
        return rsq_rec, rsq_enc, rsq_gen

    def load_checkpoint(self, path: str, epoch: int = None) -> int:
        """
        Loads checkpoint from path and returns epoch count.
        Important note: path may not contain .index!
        """
        if epoch is None:
            if not isdir(path):
                path = dirname(path)
            ckpt = tf.train.latest_checkpoint(path)
        else:
            ckpt = path.format(epoch=epoch)
        if ckpt is not None:
            self.load_weights(ckpt).expect_partial()
            return ckpt_nr(ckpt+'.index')
        return 0


class GAE:
    """Iterative Gaussianization AE with iterative gaussianization (RBIG) of the latent space.

    Example:
        ae = AE(...)
        gae = GAE(ae)
        gae.ae.fit(...)
        gae.gauss(...)
    """
    def __init__(self,
                 ae: Sequential):
        self.ae = ae
        self.G = None
        self.__name__ = self.__class__.__name__

    def gauss(self,
              x: Optional[np.ndarray] = None,
              epochs: int = 100,
              seeds: Optional[int] = None):
        if x is None and self.G is not None:
            self.G.fit(epochs, seeds)
        else:
            z = self.ae.encode(x)
            self.G = RBIG(z, epochs, seeds)
        return self.G.x

    def fit(self, epochs: int = 100):
        self.G.fit(epochs)

    def predict(self, x: np.array):
        return self.ae.predict(x)

    def sample(self, num: int) -> np.ndarray:
        z = randn(num, self.ae.latent_dim).astype(np.float32)
        x = self.decode(z)
        return x

    def encode(self, x: np.ndarray) -> np.ndarray:
        z = self.ae.encode(x)
        if self.G is None:
            self.G = RBIG(z, 100)
            z = self.G.x
        else:
            z = self.G.encode(z)
        return z

    def decode(self, z: np.ndarray) -> np.ndarray:
        z = self.G.decode(z)
        x = self.ae.decode(z)
        return x

    def anomaly(self, x: np.ndarray, thresh=None, mode='normal') -> np.ndarray:
        """
        Detect anomalies.
        Normal mode determines the mse between reconstructions and inputs.
        Latent mode determines the likelihood of the encodings.
        Applies a threshold if specified.
        """
        if mode in ['normal', 'decoding', 'reconstruction', 'rec', 'x']:
            y = self.predict(x)
            l = MSE(x.reshape(len(x), -1), y.reshape(len(y), -1))
        if mode in ['latent', 'encoding', 'enc', 'z']:
            z = self.encode(x)
            l = lognormpdf(z)
        return l if thresh is None else l > thresh

    def corrcoef(self, x, z, logxpdf, logzpdf=lognormpdf):
        """
        Coefficients of determination for gaussianizing autoencoders.
        :param x: inputs; for encoding and reconstruction
        :param z: latent samples; for generating samples
        :param logxpdf: logpdf for input space
        :param logzpdf: logpdf for latent space
        :return: coefficients of determination for reconstructions, encodings and generated samples
        """
        z_x = self.encode(x)
        x_z_x = self.predict(x).reshape(len(x), -1)
        x_z = self.decode(z).reshape(len(x), -1)

        logpx = logxpdf(x.reshape(len(x), -1))  # original
        logqz_x = logzpdf(z_x)  # encoded
        logpx_z_x = logxpdf(x_z_x)  # reconstructed
        logpz = logzpdf(z)  # latent samples
        logpx_z = logxpdf(x_z)  # generated samples

        rsq_rec = corrcoef(logpx, logpx_z_x)[0, 1]
        rsq_enc = corrcoef(logpx, logqz_x)[0, 1]
        rsq_gen = corrcoef(logpz, logpx_z)[0, 1]
        return rsq_rec, rsq_enc, rsq_gen
