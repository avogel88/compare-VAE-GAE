import os
import numpy as np
import tensorflow as tf
from os.path import dirname, join
from tensorflow.python.keras.callbacks import LambdaCallback, Callback
from modules.experiment.autoencoder import GAE
from modules.experiment.plot import strip

__all__ = ['RBIGCallback', 'StatsCallback', 'ImageCallback']


class RBIGCallback(Callback):
    def __init__(self, data, model, path, period=1, seeds=None):
        super().__init__()
        self.data = data
        self.extmodel = model
        self.period = period
        self.path = path
        self.seeds = seeds
        os.makedirs(dirname(self.path), exist_ok=True)

    def on_epoch_end(self, epoch, logs=None):
        # init RBIG
        if epoch % self.period == 0:
            encoded = self.extmodel.gauss(self.data, 100, self.seeds)
            np.save(self.path.format(epoch=epoch), self.extmodel.G.kstest())


class StatsCallback(Callback):
    """Apply functions (e.g. likelihoods, correlations...).

    ToDo: Access validation step of training to avoid multiple runs for reconstruction & encoding!
    """

    def __init__(self, data, seed, dist, model,
                 name='Operations',
                 datadir='logs/data',
                 traindir='logs',
                 period=1):
        super().__init__()
        self.data = data
        self.dist = dist
        self.seed = seed
        self.extmodel = model
        self.name = name
        self.period = period
        self.path = datadir
        os.makedirs(dirname(self.path), exist_ok=True)
        self.file_writer = tf.summary.create_file_writer(join(traindir, 'correlation'))

    def on_epoch_end(self, epoch, logs=None):
        """Write data to log."""
        if epoch % self.period == 0:
            with self.file_writer.as_default():
                rsq_rec, rsq_enc, rsq_gen = self.extmodel.corrcoef(self.data, self.seed, self.dist)
                tf.summary.scalar('enc_corr', rsq_enc, step=epoch)
                tf.summary.scalar('gen_corr', rsq_gen, step=epoch)
                tf.summary.scalar('rec_corr', rsq_rec, step=epoch)


class ImageCallback:
    """
    Logs reconstructions and generated samples as strips.
    ToDo: save images as png
    """
    def __init__(self, data, seed, model, num,
                 name='Samplings',
                 path='logs/imgs'):
        self.data = data[:num]
        self.seed = seed[:num]
        self.model = model
        self.num = num
        self.name = name
        self.file_writer = tf.summary.create_file_writer(path)

    def __call__(self):
        return LambdaCallback(on_epoch_end=self.log_sampling)

    def log_sampling(self, epoch, logs):
        """Write image to log."""
        if type(self.model) != GAE or (type(self.model) == GAE and self.model.G is not None):
            with self.file_writer.as_default():
                # inference
                reconstructed = self.model.predict(self.data)
                generated = self.model.decode(self.seed)
                # save images
                tf.summary.image(self.name,
                                 strip(reconstructed, generated),
                                 max_outputs=self.num,
                                 step=epoch)
