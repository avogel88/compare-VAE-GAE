import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Dense
from tensorflow.keras.losses import KLD, MSE


__all__ = ['Variational']


def normpdf(sample: tf.Tensor,
            mean: float = 0.,
            logvar: float = 0.,
            axis: int = 1) -> tf.Tensor:
    return tf.exp(lognormpdf(sample, mean, logvar, axis))


def lognormpdf(sample: tf.Tensor,
               mean: float = 0.,
               logvar: float = 0.,
               axis: int = 1) -> tf.Tensor:
    log2pi = tf.math.log(2. * np.pi)
    return tf.reduce_sum(
        -.5 * ((sample - mean) ** 2. * tf.exp(-logvar) + logvar + log2pi),
        axis=axis)


class Variational(Dense):
    """Variational Bayes Layer.

    Empirically captures mean and logarithmic variance of the source distribution.
    Considers the ELBO of the marginal likelihood as loss which consists of
    the divergence between posterior and prior and the reconstruction loss.
    Here the divergence between posterior and prior is treated as variational loss.

    ELBO = E_q(z|x) log p(x|z) - KLD(q(z|x), p(z))
    KLD(q(z|x), p(z)) = E_q(z|x) log[q(z|x)/p(z)]
                      = E_q(z|x) [log q(z|x) - log p(z)]

    Single sample Monte Carlo estimate of the ELBO:
    ELBO = E [log p(x|z) + log p(z) - log q(z|x)]
    Source: https://www.tensorflow.org/tutorials/generative/cvae#define_the_loss_function_and_the_optimizer
    """
    def __init__(self, units: int):
        super().__init__(2*units)

    def call(self, inputs: tf.Tensor) -> tf.Tensor:
        # Mean & Variance in one Dense Layer
        latent = super().call(inputs)
        
        # Reparameterization
        mean, logvar = tf.split(latent, num_or_size_splits=2, axis=1)
        eps = tf.random.normal(shape=tf.shape(mean))
        z = eps * tf.exp(logvar * .5) + mean
        
        # Loss
        logpz = lognormpdf(z)
        logqz_x = lognormpdf(z, mean, logvar)
        div = -tf.reduce_mean(logpz - logqz_x)
        self.add_loss(div)
        # self.add_metric(value=logpz, name='logpz')
        # self.add_metric(value=logqz_x, name='logqz_x')
        # problems with metrics:
        #   logpz & logqz_x are not scalars
        #   enc: corrcoef(logpx, logqz_x)
        #   rec: corrcoef(logpx, logpx_z)
        #   gen: corrcoef(logpz, logpx_z)
        """Alternative with KLD (was not used during the experiments).
        
        pz = normpdf(z)
        qz_x = normpdf(z, mean, logvar)
        self.add_loss(KLD(qz_x, pz))
        """
        return z
    
    @tf.function
    def MSE(x: tf.Tensor, x_decoded: tf.Tensor) -> tf.Tensor:
        """MSE-loss optimized for variational inference.

        MSE = E_q(z|x) log p(x|z)
        Here in conjunction to the variational loss:
        MSE = E log p(x|z)
        """
        cross_ent = MSE(x, x_decoded)
        cross_ent = tf.reshape(cross_ent, [tf.shape(x)[0], -1])
        logpx_z = -tf.reduce_sum(cross_ent, axis=1)
        return -tf.reduce_mean(logpx_z)
