import numpy as np
import tensorflow as tf


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


def t(x): return tf.transpose(x)


def corrcoef(x, y):
    # fix type
    x_t = tf.constant(x)
    y_t = tf.constant(y)
    # fix dtype
    x_t = tf.cast(x_t, tf.float64)
    y_t = tf.cast(y_t, tf.float64)
    # fix dimensions
    if x_t.ndim == 1:
        x_t = tf.expand_dims(x_t, axis=0)
    if y_t.ndim == 1:
        y_t = tf.expand_dims(y_t, axis=0)
    # actual corrcoef
    xy_t = tf.concat([x_t, y_t], axis=0)
    mean_t = tf.reduce_mean(xy_t, axis=1, keepdims=True)
    cov_t = ((xy_t-mean_t) @ t(xy_t-mean_t))/(x_t.shape[-1]-1)
    cov2_t = tf.linalg.diag(1/tf.sqrt(tf.linalg.diag_part(cov_t)))
    cor = cov2_t @ cov_t @ cov2_t
    return cor.numpy()

# x, y = np.random.randn(3, 3), np.random.randn(3, 3)
# np.testing.assert_allclose(np.corrcoef(x, y), corrcoef(x, y))
