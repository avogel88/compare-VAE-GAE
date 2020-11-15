import tensorflow as tf

import numpy as np
from numpy.testing import (assert_array_equal, assert_almost_equal,
                           assert_array_almost_equal, assert_equal, assert_)

from modules.tensorflow.keras.layers import lognormpdf, normpdf, Variational
from scipy.stats import norm, kstest
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input, Dense


def test_normpdf():
    a = np.arange(9).reshape([3, 3])
    b = np.sum(norm.logpdf(a), axis=1)
    c = lognormpdf(a)
    assert_array_almost_equal(b, c.numpy(), 5)
    b = np.prod(norm.pdf(a), axis=1)
    c = normpdf(a)
    assert_array_almost_equal(b, c.numpy(), 5)


class TestVariationalLayer(tf.test.TestCase):

    def setUp(self):
        super(TestVariationalLayer, self).setUp()
        self.n = n = 10
        self.model = Sequential([Input(shape=(n,)), Variational(n), Dense(n)], 'TinyAE')
        self.var = Sequential([Input(shape=(n,)), Variational(n)], 'TinyEncoder')
        self.model.compile(optimizer='adam', loss=Variational.MSE)

    def test_output(self):
        x = np.random.randint(5, size=(1000, self.n)) / 10
        self.model.fit(x, x, epochs=100, verbose=0)
        z = self.var.predict(x)
        y = self.model.predict(x)
        # test shape of output
        self.assertAllEqual(y.shape, x.shape)
        # test normality of latent space
        statistics, pvalues = np.apply_along_axis(kstest, 0, z, 'norm')
        # self.assertAllGreater(pvalues, .1)
        self.assertLess(statistics.mean(), .1)
