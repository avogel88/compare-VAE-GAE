import numpy as np
from numpy.testing import (assert_array_equal, assert_almost_equal,
                           assert_array_almost_equal, assert_equal, assert_)


from modules.scipy.special import VeroneseMap, VeroneseMapWithIdentity


def test_veronese_map():
    x = np.random.randn(10)
    n, m = len(x), 784
    V = VeroneseMap(shape=(n, m), shuffle=True)
    z = V(x)
    x_ = V(z, inverse=True)
    assert_array_almost_equal(x, x_)

def test_VeroneseMapWithIdentity():
    samples, dim_x, dim_z = 2, 10, 784
    x = np.random.randn(samples, dim_x)
    V = VeroneseMapWithIdentity(shape=(dim_x, dim_z))
    z = V(x)
    x_ = V(z, inverse=True)
    assert_array_almost_equal(x, x_)
    assert_almost_equal(VeroneseMapWithIdentity.l_compare(V.l(x), V.l(x_)), 0.0)