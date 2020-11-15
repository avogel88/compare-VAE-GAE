import numpy as np
from numpy.testing import (assert_array_equal, assert_almost_equal,
                           assert_array_almost_equal, assert_equal, assert_)


from modules.numpy.linalg import matrix_append, gauss_jordan, bool_solve, bool_rank


class TestLinalg:
    def test_gauss_jordan(self):
        A = np.array([[0, 0, 0, 0, 1, 1, 0, 0, 0, 1],
                      [0, 1, 0, 1, 0, 1, 1, 1, 0, 0],
                      [0, 0, 1, 0, 0, 0, 0, 0, 1, 1],
                      [1, 0, 0, 0, 1, 0, 0, 0, 1, 0],
                      [0, 1, 0, 0, 1, 1, 0, 0, 0, 0],
                      [0, 1, 0, 1, 0, 0, 0, 0, 1, 0],
                      [0, 0, 1, 1, 0, 0, 1, 0, 0, 0],
                      [0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
                      [0, 0, 0, 0, 1, 0, 0, 1, 0, 1],
                      [0, 0, 0, 0, 1, 0, 0, 0, 1, 1]])
        b = np.array([1, 0, 1, 1, 1, 1, 0, 1, 0, 1])
        x = np.array([1, 1, 0, 0, 0, 0, 0, 1, 0, 1])
        assert_array_equal(matrix_append(np.diag(np.ones(10)), x),
                           gauss_jordan(matrix_append(A, b)))

    def test_bool_solve(self):
        A = np.array([[0, 0, 0, 0, 1, 1, 0, 0, 0, 1],
                      [0, 1, 0, 1, 0, 1, 1, 1, 0, 0],
                      [0, 0, 1, 0, 0, 0, 0, 0, 1, 1],
                      [1, 0, 0, 0, 1, 0, 0, 0, 1, 0],
                      [0, 1, 0, 0, 1, 1, 0, 0, 0, 0],
                      [0, 1, 0, 1, 0, 0, 0, 0, 1, 0],
                      [0, 0, 1, 1, 0, 0, 1, 0, 0, 0],
                      [0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
                      [0, 0, 0, 0, 1, 0, 0, 1, 0, 1],
                      [0, 0, 0, 0, 1, 0, 0, 0, 1, 1]])
        b = np.array([1, 0, 1, 1, 1, 1, 0, 1, 0, 1])
        x = np.array([1, 1, 0, 0, 0, 0, 0, 1, 0, 1])
        assert_array_equal(x, bool_solve(A, b))

    def test_bool_rank(self):
        A = np.array([[0, 0, 0, 0, 1, 1, 0, 0, 0, 1],
                      [0, 1, 0, 1, 0, 1, 1, 1, 0, 0],
                      [0, 0, 1, 0, 0, 0, 0, 0, 1, 1],
                      [1, 0, 0, 0, 1, 0, 0, 0, 1, 0],
                      [0, 1, 0, 0, 1, 1, 0, 0, 0, 0],
                      [0, 1, 0, 1, 0, 0, 0, 0, 1, 0],
                      [0, 0, 1, 1, 0, 0, 1, 0, 0, 0],
                      [0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
                      [0, 0, 0, 0, 1, 0, 0, 1, 0, 1],
                      [0, 0, 0, 0, 1, 0, 0, 0, 1, 1]])
        assert_array_equal(10, bool_rank(A))