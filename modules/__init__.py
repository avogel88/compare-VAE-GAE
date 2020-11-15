__all__ = [] 


import tensorflow.python.util.deprecation as deprecation
from scipy._lib._testutils import PytestTester

deprecation._PRINT_DEPRECATION_WARNINGS = False
test = PytestTester(__name__)
del PytestTester
