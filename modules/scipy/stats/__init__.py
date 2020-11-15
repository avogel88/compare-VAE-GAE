__all__ = [] 


from .stats import *

from . import stats
__all__ += stats.__all__
del stats


from scipy._lib._testutils import PytestTester
test = PytestTester(__name__)
del PytestTester