__all__ = [] 


from .designmatrix import *

from . import designmatrix
__all__ += designmatrix.__all__
del designmatrix


from scipy._lib._testutils import PytestTester
test = PytestTester(__name__)
del PytestTester
