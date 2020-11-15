__all__ = [] 


from .variational import *

from . import variational
__all__ += variational.__all__
del variational


from scipy._lib._testutils import PytestTester
test = PytestTester(__name__)
del PytestTester
