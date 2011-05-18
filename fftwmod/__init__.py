from numpy.testing import Tester
from .wraps import *

__all__ = wraps.__all__
fftw_ext = wraps.fftw_ext

test = Tester().test
bench = Tester().bench
