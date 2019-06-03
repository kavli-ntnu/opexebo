"""Python neurology toolbox

Subpackages
-----------
analysis
	specialized neuroscience analysis functions (gridness score, place maps,
    e.t.c.)
general
    general signal processing function (smoothing, correlation, e.t.c.)


defaults
    default values for keyword analysis parameters
"""
from .analysis import *
from .general import *
from . import defaults

__author__ = """Vadim Frolov"""
__email__ = 'vadim.frolov@ntnu.no'
__version__ = '0.1.0'

