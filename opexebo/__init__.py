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
from . import defaults
from . import errors
from . import analysis
from . import general

__author__ = """Simon Ball"""
__email__ = 'simon.ball@ntnu.no'
__version__ = '0.5.1'
