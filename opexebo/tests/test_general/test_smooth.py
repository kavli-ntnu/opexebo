""" Tests for smoothing"""
import numpy as np
import pytest

from opexebo.general import smooth as func

print("=== tests_general_smooth ===")


###############################################################################
################                MAIN TESTS
###############################################################################

# Want to test both sparse and well occupied maps
# Should also test square vs circle vs linear


def test_invalid_inputs():
    # >2D dimensional data
    with pytest.raises(NotImplementedError):
        d = np.ones((5, 5, 5))
        func(d, sigma=2)
    print("test_invalid_inputs passed")
