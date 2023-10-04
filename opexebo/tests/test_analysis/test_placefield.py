"""Tests for PlaceField"""
import numpy as np
import pytest

from opexebo.analysis import place_field as func
import opexebo.errors as err

print("=== tests_analysis_placeField ===")


###############################################################################
################                MAIN TESTS
###############################################################################

# Simple input validation
def test_invalid_input():
    fmap = np.ones((40, 40))
    with pytest.raises(err.ArgumentError):  # invalid input threshold
        init_thresh = 1.2
        func(fmap, init_thresh=init_thresh)
    with pytest.raises(err.ArgumentError):  # invalid input threshold
        init_thresh = 0
        func(fmap, init_thresh=init_thresh)
    with pytest.raises(err.ArgumentError):  # invalid search method - wrong type
        sm = 3
        func(fmap, search_method=sm)
    with pytest.raises(err.ArgumentError):  # invalid search method - unknown
        sm = "3"
        func(fmap, search_method=sm)
    with pytest.raises(
        NotImplementedError
    ):  # invalid search method - known but not implemented
        sm = "not implemented"
        func(fmap, search_method=sm)
    print("test_invalid_input() passed")


def test_unchanging_ratemap():
    """Check that the input arguments are not modified in place by the placefield detection function"""
    # All finite
    fmap = np.ones((40, 40))
    fmap_original = fmap.copy()
    func(fmap)
    assert np.array_equal(fmap, fmap_original)
    # Some non-finite
    fmap = np.ones((40, 40))
    fmap[21, 3] = np.inf
    fmap_original = fmap.copy()
    func(fmap)
    assert np.array_equal(fmap, fmap_original)
    # Masked array
    fmap = np.ones((40, 40))
    fmap[21, 3] = np.inf
    fmap = np.ma.masked_invalid(fmap)
    fmap_original = fmap.copy()
    func(fmap)
    assert np.array_equal(fmap, fmap_original)
    print("test_unchanging_ratemap() passed")
