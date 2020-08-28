""" Tests for spatialCrossCorrelation"""
import numpy as np
import pytest

from opexebo.general import spatial_cross_correlation as func

print("=== tests_general_spatial_cross_correlation ===")


###############################################################################
################                HELPER FUNCTIONS
###############################################################################


###############################################################################
################                MAIN TESTS
###############################################################################


def test_invalid_inputs():
    with pytest.raises(ValueError):  # 1d array, mismatched sizes
        arr0 = np.zeros(5)
        arr1 = np.zeros(6)
        func(arr0, arr1)
    with pytest.raises(ValueError):  # 2d array, mismatched sizes
        arr0 = np.zeros((3, 3))
        arr1 = np.zeros((3, 4))
        func(arr0, arr1)
    with pytest.raises(ValueError):  # Random arguments
        arr0 = np.zeros(5)
        arr1 = "b"
        func(arr0, arr1)
    print("test_invalid_inputs passed")
    return True


def test_trivial_inputs_1d():
    num_bins = 10
    """Correlation of all-zeros should be NaN"""
    arr0 = np.zeros(num_bins)
    arr1 = np.zeros(num_bins)
    p = func(arr0, arr1)
    assert np.isnan(p[0])

    """Correlation of equal np.arange"""
    arr0 = np.arange(num_bins)
    arr1 = np.arange(num_bins)
    p = func(arr0, arr1)
    assert np.isclose(p[0], 1)

    """Correlation of equal np.arange"""
    arr0 = np.arange(num_bins)
    arr1 = np.flip(np.arange(num_bins))
    p = func(arr0, arr1)
    assert np.isclose(p[0], -1)

    """Handling occasionally NaNs"""
    arr0 = np.arange(num_bins).astype(float)
    arr1 = np.arange(num_bins).astype(float)
    arr0[2] = np.nan
    arr1[5] = np.nan
    p = func(arr0, arr1)
    assert np.isclose(p[0], 1)
    print("test_trivial_inputs_1d passed")


def test_trivial_inputs_2d():
    num_bins = (10, 11)
    """Correlation of all-zeros should be NaN"""
    arr0 = np.zeros(num_bins)
    arr1 = np.zeros(num_bins)
    p = func(arr0, arr1)
    assert np.isnan(p[0])
    assert np.isnan(p[1]).all()

    """correlation of equal arange"""
    arr0 = np.arange(num_bins[0] * num_bins[1]).reshape(*num_bins)
    arr1 = np.arange(num_bins[0] * num_bins[1]).reshape(*num_bins)
    p = func(arr0, arr1, row_major=True)
    assert np.isclose(p[0], 1)
    assert np.isclose(p[1], 1).all()
    p = func(arr0, arr1, row_major=False)
    assert np.isclose(p[1], 1).all()

    """correlation of opposite arange"""
    arr0 = np.arange(num_bins[0] * num_bins[1]).reshape(*num_bins)
    arr1 = np.flip(np.arange(num_bins[0] * num_bins[1]).reshape(*num_bins))
    p = func(arr0, arr1, row_major=True)
    assert np.isclose(p[0], -1)
    assert np.isclose(p[1], -1).all()
    p = func(arr0, arr1, row_major=False)
    assert np.isclose(p[1], -1).all()

    """Handling occasional NaNs"""
    arr0 = np.arange(num_bins[0] * num_bins[1]).reshape(*num_bins).astype(float)
    arr1 = np.arange(num_bins[0] * num_bins[1]).reshape(*num_bins).astype(float)
    arr0[2, 3] = np.nan
    arr1[4, 8] = np.nan
    p = func(arr0, arr1, row_major=True)
    assert np.isclose(p[0], 1)
    assert np.isclose(p[1], 1).all()
    assert np.sum(np.isnan(p[1])) == 0
    p = func(arr0, arr1, row_major=False)
    assert np.isclose(p[1], 1).all()
    assert np.sum(np.isnan(p[1])) == 0
    print("test_trivial_inputs_2d passed")


# if __name__ == "__main__":
#    test_trivial_inputs_1d()
#    test_trivial_inputs_2d()
