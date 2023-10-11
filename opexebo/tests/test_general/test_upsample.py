""" Tests for upsampling"""
import numpy as np
import pytest

from opexebo.general import upsample as func

print("=== tests_upsampling ===")


###############################################################################
################                MAIN TESTS
###############################################################################


def test_invalid_inputs():
    # Invalid array format
    with pytest.raises(NotImplementedError):
        upscale = 2
        func(3, upscale)
        func("abc", upscale)
        func({2: 2}, upscale)

    # Invalid integerr upscaling
    # Force integer upsclaing with masked array inputs
    with pytest.raises(NotImplementedError):
        masked_array = np.ma.ones((4, 6))
        high_dim_ma = np.ma.ones((4, 6, 7))
        fractional_upscale = 2.5
        shrinking_upscale = 0
        valid_upscale = 2
        func(masked_array, fractional_upscale)
        func(masked_array, shrinking_upscale)
        func(high_dim_ma, valid_upscale)

    # Invalid fractional upscaling
    # force fractional upscaling with floating point upscale
    with pytest.raises(ValueError):
        ndarray = np.ones((4, 6))
        zero_upscale = float(0)
        neg_upscale = -0.5
        func(ndarray, zero_upscale)
        func(ndarray, neg_upscale)

    print("test_invalid_inputs passed")


def test_integer_scaling_2d():
    upscale = int(2)
    ndarray = np.random.rand(50, 50)
    maarray = ndarray.copy()
    maarray[25:28, 10:16] = np.nan
    maarray = np.ma.masked_invalid(maarray)

    # Test behavior on masked array
    new_array = func(maarray, upscale)
    assert np.array_equal(np.array(maarray.shape) * upscale, np.array(new_array.shape))
    assert np.sum(maarray.mask) * (upscale ** 2) == np.sum(new_array.mask)

    # Test behavior on ndarray
    new_array = func(ndarray, upscale)
    assert np.array_equal(np.array(ndarray.shape) * upscale, np.array(new_array.shape))
    print("test_integer_scaling_2d passed")


def test_fractional_scaling_2d():
    upscale = 1.45
    ndarray = np.random.rand(50, 75)
    new_array = func(ndarray, upscale)
    assert np.array_equal(np.round(np.array(ndarray.shape) * upscale), new_array.shape)
    print("test_fractional_scaling_2d passed")


# if __name__ == '__main__':
#    test_invalid_inputs()
#    test_integer_scaling_2d()
#    test_fractional_scaling_2d()
