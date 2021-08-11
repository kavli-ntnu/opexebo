"""Tests for RateMap"""
import numpy as np
import pytest

import opexebo
from opexebo.analysis import rate_map as func
from opexebo import errors as err

print("=== tests_analysis_rateMap ===")


###############################################################################
################                MAIN TESTS
###############################################################################


def test_rmap_invalid_inputs():
    # Mismatched dimensions
    # 2d time map but 1d spike pos
    with pytest.raises(err.DimensionMismatchError):
        tmap = np.ones((10, 10))
        time = np.arange(100)
        spikes_x = np.random.rand(100)
        spikes_tracking = np.array((time, spikes_x))
        func(tmap, spikes_tracking, arena_size=1)
    # 1d time map but 2d spike_pos
    with pytest.raises(err.DimensionMismatchError):
        tmap = np.ones(10)
        time = np.arange(100)
        spikes_x = np.random.rand(100)
        spikes_y = np.random.rand(100)
        spikes_tracking = np.array((time, spikes_x, spikes_y))
        func(tmap, spikes_tracking, arena_size=(10, 10))
    # invalid input types
    with pytest.raises(err.ArgumentError):
        tmap = (1, 2, 3, 4)  #! tmap should be an ndarray
        spikes = np.ones((2, 100))
        func(tmap, spikes, arena_size=1)
    with pytest.raises(err.ArgumentError):
        tmap = np.ones(10)
        spikes = ([0.23, 1], [0.5, 1.2], [0.75, -3])  #! spikes should be an ndarray
        func(tmap, spikes, arena_size=1)
    with pytest.raises(TypeError):
        tmap = np.ones((10, 10))
        spikes = np.ones((3, 100))  # t, x, y
        func(tmap, spikes)  #! missing arena_size
    with pytest.raises(ValueError):
        tmap = np.ones((10, 10))
        spikes = np.ones((3, 100))
        func(tmap, spikes, arena_size=1, limits="abc")  # limits should be a tuple
    # mismatched sizes
    with pytest.raises(err.DimensionMismatchError):
        tmap = np.ones((40, 40))
        spikes = np.ones((3, 100))
        func(tmap, spikes, arena_size=80, bin_width=4)  #! 80/4 != 40

    print("test_rmap_invalid_inputs passed")
    return True


def test_1d_ratemap():
    arena_size = 80
    bin_width = 2.5
    _num_bins = opexebo.general.bin_width_to_bin_number(arena_size, bin_width)
    tmap = np.ones(_num_bins)  # equal occupancy in all locations
    n = 5000
    times = np.sort(np.random.rand(n)) * 1200  # 20 minute session
    pos = np.random.rand(n) * arena_size
    spikes_tracking = np.array([times, pos])
    rmap = func(
        tmap,
        spikes_tracking,
        bin_width=bin_width,
        arena_size=arena_size,
    )
    assert rmap.ndim == 1
    assert rmap.shape == tmap.shape


def test_2d_ratemap():
    arena_size = (80, 120)
    bin_number = (16, 24)
    tmap = np.ones(
        bin_number
    ).T  # Want to be [x] 16 by [y] 24, whereas Numpy writes this the opposite, so transpose
    n = 5000
    times = np.sort(np.random.rand(n)) * 1200  # 20 minute session
    pos_x = np.random.rand(n) * arena_size[0]
    pos_y = np.random.rand(n) * arena_size[1]
    spikes_tracking = np.array([times,  pos_x, pos_y])
    rmap = func(
        tmap,
        spikes_tracking,
        bin_number=bin_number,
        arena_size=arena_size,
    )
    assert rmap.ndim == 2
    assert rmap.shape == tmap.shape
