"""Tests for WalkFilter"""
import numpy as np
import pytest

from opexebo.general import walk_filter as func
from opexebo import errors as err

print("=== tests_general_walk_filter ===")

speeds = np.arange(5000)
speed_cutoff = 1257.3
min_out = 1258
remaining = 3742

def test_remove():
    out = func(speeds, speed_cutoff, fmt="remove")
    assert out.size == remaining
    assert np.min(out) == min_out

def test_mask():
    out = func(speeds, speed_cutoff, fmt="mask")
    assert type(out) == np.ma.MaskedArray
    assert np.sum(out.mask) == min_out
    assert np.min(out) == min_out

def test_filter_other_arrays():
    positions = np.linspace(-1000, 1000, speeds.size)
    fspeed, out = func(speeds, speed_cutoff, positions, fmt="remove")
    assert fspeed.shape == out.shape
    