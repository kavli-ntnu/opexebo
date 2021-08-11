""" Tests for spatial occupancy"""
from opexebo.general import shuffle as func
from opexebo import errors

import numpy as np
import pytest

print("=== tests_general_shuffle ===")

###############################################################################
################                MAIN TESTS
###############################################################################


t_start = 1
t_stop = 13
offset_lim = 2.3
iterations = 1000
times = np.arange(t_start + 0.5, t_stop - 0.5)


def test_increment_creation():
    """
    Verify that the increments are correctly created between (t_start + offset_lim) and (t_stop - offset_lim)
    """

    out, inc = func(times, offset_lim, iterations, t_start, t_stop)
    assert inc.size == iterations
    assert min(inc) >= t_start + offset_lim
    assert (
        min(inc) <= t_start + offset_lim + 0.5
    )  # this is a random test, so it _could_ fail...
    assert max(inc) <= t_stop - offset_lim
    assert max(inc) >= t_stop - offset_lim - 0.5
    return


def test_output_creation():
    """
    verify that the output array is the correct shape and size, can be indexed correctly
    """

    out, inc = func(times, offset_lim, iterations, t_start, t_stop)
    assert out.shape == (iterations, times.size)
    assert out[0].shape == times.shape
    return


def test_output_logic():
    """
    Verify that the output is a shuffled copy of the input
    """

    out, inc = func(times, offset_lim, iterations, t_start, t_stop)

    assert np.min(out) >= t_start
    assert np.max(out) <= t_stop
    for row in out:
        assert np.array_equal(row, np.array(sorted(row)))
    return


def test_edge_cases():
    """
    Test some obvious edge cases
    """
    # Basic arguments
    t_start = 1
    t_stop = 13
    offset_lim = 2.3
    iterations = 1000

    times = np.arange(t_start + 0.5, t_stop - 0.5)

    # Few spikes compared to large time range
    out, _ = func(times, 1e3, iterations, t_start, 1e4)
    assert np.array_equal(np.diff(times), np.diff(out[0]))

    # One spike
    spk1 = np.array([5.3])
    out, _ = func(spk1, offset_lim, iterations, t_start, t_stop)
    assert out.shape == (iterations, spk1.size)


"""
Verify that the defensive logic covers as many stupid arguments as I can think of
"""


invalid_times = [
    np.ones((2, 2)),  # 2d data
    3,  # non-array data
    np.full(4, np.nan),  # non-finite data
    "a",
]
invalid_offset = [
    -1,  # negative
    0,
    None,
    8,  # too high, i.e. greater than half the difference
    np.nan,  # nonfinite
    "a",
]
invalid_iterations = [
    -1,  # negative
    0,
    None,
    np.nan,  # nonfinite
    2.5,  # non-integer
    "a",  # wrong type
]
invalid_t_start = [
    np.nan,
    np.inf,
    1.6,  # larger than min(times)
    "a",
]
invalid_t_stop = [
    np.nan,
    np.inf,
    10,  # smaller than max(times)
    -1,  # smaller than t_start
    "a",
]


@pytest.mark.parametrize("invalid_data", invalid_times, ids=str)
def test_invalid_times(invalid_data):
    with pytest.raises(errors.ArgumentError):
        func(invalid_data, offset_lim, iterations, t_start, t_stop)


@pytest.mark.parametrize("invalid_data", invalid_offset, ids=str)
def test_invalid_offsets(invalid_data):
    with pytest.raises(errors.ArgumentError):
        func(times, invalid_data, iterations, t_start, t_stop)


@pytest.mark.parametrize("invalid_data", invalid_iterations, ids=str)
def test_invalid_iterations(invalid_data):
    with pytest.raises(errors.ArgumentError):
        func(times, offset_lim, invalid_data, t_start, t_stop)


@pytest.mark.parametrize("invalid_data", invalid_t_start, ids=str)
def test_invalid_t_start(invalid_data):
    with pytest.raises(errors.ArgumentError):
        func(times, offset_lim, iterations, invalid_data, t_stop)


@pytest.mark.parametrize("invalid_data", invalid_t_stop, ids=str)
def test_invalid_t_stop(invalid_data):
    with pytest.raises(errors.ArgumentError):
        func(times, offset_lim, iterations, t_start, invalid_data)
