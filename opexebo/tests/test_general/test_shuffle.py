""" Tests for spatial occupancy"""
from opexebo.general import shuffle as func

import numpy as np
import pytest

print("=== tests_general_shuffle ===")

###############################################################################
################                MAIN TESTS
###############################################################################


def test_output_structure():
    """Verify that the function correctly returns an array where each row is
    a time-shifted copy of the input, with num_shuffle rows"""

    times = np.random.randint(0, 100, size=1000)
    offset_lim = 13
    iterations = 274
    tr = np.arange(111)
    output, _ = func(times, offset_lim, iterations, tracking_range=tr)

    # output.shape = rows, cols
    assert output.shape[0] == iterations
    assert output.shape[1] == times.size
    print("test_output_structure() passed")


def WIP_test_consistent_offset():
    '''All the extra faff here is to deal with the fact that since times does
    not necessarily fill the entire tracking range. Information can be "lost"
    to np.diff because it's not circular (i.e. it doesn't compare last to first)
    Therefore, we manually add this value in at the end "miss"'''
    tr = np.arange(173)
    times = np.random.randint(0, 100, size=10)
    times = np.sort(times)
    offset_lim = 13
    iterations = 2

    output, inc = func(times, offset_lim, iterations)
    miss = (np.min(times) - np.min(tr)) + (np.max(tr) - np.max(times))
    inc = np.atleast_2d(inc + miss).transpose()
    output = np.hstack((output, inc))
    times = np.append(times, miss)

    true_d = np.diff(times)
    # true_d = np.sort(true_d).astype(int)
    d = np.diff(output, axis=1)

    # The order of each row of d **should** be preserved circularly, but we can't
    # just compare like-for-like
    # The simplest, but not best, way of comparison is to sort both
    print(times)
    print(true_d)
    for r in range(iterations):
        row = d[r].astype(int)
        print(row)
    print("test_consistent_offset() passed")


def test_stupid_values():
    times = np.random.randint(0, 100, size=1000)
    offset_lim = 13
    iterations = 274
    tr = np.arange(111)
    with pytest.raises(ValueError):
        stupid_times = np.random.randint(0, 100, size=(100, 100))  # wrong shape
        func(stupid_times, offset_lim, iterations, tracking_range=tr)
    with pytest.raises(ValueError):
        stupid_times = times.copy().astype(float)
        stupid_times[813] = np.nan  # Contains NaN
        func(stupid_times, offset_lim, iterations, tracking_range=tr)
    with pytest.raises(ValueError):
        stupid_offset = 120
        func(times, stupid_offset, iterations, tracking_range=tr)
    print("test_stupid_values() passed")


# if __name__ =='__main__':
#    test_output_structure()
#    test_consistent_offset()
#    test_stupid_values()
