""" Tests for spatial occupancy"""
import numpy as np
import pytest
from opexebo.analysis import spatial_occupancy as func

print("=== tests_analysis_spatial_occupancy ===")


def test_circular_arena():
    times = np.arange(5)
    positions = np.array([[0, 0], [1, 0], [0, 1], [-1, 0], [0, -1]]).T - 1
    speeds = np.ones(5)
    kwargs = {
        "arena_shape": "circ",
        "arena_size": 3,
        "bin_width": 1,
        "speed_cutoff": 0.1,
        "limits": (-2, 2.01, -2, 2.01),
    }
    map, coverage, bin_edges = func(times, positions, speeds, **kwargs)


#    import matplotlib.pyplot as plt
#    plt.imshow(map)
#    print(coverage, bin_edges)


def test_linear_arena():
    # TODO!
    pass


def test_invalid_inputs():
    # wrong dimensions to positions
    n = 10

    with pytest.raises(ValueError):
        times = np.arange(n)
        positions = np.ones((3, n))  #!
        speeds = np.ones(n)
        func(times, positions, speeds, arena_size=1)
    with pytest.raises(ValueError):
        times = np.arange(n)
        positions = np.ones((2, n))
        speeds = np.ones((2, n))  #!
        func(times, positions, speeds, arena_size=1)

    # Mismatched pos/speed
    with pytest.raises(ValueError):
        times = np.arange(n)
        positions = np.ones((2, n))
        speeds - np.ones(n + 1)  #!
        func(times, positions, speeds, arena_size=1)

    # No arena size
    with pytest.raises(TypeError):
        times = np.arange(n)
        positions = np.ones((2, n))
        speeds = np.ones(n)
        func(times, positions, speeds)  #!

    # All nan
    # This isn't explicit in the function, but comes as a result of excluding
    # all non-finite values, and then being left with an empty array
    with pytest.raises(ValueError):
        times = np.arange(n)
        positions = np.full((2, n), np.nan)
        speeds = np.full(n, np.nan)
        func(times, positions, speeds, arena_size=1)

    print("test_invalid_inputs passed")


# if __name__ == '__main__':
#    test_circular_arena()
#    test_linear_arena()
#    test_invalid_inputs()
#
