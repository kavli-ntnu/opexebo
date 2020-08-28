"""Tests for accumulate_spatial"""
import numpy as np
import pytest

import opexebo
from opexebo.general import accumulate_spatial as func

print("=== tests_general_accumulate_spatial ===")


def test_invalid_inputs():
    # No `arena_size` keyword
    with pytest.raises(TypeError):
        pos = np.random.rand(100)
        func(pos)
    # Misdefined bins
    with pytest.raises(KeyError):
        pos = np.random.rand(100)
        func(pos, arena_size=1, bin_number=10, bin_width=2.5)
    # Misdefined `limit` keyword
    with pytest.raises(ValueError):
        post = np.random.rand(100)
        func(post, arena_size=1, limits="abc")


def test_1d_input():
    arena_size = 80
    pos = np.random.rand(1000) * arena_size
    bin_width = 2.32
    limits = (np.nanmin(pos), np.nanmax(pos) * 1.0001)
    hist, edges = func(pos, arena_size=arena_size, limits=limits, bin_width=bin_width)
    assert hist.ndim == 1
    assert hist.size == opexebo.general.bin_width_to_bin_number(arena_size, bin_width)
    assert edges.size == hist.size + 1
    assert pos.size == np.sum(hist)


def test_2d_input():
    arena_size = np.array((80, 120))
    pos = (np.random.rand(1000, 2) * arena_size).transpose()

    limits = (0, 80.001, 0, 120.001)
    bin_width = 4.3
    hist, (edge_x, edge_y) = func(
        pos, arena_size=arena_size, limits=limits, bin_width=bin_width
    )
    assert edge_x[0] == limits[0]
    assert hist.ndim == 2
    for i in range(hist.ndim):
        # Note: the array is transposed, so the shape swaps order
        #        print(hist.shape)
        #        print(opexebo.general.bin_width_to_bin_number(arena_size, bin_width))
        #        print(edge_x[0], edge_x[1])
        #        print(np.min(pos[0]), np.max(pos[0]))
        assert (
            hist.shape[i]
            == opexebo.general.bin_width_to_bin_number(arena_size, bin_width)[i - 1]
        )
    assert pos.shape[1] == np.sum(hist)


def test_2d_bin_number():
    arena_size = np.array((80, 120))
    pos = (np.random.rand(1000, 2) * arena_size).transpose()
    limits = (0, 80.001, 0, 120.001)
    bin_number = (8, 12)
    hist, (edge_x, edge_y) = func(
        pos, arena_size=arena_size, limits=limits, bin_number=bin_number
    )
    assert edge_x.size == bin_number[0] + 1
    assert edge_y.size == bin_number[1] + 1
    assert pos.shape[1] == np.sum(hist)

    bin_number = 8
    hist, (edge_x, edge_y) = func(
        pos, arena_size=arena_size, limits=limits, bin_number=bin_number
    )
    assert edge_x.size == edge_y.size == bin_number + 1
    assert pos.shape[1] == np.sum(hist)


def test_2d_bin_edges():
    arena_size = np.array((80, 120))
    pos = (np.random.rand(1000, 2) * arena_size).transpose()
    limits = (0, 80.001, 0, 120.001)
    bin_edges = [np.arange(arena_size[i] + 1) for i in range(2)]
    hist, edges = func(pos, arena_size=arena_size, limits=limits, bin_edges=bin_edges)
    for i in range(2):
        assert np.array_equal(edges[i], bin_edges[i])


if __name__ == "__main__":
    test_invalid_inputs()
    test_2d_input()
    test_2d_bin_number()
    test_2d_bin_edges()
