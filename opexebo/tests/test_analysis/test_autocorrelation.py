""" Tests for autocorrelogram"""
import numpy as np

import opexebo.tests as th
from opexebo.analysis import autocorrelation as func

print("=== tests_analysis_autocorrelation ===")


###############################################################################
################                MAIN TESTS
###############################################################################


def test_random_input():
    firing_map = np.random.rand(80, 80)
    acorr = func(firing_map)
    return acorr


def test_perfect_artificial_grid():
    firing_map = th.generate_2d_map(
        "rect",
        1,
        x=80,
        y=80,
        coverage=0.95,
        fields=th.generate_hexagonal_grid_fields_dict(),
    )
    acorr = func(firing_map)
    return acorr
