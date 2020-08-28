""" Tests for circular_mask"""
import numpy as np

from opexebo.general import circular_mask as func

print("=== tests_circular_mask ===")


###############################################################################
################                MAIN TESTS
###############################################################################


def test_large_circle():
    axes = [np.linspace(-100, 100, 501) for i in range(2)]
    diameter = 175
    mask, distance_map, angular_map = func(axes, diameter)


# if __name__ == '__main__':
#   test_large_circle()
