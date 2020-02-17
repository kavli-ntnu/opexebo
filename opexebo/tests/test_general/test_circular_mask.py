""" Tests for circular_mask"""
import os
os.environ['HOMESHARE'] = r'C:\temp\astropy'
import sys
sys.path.insert(1, os.path.join(sys.path[0], '..'))

import numpy as np
import pytest
import matplotlib.pyplot as plt

from opexebo.general import circular_mask as func

print("=== tests_circular_mask ===")




###############################################################################
################                MAIN TESTS
###############################################################################

def test_large_circle():
    axes = [np.linspace(-100, 100, 501) for i in range(2)]
    diameter = 175
    mask, distance_map, angular_map = func(axes, diameter)
    plt.close("all")
    fig, ax = plt.subplots(1,3)
    ax[0].imshow(mask)
    ax[0].invert_yaxis()
    ax[1].imshow(distance_map)
    ax[1].invert_yaxis()
    im2 = ax[2].imshow(angular_map)
    ax[2].invert_yaxis()
    fig.colorbar(im2, orientation='horizontal')

#if __name__ == '__main__':
#   test_large_circle()