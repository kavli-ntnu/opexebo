""" Tests for spatial occupancy"""

import os
import sys
sys.path.insert(1, os.path.join(os.getcwd(), '..'))
os.environ['HOMESHARE'] = r'C:\temp\astropy'
print(sys.path)

import scipy.io as spio
import numpy as np
import pytest

import test_helpers as th
from opexebo.analysis import spatial_occupancy as func

print("=== tests_analysis_spatial_occupancy ===")



def test_circular_arena():
    # TODO!
    pass

def test_linear_arena():
    # TODO!
    pass

def test_invalid_inputs():
    # wrong dimensions to positions
    n = 10

    with pytest.raises(ValueError):
        times = np.arange(n)
        positions = np.ones((3, n)) #!
        speeds = np.ones(n)
        func(times, positions, speeds, arena_size = 1)
    with pytest.raises(ValueError):
        times = np.arange(n)
        positions = np.ones((2, n))
        speeds = np.ones((2, n)) #!
        func(times, positions, speeds, arena_size = 1)
    
    # Mismatched pos/speed
    with pytest.raises(ValueError):        
        times = np.arange(n)
        positions = np.ones((2, n))
        speeds - np.ones(n+1) #!
        func(times, positions, speeds, arena_size = 1)
    
    # No arena size
    with pytest.raises(KeyError):       
        times = np.arange(n)
        positions = np.ones((2, n))
        speeds = np.ones(n)
        func(times, positions, speeds) #!
        
    # All nan
    # This isn't explicit in the function, but comes as a result of excluding
    # all non-finite values, and then being left with an empty array
    with pytest.raises(ValueError):       
        times = np.arange(n)
        positions = np.full((2,n), np.nan)
        speeds = np.full(n, np.nan)
        func(times, positions, speeds, arena_size = 1)
    
    print("test_invalid_inputs passed")
        
        

#if __name__ == '__main__':
#    test_circular_arena()
#    test_linear_arena()
#    test_invalid_inputs()
#    