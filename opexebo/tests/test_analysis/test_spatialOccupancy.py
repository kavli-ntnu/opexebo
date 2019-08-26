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



###############################################################################
################                HELPER FUNCTIONS
###############################################################################


def get_time_map_opexebo(data, key, arena_size = 80, bin_width=2.0):
    if type(arena_size)==int:
        lim = (-arena_size/2, arena_size/2, -arena_size/2, arena_size/2)
    elif type(arena_size) in (tuple, list, np.ndarray):
        lim = (-arena_size[0]/2, arena_size[0]/2, -arena_size[1]/2, arena_size[1]/2)

    t, x, y=data['cellsData'][key,0]['epochs'][0,0][0,0][0,0]['pos'].transpose()
    
    position = np.array([x, y])
    position = np.ma.masked_invalid(position)
    speed = np.ones(t.shape)
    spc = 0
    ma = func(t, position, speed, arena_size=arena_size, 
                                           bin_width=bin_width, speed_cutoff=spc, limits=lim)
    return ma


def get_time_map_bnt(data, key):
    time_smoothed = data['cellsData'][key,0]['epochs'][0,0][0,0]['map'][0,0]['time'][0,0]
    time_raw = data['cellsData'][key,0]['epochs'][0,0][0,0]['map'][0,0]['timeRaw'][0,0]
    return time_smoothed, time_raw

def rms(img):
    return np.sqrt(np.nanmean(np.square(img)))

###############################################################################
################                MAIN TESTS
###############################################################################

def test_square_arena():
    data = spio.loadmat(th.test_data_square)
    ds = np.arange(th.get_data_size(data))
    arena_size = 80
    bin_width = 2.0
    rtol = 1e-3
    
    for key in ds:
    
        bnt = get_time_map_bnt(data, key)[1] # raw map only
        ope = get_time_map_opexebo(data, key, arena_size=arena_size, bin_width=bin_width)[0]

        #  Test that the binning and arena size is valid
        assert(bnt.shape == ope.shape)
        
        # Test that the voids in the map due to the animal not visiting are 
        # consistent
        assert(np.isnan(bnt).all() == ope.mask.all())
        
        # Test that the map produces similar absolute times
        assert(np.isclose(np.nanmean(ope), np.nanmean(bnt), rtol = rtol))
        
        # Test that the maximum and minimum values are similar
        assert(np.isclose(np.nanmax(ope), np.nanmax(bnt), rtol = rtol))
        assert(np.isclose(np.nanmin(ope), np.nanmin(bnt), rtol = rtol))
        
    print("test_square_arena() passed")
    return True

def test_rectangular_arena():
    data = spio.loadmat(th.test_data_nonsquare)
    ds = np.arange(th.get_data_size(data))
    arena_size = (200,100)
    bin_width = 2.0
    rtol = 1e-3
    
    for key in ds:
    
        bnt = get_time_map_bnt(data, key)[1] # raw map only
        ope = get_time_map_opexebo(data, key, arena_size=arena_size, bin_width=bin_width)[0]
        
        #  Test that the binning and arena size is valid
        assert(bnt.shape == ope.shape)
        
        # Test that the voids in the map due to the animal not visiting are 
        # consistent
        assert(np.isnan(bnt).all() == ope.mask.all())
        
        # Test that the map produces similar absolute times
        assert(np.isclose(np.nanmean(ope), np.nanmean(bnt), rtol = rtol))
        
        # Test that the maximum and minimum values are similar
        assert(np.isclose(np.nanmax(ope), np.nanmax(bnt), rtol = rtol))
        assert(np.isclose(np.nanmin(ope), np.nanmin(bnt), rtol = rtol))
        
    print("test_rectangular_arena() passed")
    return True

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
#    data = spio.loadmat(th.test_data_square)
#    get_time_map_opexebo(data, 0, arena_size = 80, bin_width=2.0)
#    test_square_arena()
#    test_rectangular_arena()
#    test_circular_arena()
#    test_linear_arena()
#    test_invalid_inputs()
    