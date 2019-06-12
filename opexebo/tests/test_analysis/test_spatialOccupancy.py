""" Tests for spatial occupancy"""

import opexebo
from opexebo.analysis import spatialoccupancy as func

import scipy.io as spio
import numpy as np
import pytest

print("=== tests_analysis_spatialOccupancy ===")

###############################################################################
################                HELPER FUNCTIONS
###############################################################################


def get_time_map_opexebo(data, key, arena_size = 80, bin_width=2.0):
    if type(arena_size)==int:
        lim = (-arena_size/2, arena_size/2, -arena_size/2, arena_size/2)
    elif type(arena_size) in (tuple, list, np.ndarray):
        lim = (-arena_size[0]/2, arena_size[0]/2, -arena_size[1]/2, arena_size[1]/2)

    positions=data['cellsData'][key,0]['epochs'][0,0][0,0][0,0]['pos'].transpose()
    positions = np.ma.masked_where(np.isnan(positions), positions)
    speeds = np.ones((2, positions.shape[1]))
    spc = 0
    ma = opexebo.analysis.spatialoccupancy(positions, speeds, arena_size=arena_size, 
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
    data = spio.loadmat(r"N:\simoba\opexebo_working_area\test_data\generic\simple_square_input_vars.mat")
    arena_size = 80
    bin_width = 2.0
    
    for key in np.arange(20):
    
        bnt = get_time_map_bnt(data, key)[1] # raw map only
        ope = get_time_map_opexebo(data, key, arena_size=arena_size, bin_width=bin_width)

        #  Test that the binning and arena size is valid
        assert(bnt.shape == ope.shape)
        
        # Test that the voids in the map due to the animal not visiting are 
        # consistent
        assert(np.isnan(bnt).all() == ope.mask.all())
        
        # Test that the map produces similar absolute times
        assert(np.isclose(np.nanmean(bnt), np.nanmean(ope), rtol = 1e-5))
        
        # Test that the maximum and minimum values are similar
        assert(np.isclose(np.nanmax(bnt), np.nanmax(ope), rtol = 1e-5))
        assert(np.isclose(np.nanmin(bnt), np.nanmin(ope), rtol = 1e-5))
        
    print("test_square_arena() passed")
    return True

def test_rectangular_arena():
    data = spio.loadmat(r"N:\simoba\opexebo_working_area\test_data\non-square\input_file_vars.mat")
    arena_size = (200,100)
    bin_width = 2.0
    
    for key in np.arange(5):
    
        bnt = get_time_map_bnt(data, key)[1] # raw map only
        ope = get_time_map_opexebo(data, key, arena_size=arena_size, bin_width=bin_width)
        
        #  Test that the binning and arena size is valid
        assert(bnt.shape == ope.shape)
        
        # Test that the voids in the map due to the animal not visiting are 
        # consistent
        assert(np.isnan(bnt).all() == ope.mask.all())
        
        # Test that the map produces similar absolute times
        assert(np.isclose(np.nanmean(bnt), np.nanmean(ope), rtol = 1e-5))
        
        # Test that the maximum and minimum values are similar
        assert(np.isclose(np.nanmax(bnt), np.nanmax(ope), rtol = 1e-5))
        assert(np.isclose(np.nanmin(bnt), np.nanmin(ope), rtol = 1e-5))
        
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
    with pytest.raises(ValueError):
        positions = np.ones(10) #!
        speeds = np.ones((2,10))
        func(positions, speeds, arena_size = 1)
    with pytest.raises(ValueError):
        positions = np.ones((4, 10)) #!
        speeds = np.ones((2, 10))
        func(positions, speeds, arena_size = 1)
    with pytest.raises(ValueError):
        positions = np.ones((3, 10))
        speeds = np.ones((1, 10)) #!
        func(positions, speeds, arena_size = 1)
    
    # Mismatched pos/speed
    with pytest.raises(ValueError):
        positions = np.ones((3,10))
        speeds - np.ones((2,11)) #!
        func(positions, speeds, arena_size = 1)
    
    # No arena size
    with pytest.raises(KeyError):
        positions = np.ones((3, 10))
        speeds = np.ones((2, 10))
        func(positions, speeds) #!
        
    # All nan
    # This isn't explicit in the function, but comes as a result of excluding
    # all non-finite values, and then being left with an empty array
    with pytest.raises(ValueError):
        positions = np.full((3,10), np.nan)
        speeds = np.full((2,10), np.nan)
        func(positions, speeds, arena_size = 1)
    
    print("test_invalid_inputs passed")
        
        

if __name__ == '__main__':
    
    #test_square_arena()
    #test_rectangular_arena()
    #test_circular_arena()
    #test_linear_arena()
    test_invalid_inputs()