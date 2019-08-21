"""Tests for RateMap"""

import os
os.environ['HOMESHARE'] = r'C:\temp\astropy'
import sys
sys.path.insert(1, os.path.join(sys.path[0], '..'))


import scipy.io as spio
import numpy as np
import pytest

import test_helpers as th
from opexebo.analysis import rate_map as func

print("=== tests_analysis_rateMap ===")

###############################################################################
################                HELPER FUNCTIONS
###############################################################################



def get_ratemap_opexebo(data, key, arena_size = 80, bin_width=2.0):
    if type(arena_size)==int:
        lim = (-arena_size/2, arena_size/2, -arena_size/2, arena_size/2)
    elif type(arena_size) in (tuple, list, np.ndarray):
        lim = (-arena_size[0]/2, arena_size[0]/2, -arena_size[1]/2, arena_size[1]/2)

    time_map = data['cellsData'][key,0]['epochs'][0,0][0,0]['map'][0,0]['timeRaw'][0,0]
    spikes = data['cellsData'][key,0]['epochs'][0,0][0,0]['spikes2Pos'][0,0].transpose()
    rmap_raw = func(time_map, spikes, arena_size=arena_size, 
                                           bin_width=bin_width, limits=lim)
    return rmap_raw


def get_ratemap_bnt(data, key):
    rmap_smooth = data['cellsData'][key,0]['epochs'][0,0][0,0]['map'][0,0]['z'][0,0]
    rmap_raw = data['cellsData'][key,0]['epochs'][0,0][0,0]['map'][0,0]['zRaw'][0,0]
    return rmap_smooth, rmap_raw

def rms(img):
    return np.sqrt(np.nanmean(np.square(img)))

###############################################################################
################                MAIN TESTS
###############################################################################
    
def test_rmap_invalid_inputs():
    # Mismatched dimensions 
    with pytest.raises(ValueError):
        tmap = np.ones((10, 10)) #x, y
        spikes = np.ones((2, 100)) # t, x
        func(tmap, spikes, arena_size=1)
    with pytest.raises(ValueError):
        tmap = np.ones(10) #x
        spikes = np.ones((3, 100)) # t, x, y
        func(tmap, spikes, arena_size = 1)
    
    # invalid input types
    with pytest.raises(ValueError):
        tmap = (1,2,3,4) #! tmap should be an ndarray
        spikes = np.ones((2,100))
        func(tmap, spikes, arena_size=1)
    with pytest.raises(ValueError):
        tmap = np.ones(10)
        spikes = ([0.23,1], [0.5, 1.2], [0.75, -3]) #! spikes should be an ndarray
        func(tmap, spikes, arena_size=1)
    with pytest.raises(KeyError):
        tmap = np.ones((10,10))
        spikes = np.ones((3, 100)) # t, x, y
        func(tmap, spikes) #! missing arena_size
    with pytest.raises(ValueError):
        tmap = np.ones((10,10))
        spikes = np.ones((3, 100))
        func(tmap, spikes, arena_size=1, limits="abc") # limits should be a tuble
    
    # mismatched sizes
    with pytest.raises(ValueError):
        tmap = np.ones((40,40))
        spikes = np.ones((3, 100))
        func(tmap, spikes, arena_size=80, bin_width=4) #! 80/4 != 40
    
    print("test_rmap_invalid_inputs passed")
    return True
    
    
def test_rmap_simple():
    data = spio.loadmat(th.test_data_square)
    ds = np.arange(th.get_data_size(data))
    arena_size = 80
    bin_width = 2.0
    rtol = 1e-4
    
    for key in ds:
    
        bnt = get_ratemap_bnt(data, key)[1] # raw map only
        ope = get_ratemap_opexebo(data, key, arena_size=arena_size, bin_width=bin_width)

        #  Test that the binning and arena size is valid
        assert(bnt.shape == ope.shape)
        
        # Test that the voids in the map due to the animal not visiting are 
        # consistent
        assert(np.isnan(bnt).all() == ope.mask.all())
        
        # Test that the map produces similar absolute rates
        assert(np.isclose(np.nanmean(bnt), np.nanmean(ope), rtol = rtol))
        
        # Test that the maximum and minimum values are similar
        assert(np.isclose(np.nanmax(bnt), np.nanmax(ope), rtol = rtol))
        assert(np.isclose(np.nanmin(bnt), np.nanmin(ope), rtol = rtol))
        
    print("test_ratemap_simple passed")
    return True



if __name__ == '__main__':
    test_rmap_invalid_inputs()
    test_rmap_simple()
