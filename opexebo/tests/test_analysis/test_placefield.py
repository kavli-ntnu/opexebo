"""Tests for PlaceField"""

import os
os.environ['HOMESHARE'] = r'C:\temp\astropy'
import sys
sys.path.insert(0, '..')


import scipy.io as spio
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import pytest

import test_helpers as th
from opexebo.analysis import place_field as func

print("=== tests_analysis_placeField ===")

###############################################################################
################                HELPER FUNCTIONS
###############################################################################

def get_fields_bnt(data, key):
    fmap = data['cellsData'][key,0]['epochs'][0,0][0,0]['fieldsMap'][0,0]
    try:
        fields = data['cellsData'][key,0]['epochs'][0,0][0,0]['fields'][0,0][0,:]
    except IndexError:
        fields = np.array([])
    return fmap, fields

def rms(img):
    return np.sqrt(np.nanmean(np.square(img)))

def show_map(data, key):
    rmap = th. get_ratemap_bnt(data, key)[0]
    fmap_bnt = get_fields_bnt(data, key)[0]
    fmap_ope = func(rmap)[1]
    plt.figure(figsize=(18,6))
    plt.subplot(131)
    plt.title("Rate map")
    plt.imshow(rmap)
    plt.colorbar()
    plt.subplot(132)
    plt.title("BNT Field Map")
    plt.imshow(fmap_bnt)
    plt.subplot(133)
    plt.imshow(fmap_ope)
    plt.title("Opexebo Field Map")
    plt.show()

###############################################################################
################                MAIN TESTS
###############################################################################

# Simple input validation
def test_invalid_input():
    fmap = np.ones((40,40))
    with pytest.raises(ValueError): # invalid input threshold
        init_thresh = 1.2
        func(fmap, init_thresh = init_thresh)
    with pytest.raises(ValueError): # invalid input threshold
        init_thresh = 0
        func(fmap, init_thresh = init_thresh)
    with pytest.raises(ValueError): # invalid search method - wrong type
        sm = 3
        func(fmap, search_method=sm)
    with pytest.raises(ValueError): # invalid search method - unknown
        sm = "3"
        func(fmap, search_method=sm)
    with pytest.raises(NotImplementedError): # invalid search method - known but not implemented
        sm = "not implemented"
        func(fmap, search_method=sm)
    print("test_invalid_input() passed")
    return True

def test_unchanging_ratemap():
    '''Check that the input arguments are not modified in place by the placefield detection function'''
    # All finite
    fmap = np.ones((40,40))
    fmap_original = fmap.copy()
    func(fmap)
    assert(np.array_equal(fmap, fmap_original))
    # Some non-finite
    fmap = np.ones((40,40))
    fmap[21,3] = np.inf
    fmap_original = fmap.copy()
    func(fmap)
    assert(np.array_equal(fmap, fmap_original))
    #Masked array
    fmap = np.ones((40,40))
    fmap[21,3] = np.inf
    fmap = np.ma.masked_invalid(fmap)
    fmap_original = fmap.copy()
    func(fmap)
    assert(np.array_equal(fmap, fmap_original))
    print("test_unchanging_ratemap() passed")
    return True
    
'''
Due to changes made in the field detection algorithm, it is highly likely that:
    * A different number of fields will be detected (e.g. due to merging/splitting in marginal areas)
    * The exact area covered will be slightly different around the edges

Therefore, it may be better, instead of comparing field-wise for similarity, to
instead compare the combined infield/outfield area for similarity
'''
        
    

def test_messy_data():
    # This uses a miscellanious bunch of cells from Dave's work in deep layers
    # That meanst that they're not very good examples, but there are a *lot* 
    # of them
    # Unfortunately, I think the combination of messy and the changes made to
    # the Python version mean that they're never going to agree very well. 
    data = spio.loadmat(th.test_data_big)
    ds = np.arange(th.get_data_size(data))
    sm = "sep"
    areas = np.zeros((ds.size, 3))
    numbers = np.zeros((ds.size, 3))
    np.seterr(all="ignore")
    for key in tqdm(ds):
        rmap = th. get_ratemap_bnt(data, key)[0]
        fmap_bnt, fields_bnt = get_fields_bnt(data, key)
    
        fields, fmap = func(rmap, search_method=sm)
        
        # Compare total area of all fields
        area_bnt = np.count_nonzero(fmap_bnt)
        area_ope = np.count_nonzero(fmap)
        areas[key] = (area_bnt, area_ope, np.divide(area_bnt, area_ope))
        
        
        # Compare total number of fields
        numbers[key] = (len(fields_bnt), len(fields), len(fields_bnt)-len(fields))
    return areas, numbers
        
    
    
#areas, numbers = test_messy_data()
#plt.figure()
#plt.plot(areas[:,2])
#plt.show()
