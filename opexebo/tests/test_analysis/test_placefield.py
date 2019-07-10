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
    with pytest.raises(SyntaxError): # invalid search method - wrong type
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
    data = spio.loadmat(th.test_data_big)
    ds = np.arange(th.get_data_size(data))
    sm = "sep"
    values = np.zeros(ds.shape)
    for key in tqdm(ds):
        rmap = th. get_ratemap_bnt(data, key)[0]
        fmap_bnt, fields_bnt = get_fields_bnt(data, key)
    
        fields, fmap = func(rmap, search_method=sm)
        
        fmap_bnt[fmap_bnt>1] = 1
        fmap[fmap>1] = 1

        bnt = np.sum(fmap_bnt) / fmap_bnt.size # these are [0,1]
        ope = np.sum(fmap) / fmap.size
        if ope != 0:
            similarity = bnt/ope # this should ideally be 1
            
            values[key] = similarity
    return values
        
    
    

