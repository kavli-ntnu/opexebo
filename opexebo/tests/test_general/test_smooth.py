""" Tests for smoothing"""

import opexebo
from opexebo.general import smooth as func

import scipy.io as spio
import numpy as np
import matplotlib.pyplot as plt
import pytest

print("=== tests_general_smooth ===")


###############################################################################
################                HELPER FUNCTIONS
###############################################################################

def get_sample_size():
    return data['allStatistics'].size

def get_time_map_bnt(data, key):
    time_smoothed = data['cellsData'][key,0]['epochs'][0,0][0,0]['map'][0,0]['time'][0,0]
    time_raw = data['cellsData'][key,0]['epochs'][0,0][0,0]['map'][0,0]['timeRaw'][0,0]
    time_raw = np.ma.masked_invalid(time_raw)
    return time_smoothed, time_raw

def rms(img):
    return np.sqrt(np.nanmean(np.square(img)))

###############################################################################
################                MAIN TESTS
###############################################################################

# Want to test both sparse and well occupied maps
# Should also test square vs circle vs linear

def test_invalid_inputs():
    # >2D dimensional data
    with pytest.raises(NotImplementedError):
        d = np.ones((5,5,5))
        func(d, sigma=2)
    print("test_invalid_inputs passed")
    return True

def test_smoothing_simple():
    z = get_sample_size()
    for key in np.arange(z):
        bnt, tr = get_time_map_bnt(data, key)
        ope = func(tr, sigma=2, mask_fill=0)
        
        abs_error = np.abs(bnt-ope)
        rel_error = abs_error / np.nanmax(bnt)
        assert(np.ma.less_equal(rel_error, 1e-4).all())

def single():
    key = 10
    bnt, tr = get_time_map_bnt(data, key)
    ope = func(tr, sigma=2, mask_fill=0)
    ratio = (bnt-ope)/np.nanmax(bnt)

    plt.figure()
    plt.subplot(1,3,1)
    plt.title("BNT")
    plt.imshow(bnt)
    plt.colorbar()
    plt.subplot(1,3,2)
    plt.title("Opexebo")
    plt.imshow(ope)
    plt.colorbar()
    plt.subplot(1,3,3)
    plt.title("Ratio")
    plt.imshow(ratio)
    plt.colorbar()
    plt.show()
    
if __name__ == '__main__':
    data = spio.loadmat(r'C:\Users\simoba\Documents\_work\Kavli\bntComp\Output_2\auto_input_file_vars.mat')
    #data = spio.loadmat(r"N:\simoba\opexebo_working_area\test_data\generic\simple_square_input_vars.mat")
    test_invalid_inputs()
    test_smoothing_simple()
    single()
    
    