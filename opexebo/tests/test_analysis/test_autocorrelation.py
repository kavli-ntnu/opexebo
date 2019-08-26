""" Tests for autocorrelogram"""

import os
os.environ['HOMESHARE'] = r'C:\temp\astropy'
import sys
sys.path.insert(1, os.path.join(os.getcwd(), '..'))

import inspect
import scipy.io as spio
import numpy as np
import pytest

import test_helpers as th
from opexebo.analysis import autocorrelation as func

print("=== tests_analysis_autocorrelation ===")



###############################################################################
################                HELPER FUNCTIONS
###############################################################################

def get_ratemap_bnt(data, key):
    rmap_smooth = data['cellsData'][key,0]['epochs'][0,0][0,0]['map'][0,0]['z'][0,0]
    return rmap_smooth

def get_acorr_bnt(data, i):
    acorr = data['cellsData'][i,0]['epochs'][0,0][0,0]['aCorr'][0,0]
    return acorr

###############################################################################
################                MAIN TESTS
###############################################################################

def test_acorr_similarity():
    data = spio.loadmat(th.test_data_square)
    ds = np.arange(th.get_data_size(data))
    for key in ds:
        rmap = get_ratemap_bnt(data, key)
        acorr_bnt = get_acorr_bnt(data, key)
        acorr_ope = func(rmap)
        assert(np.isclose(acorr_ope, acorr_bnt, rtol=3e-3).all())
    print(f"{inspect.stack()[0][3]} passed")
    return True



if __name__ == '__main__':
    test_acorr_similarity()
    