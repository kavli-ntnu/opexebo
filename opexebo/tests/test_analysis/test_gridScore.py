""" Tests for grid score"""

import os
os.environ['HOMESHARE'] = r'C:\temp\astropy'
import sys
sys.path.insert(1, os.path.join(os.getcwd(), '..'))

import matplotlib.pyplot as plt
import scipy.io as spio
import numpy as np
import pytest
import inspect

import test_helpers as th
import opexebo
from opexebo.analysis import grid_score as func

print("=== tests_analysis_grid_score ===")



###############################################################################
################                HELPER FUNCTIONS
###############################################################################

def get_grid_score_bnt(data, i):
    score = data['cellsData'][i,0]['epochs'][0,0][0,0]['gridScore'][0,0][0,0]
    return score

def get_acorr_bnt(data, i):
    acorr = data['cellsData'][i,0]['epochs'][0,0][0,0]['aCorr'][0,0]
    return acorr

###############################################################################
################                MAIN TESTS
###############################################################################
    
def test_grid_score_similarity():
    data = spio.loadmat(th.test_data_square)
    print("Data loaded")
    ds = np.arange(th.get_data_size(data))
    vals = np.zeros_like(ds).astype(float)
    for key in ds:
        acorr = get_acorr_bnt(data, key)
        bnt = get_grid_score_bnt(data, key)
        ope, _ = func(acorr, min_orientation=15, search_method="sep")
        if np.isnan(ope):
            ratio = np.abs(bnt/ope-1)
            vals[key] = ratio
    tol = 1e-2
    assert((vals<tol).all())
    print(f"{inspect.stack()[0][3]} passed")
    return True

def test_perfect_grid_cell():
    firing_map = th.generate_2d_map("rect", 1, x=80, y=80, coverage=0.95, fields=th.generate_hexagonal_grid_fields_dict())
#    plt.figure()
#    plt.imshow(firing_map)
    acorr = opexebo.analysis.autocorrelation(firing_map)
#    plt.figure()
#    plt.imshow(acorr)
#    plt.show()
    gs, stats = func(acorr, debug=True, search_method="default")
    assert stats["grid_ellipse_aspect_ratio"] > 0.985
    assert stats["grid_ellipse_aspect_ratio"] < 1.15

#if __name__ == '__main__':
#    test_perfect_grid_cell()