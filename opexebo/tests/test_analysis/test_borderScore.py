""" Tests for spatial occupancy"""

import os
os.environ['HOMESHARE'] = r'C:\temp\astropy'
import sys
sys.path.insert(0, '..')

import scipy.io as spio
import numpy as np
import matplotlib.pyplot as plt
import pytest

import test_helpers as th
from opexebo.analysis import border_score as func

print("=== tests_analysis_border_score ===")



###############################################################################
################                HELPER FUNCTIONS
###############################################################################


def get_fields_opexebo(data, key):
    # Rely on cellsData and allStatistics being in the same order
    # get the identifying information from allStats
    path = data['allStatistics'][0,key]['Path'][0]
    basename = data['allStatistics'][0,key]['Basename'][0]
    tetrode = data['allStatistics'][0,key]['Tetrode'][0,0]
    cell = data['allStatistics'][0,key]['Cell'][0,0]
    
    fi = []
    for j in range(data['allFields'].size): # start no sooner than i - this assumes that a ratemap NEVER has zero fields
        pathj = data['allFields'][0,j]['Path'][0]
        basenamej = data['allFields'][0,j]['Basename'][0]
        tetrodej = data['allFields'][0,j]['Tetrode'][0,0]
        cellj = data['allFields'][0,j]['Cell'][0,0]
        if path == pathj and basename == basenamej and tetrode == tetrodej and cell == cellj:
            fi.append(j)
    return fi

def get_border_score_opexebo(data, key):
    rmap = 0
    fmap = 0
    fields = 0


def get_border_score_bnt(data, key):
    return data['cellsData'][key,0]['epochs'][0,0][0,0]['borderScore'][0,0][0,0]
    



###############################################################################
################                MAIN TESTS
###############################################################################
    


if __name__ == '__main__':
    #data = spio.loadmat(th.test_data_square)
    ds = np.arange(th.get_data_size(data))
    key = 14
    fields = get_fields_bnt(data, key)
    print(key, len(fields))
    plt.figure()
    plt.imshow(th.get_ratemap_bnt(data, key)[0])
    plt.colorbar()
    plt.show()