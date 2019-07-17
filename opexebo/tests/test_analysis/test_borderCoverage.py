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
from opexebo.analysis import border_coverage as func

print("=== tests_analysis_border_coverage ===")

###############################################################################
################                MAIN TESTS
###############################################################################
    

def test_perfect_fields():
    '''Perfect field: 100% coverage of wall, zero distance, everywhere else zero'''
    perfect_left_field = np.zeros((40,40))
    perfect_left_field[:,0] = 1
    left = {"map":perfect_left_field}
    sw = 1
    walls= "L"
    assert(func(left, search_width=sw, walls=walls) == 1)
    right = {'map': np.fliplr(perfect_left_field)}
    walls="R"
    assert(func(right, search_width=sw, walls=walls) == 1)
    top = {"map": np.rot90(perfect_left_field)}
    walls="T"
    assert(func(top, search_width=sw, walls=walls) == 1)
    bottom = {"map": np.flipud(np.rot90(perfect_left_field))}
    walls="B"
    assert(func(bottom, search_width=sw, walls=walls) == 1)
    # The above verify that the function agrees with a field that is there
    # The below verify that the function correctly identifies that a field is *not* there. 
    assert(func(left, search_width=sw, walls="TRB") == 1/40)
    print("test_perfect_fields() passed")
    return True
    
def test_offset_perfect_field():
    '''Offset field: 100% coverage of wall, NaNs filling space between wall and field, distance 4, all else zero'''
    field = np.zeros((40,40), dtype=float)
    field[:,3] = 1
    field[:, :3] = np.nan
    field = np.ma.masked_invalid(field)
    print(field)
    fields = {"map":field}
    sw = 8
    walls= "L"
    assert(func(fields, search_width=sw, walls=walls) == 1)
    print("test_offset_field() passed")
    return True

def test_partial_field():
    '''Fields that do not extend over the whole wall'''
    field = np.zeros((40,40))
    field[:10, 0] = 1
    fields = {"map": field}
    walls = "L"
    assert(func(fields, walls=walls)==0.25)
    field[:20, 0] = 1
    assert(func(fields, walls=walls)==0.5)
    field[:30, 0] = 1
    assert(func(fields, walls=walls)==0.75)
    print("test_partial_field() passed")
    return True

def test_extended_field():
    '''Field extending in two dimensions. Should get different answers depending
    on which wall direction is requested'''
    field = np.zeros((40, 40))
    field[:20, :30] = 1
    fields = {"map":field}
    assert(func(fields, walls="L") == 0.5)
    assert(func(fields, walls="B") == 0.75)
    assert(func(fields, walls="TRBL") == 0.75)
    print("test_extended_field() passed")
    return True

def test_central_field():
    '''A large firing field touching no wall'''
    field = np.zeros((40,40))
    field[5:-5, 5:-5] = 1
    fields = {"map":field}
    assert(func(fields, search_width=8, walls="TRBL") == 0)
    # The field is within the search width, BUT the search_wdth is about handling 
    # locations that the animal never visited - the field MUST extend into the 
    # location closest to the wall (that the animal visited) in order to qualify
    print("test_central_field() passed")
    return True

def test_zero_fields():
    fields = []
    assert(func(fields)==0)
    print("test_zero_fields() passed")
    return True


if __name__ == '__main__':
    test_perfect_fields()
    test_offset_perfect_field()
    test_partial_field()
    test_extended_field()
    test_central_field()
    test_zero_fields()