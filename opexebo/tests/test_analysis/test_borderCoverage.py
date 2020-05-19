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
from opexebo.general import circular_mask
from opexebo.analysis import border_coverage as func
from opexebo.analysis.border_coverage import _validate_keyword_walls as validate_func
from opexebo.errors import ArgumentError

print("=== tests_analysis_border_coverage ===")
kw_field_map = "field_map"

###############################################################################
################                validate_func
###############################################################################

def test_validate_func_square_correct_inputs():
    # Test correct strings
    # 1-4 characters, should yield the same string in lower case
    # Resulting string will be in alphabetical order and remove duplicates
    shape = "s"
    assert validate_func("TRBL", shape) == 'blrt'
    assert validate_func("tBr", shape) == "brt"
    assert validate_func("b", shape) == "b"
    assert validate_func("bbbb", shape) == "b"
    assert validate_func("bbbTrBLl", shape) == 'blrt'

def test_validate_func_square_incorrect_inputs():
    shape = "s"
    # zero length
    with pytest.raises(ValueError):
        validate_func("", shape)
    # >4 length of unique characters
    with pytest.raises(ValueError):
        validate_func("abcdef", shape)
    # With invalid characters
    with pytest.raises(ValueError):
        validate_func("abcd", shape)
    # angular array
    with pytest.raises(ArgumentError):
        validate_func([(15, 35), (270, 360)], shape)

###############################################################################
################                Square arena
###############################################################################


def test_perfect_fields_square():
    '''Perfect field: 100% coverage of wall, zero distance, everywhere else zero'''
    shape = "s"
    perfect_left_field = np.zeros((40,40))
    perfect_left_field[:,0] = 1
    left = {kw_field_map:perfect_left_field}
    sw = 1
    assert(func(left, shape, search_width=sw, walls="L") == 1)
    
    right = {kw_field_map: np.fliplr(perfect_left_field)}
    assert(func(right, shape, search_width=sw, walls="R") == 1)
    
    top = {kw_field_map: np.rot90(perfect_left_field)}
    assert(func(top, shape, search_width=sw, walls="T") == 1)
    
    bottom = {kw_field_map: np.flipud(np.rot90(perfect_left_field))}
    assert(func(bottom, shape, search_width=sw, walls="B") == 1)
    
    # The above verify that the function agrees with a field that is there
    # The below verify that the function correctly identifies that a field is *not* there. 
    assert(func(left, shape, search_width=sw, walls="TRB") == 1/40)
    print("test_perfect_fields() passed")
    return True
    
def test_offset_perfect_field_square():
    '''Offset field: 100% coverage of wall, NaNs filling space between wall and field, distance 4, all else zero'''
    field = np.zeros((40,40), dtype=float)
    field[:,3] = 1
    field[:, :3] = np.nan
    field = np.ma.masked_invalid(field)
    fields = {kw_field_map:field}
    shape = "s"
    assert(func(fields, shape, search_width=8, walls="L") == 1)
    print("test_offset_field() passed")
    return True


def test_partial_field_square():
    '''Fields that do not extend over the whole wall'''
    field = np.zeros((40,40))
    field[:10, 0] = 1
    fields = {kw_field_map: field}
    walls = "L"
    assert(func(fields, "s", walls=walls)==0.25)
    field[:20, 0] = 1
    assert(func(fields, "s", walls=walls)==0.5)
    field[:30, 0] = 1
    assert(func(fields, "s", walls=walls)==0.75)
    print("test_partial_field() passed")
    return True

def test_extended_field_square():
    '''Field extending in two dimensions. Should get different answers depending
    on which wall direction is requested'''
    field = np.zeros((40, 40))
    field[:20, :30] = 1
    fields = {kw_field_map:field}
    assert(func(fields, "s", walls="L") == 0.5)
    assert(func(fields, "s", walls="B") == 0.75)
    assert(func(fields, "s", walls="TRBL") == 0.75)
    print("test_extended_field() passed")
    return True

def test_central_field_square():
    '''A large firing field touching no wall'''
    field = np.zeros((40,40))
    field[5:-5, 5:-5] = 1
    fields = {kw_field_map:field}
    assert(func(fields, "s", search_width=8, walls="TRBL") == 0)
    # The field is within the search width, BUT the search_wdth is about handling 
    # locations that the animal never visited - the field MUST extend into the 
    # location closest to the wall (that the animal visited) in order to qualify
    print("test_central_field() passed")
    return True

def test_zero_fields_square():
    fields = []
    assert(func(fields, "s")==0)
    print("test_zero_fields() passed")
    return True

###############################################################################
################                Circular arenas
###############################################################################

#def test_perfect_field_circ():
#    plt.close("all")
#    num_bins = 410
#    axes = [np.linspace(-(num_bins-1), num_bins-1, num_bins) for i in range(2)]
#    diameter = 500#np.max(axes[0]) - np.min(axes[0])
#    in_field, _, _ = circular_mask(axes, diameter)
#    fields = {kw_field_map: in_field}
#    cov = func(fields, "circ", walls="tlbr", debug=True)
#    
#    fig, ax = plt.subplots(1, 2)
#    ax[0].imshow(in_field)
#    print(cov)
    


if __name__ == '__main__':
    test_validate_func_square_correct_inputs()
    test_validate_func_square_incorrect_inputs()
    test_perfect_fields_square()
    test_offset_perfect_field_square()
    test_partial_field_square()
    test_extended_field_square()
    test_central_field_square()
    test_zero_fields_square()
#    test_perfect_field_circ()