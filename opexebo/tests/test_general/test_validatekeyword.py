""" Tests for keyword validation"""
import os
os.environ['HOMESHARE'] = r'C:\temp\astropy'
import sys
sys.path.insert(1, os.path.join(sys.path[0], '..'))

import numpy as np
import pytest

from opexebo.general import validatekeyword__arena_size as func
import test_helpers as th

print("=== tests_general_validatekeyword ===")




###############################################################################
################                MAIN TESTS
###############################################################################



def test_invalid_inputs():
    # >2D dimensional data
    with pytest.raises(ValueError): # Negative length, 1d     
        kwv = -1
        dim = 1
        func(kwv, dim)
    with pytest.raises(ValueError): # Negative length, 2d    
        kwv = -1
        dim = 2
        func(kwv, dim)
    with pytest.raises(ValueError): # zero length        
        kwv = 0
        dim = 1
        func(kwv, dim)
    with pytest.raises(NotImplementedError): # Invalid # dimensions   
        kwv = 80
        dim = 3
        func(kwv, dim)
    with pytest.raises(NotImplementedError): # Invalid # dimensions   
        kwv = 80
        dim = 0
        func(kwv, dim)
    with pytest.raises(IndexError): # Invalid # dimensions   
        kwv = (80, 80)
        dim = 1
        func(kwv, dim)
    print("test_invalid_inputs passed")
    return True


def test_1d_int_valid():
    kwv = 80
    dim = 1
    ars, is_2d = func(kwv, dim)
    assert(ars == 80.0)
    assert(not is_2d)
    print("test_1d_int_valid passed")
    return True

def test_1d_str_valid():
    kwv = "80"
    dim = 1
    ars, is_2d = func(kwv, dim)
    assert(ars == 80.0)
    assert(not is_2d)
    print("test_1d_str_valid passed")
    return True

def test_2d_square_valid():
    kwv = 80
    dim = 2
    ars, is_2d = func(kwv, dim)
    assert(ars == 80.0)
    assert(is_2d)
    print("test_2d_square_valid passed")
    return True

def test_2d_rectangle_valid():
    kwv = [80, 120]
    dim = 2
    ars, is_2d = func(kwv, dim)
    assert(np.array_equal(ars, np.array((80, 120))))
    assert(is_2d)
    print("test_2d_rectangle_valid passed")
    return True