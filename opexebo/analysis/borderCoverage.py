"""Provide function for calculating a Border Score"""

import numpy as np
from scipy.ndimage import distance_transform_edt

def borderCoverage(fields, **kwargs):
    '''
    Calculate border coverage for detected fields.
    
    This function calculates firing map border coverage that is further
    used in calculation of a border score.
    
    
    Parameters
    ----------
    fields      :   dict
        Dictionary of all known firing fields
        The key is the field_id corresponding to the labelled fields_map
        Each firing field is also given as a dictionary
        Example:
            fields = {1:field_one, 2:field_two}
            field_one = {'field_id':1, 'field_size':2.6, ...}
    kwargs
        search_width    :   int
            rate_map and fields_map have masked values, which may occur within the region of border 
            pixels. To mitigate this, we check rows/columns within search_width pixels of the border
            If no value is supplied, default 8
        walls           :   str
            Definition of walls along which the border score is calculated. Provided by
            a string which contains characters that stand for walls:
                      T - top wall (we assume the bird-eye view on the arena)
                      R - right wall
                      B - bottom wall
                      L - left wall
                      Characters are case insensitive. Default value is 'TRBL' meaning that border
                      score is calculated along all walls. Any combination is possible, e.g.
                      'R' to calculate along right wall, 'BL' to calculate along two walls, e.t.c.
    Returns
    -------
    coverage    : float
        Border coverage, ranges from 0 to 1.
    '''
    
    # Extract keyword arguments or set defaults
    default_search_width = 8
    default_walls = 'trbl'
    sw = kwargs.get('search_width', default_search_width)
    walls = kwargs.get('walls', default_walls).lower()
    
    # Check that the wall definition is valid
    _validate_wall_definition(walls)
    
   
    
    # Check coverage of each field in turn
    coverage = 0
    for field in fields:
        fmap = field['field map'] # binary image of field: values are 1 inside field, 0 outside
        if "l" in walls:
            aux_map = fmap[:,:sw]
            _wall_field(aux_map)
        
        
        
        if "r" in walls:
            pass
        if "t" in walls:
            pass
        if "b" in walls:
            pass
    
    
def _wall_field(wfmap):
    '''Evaluate what fraction of the border area is covered by a single field
    
    Border coverage is provided as two values: 
        covered : the sum of the values across all sites immediately adjacent to the border, where the values
                are calculated from the distance of those sites to the firing field, excluding NaN, inf, and masked values
        norm    : the number of non nan, inf, masked values considered in the above sum
    
    The border area is defined by wfmap - this is the subsection of the binary firing map of a 
    single field that lies within search_width of a border. wfmap is must be of size NxM where
        N : arena_size / bin_size
        M : search_width
    and where the 0th column (wfmap[:,0], len()=N) repsents the sites closest to the border
    
    wfmap: has value 1 inside the field and 0 outside the field
    '''
    N = wfmap.shape[0]
    inverted_wfmap = 1-wfmap 
    distance = distance_transform_edt(inverted_wfmap)
    # distance_transform_edt(1-map) is the Python equivalent to (Matlab bwdist(map))
    # Cells in map with value 1 go to value 0
    # Cells in map with value 0 go to the geomretric distance to the nearest value 1 in map
    
    adjacent_sites = distance[:,0]
    
    
    ##### TODO TODO TODO
    # Vadim transformed the matrix wfmap in the if, if, if, if area, such that the first column in wfmap are the bins closest to the wall.
    
    
def _validate_wall_definition(walls):
    '''Parse the walls argument for invalid entry'''
    if type(walls) != str:
        raise ValueError("Wall definition must be given as a string, e.g. 'trbl'. %s is not a valid input." % str(walls))
    elif len(walls) > 4:
        raise ValueError("Wall definition may not exceed 4 characters. String '%s' contains %d characters." % (walls, len(walls)))
    elif len(walls) == 0:
        raise ValueError("Wall definition must contain at least 1 character from the set [t, r, b, l]")
    else:
        for char in walls:
            if char.lower() not in ["t","r","b","l"]:
                raise ValueError("Character %s is not a valid entry in wall definition. Valid characters are [t, r, b, l]" % char)
    
    
    
    