"""Provide function for calculating a Border Score"""

import numpy as np
from scipy.ndimage import distance_transform_cdt
from opexebo.analysis import border_coverage
import opexebo.defaults as default


def border_score(rate_map, fields_map, fields, **kwargs):
    '''
    Calculate border score for a firing map
    
    STATUS : EXPERIMENTAL
    
    TODO: Update to account for the use of Vadim's placefield code

    Calculates a border score for a firing rate map according to the article 
    "Representation of Geometric Borders in the Entorhinal Cortex" by Solstad 
    et. al. (Science 2008).
    Border score ranges from -1 to +1 with +1 being "a perfect border cell". If
    the firing map contains no firing fields, then the returned score is -1.
    The score reflects not only how close a field is to a border and how big 
    the coverage of this field is, but also it reflects spreadness of a field. 
    The best border score (+1) will be calculated for a thin line (1 px, bin) 
    that lies along the wall and fully covers this wall.
    
    Parameters
    ----------
    rate_map    :   np.ma.MaskedArray
        rate map: N x M array where each cell value is the firing rate of the cell
    fields_map  :   np.ma.MaskedArray
        Integer array of labelled fields. Each cell value is a positive integer. 
        Cells that are members of field have the value corresponding to field_id 
        (non-zero positive  unique integers. Not necessarily contiguous 
        (e.g. 1, 2, 3, 5)). Cells that are not members of fields have value zero
    fields      :   dict
        Dictionary of all known firing fields
        The key is the field_id corresponding to the labelled fields_map
        Each firing field is also given as a dictionary
        Example:
        fields = {1:field_one, 2:field_two}
        field_one = {'field_id':1, 'field_size':2.6, ...}
    kwargs
        search_width    :   int
            rate_map and fields_map have masked values, which may occur within 
            the region of border  pixels. To mitigate this, we check rows/columns
            within search_width pixels of the border.
            If no value is supplied, default 8
        walls           :   str
            Definition of walls along which the border score is calculated. Provided by
            a string which contains characters that stand for walls:
                      T - top wall (we assume the bird-eye view on the arena)
                      R - right wall
                      B - bottom wall
                      L - left wall
                      Characters are case insensitive. Default value is 'TRBL' 
                      meaning that border score is calculated along all walls.
                      Any combination is possible, e.g.  'R' to calculate along
                      right wall, 'BL' to calculate along two walls, e.t.c.

    Returns
    -------
    score   : float
        Border score in the range [-1, 1].
        A value of -1 is given when no fields are provided


    See also
    --------
    BNT.+analyses.placefield
    BNT.+analyses.borderScore
    BNT.+analyses.borderCoverage
    opexebo.analysis.placefield
    opexebo.analysis.bordercoverage
        
    Copyright (C) 2019 by Simon Ball

    This program is free software; you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation; either version 3 of the License, or
    (at your option) any later version.
    '''

    # Extract keyword arguments or set defaults
    sw = kwargs.get('search_width', default.search_width)
    walls = kwargs.get('walls', default.walls)

    # Check that fields exist
    if np.ma.max(fields_map) == 0:
        # No fields exist
        score = -1
    else:
        # 1 or more fields exist
        fields_map_unlabelled = np.copy(fields_map)
        fields_map_unlabelled[fields_map>0] = 1
        coverage = border_coverage(fields, search_width = sw, walls = walls)
        fields_rate_map = fields_map_unlabelled * rate_map
        fields_rate_map = np.ma.masked_invalid(fields_rate_map)
        wfd = _weighted_firing_distance(fields_rate_map)
        score = (coverage - wfd)/(coverage + wfd)
    return score


def _weighted_firing_distance(rmap):
    '''
    parameters
    ---
    rmap    : np.ma.MaskedArray
        Rate map of cells inside fields. NxM array where cells inside fields
        give the firing rate, and cells outside fields are zero
    returns
    -------
    wfd     : float
        Weighted firing distance. 
    '''
    # Check that the provided array is properly masked
    if type(rmap != np.ma.MaskedArray):
        rmap = np.ma.masked_invalid(rmap)

    # Normalise firing map by the sum of the firing map, such that it resembles a PDF
    rmap /= np.ma.sum(rmap)

    # Create an array, same size as rmap, where the cell value is the distance (in bins) from the edge of the array:
    x = np.ones(rmap.shape) # define area same size as ratemap
    x = np.pad(x, 1, mode='constant') # Pad outside with zeros, we calculate distance from nearest zero
    dist = distance_transform_cdt(x, metric='taxicab') # taxicab metric: distance along axes, not diagonal distance. 
    #wfd = np.ma.sum( np.ma.dot(dist[1:-1, 1:-1], rmap) )
    wfd = np.ma.sum( dist[1:-1, 1:-1] * rmap )

    # Normalise by half of the smallest arena dimensions
    wfd = 2 * wfd / np.min(rmap.shape)
    return wfd
