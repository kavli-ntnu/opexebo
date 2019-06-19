"""Provide function for calculating a Border Score"""

import numpy as np
from scipy.ndimage import distance_transform_edt
import opexebo.defaults as default


def border_coverage(fields, **kwargs):
    '''
    Calculate border coverage for detected fields.
    
    This function calculates firing map border coverage that is further
    used in calculation of a border score.


    TODO
    I havefollowed the approach used in BNT, but I want to double check whether there 
    is a better way to do this - it only tells you that *a* field (it doesn't tell 
    you which one) has coverage of *a* border (it doesn't tell you which one)).


    It seems like there should be a better way of doing this
        (e.g. return a vector of coverage, i.e. a value for each border checked, and 
        return an index of the best field for each border, or something similar)

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

    See also
    --------
    BNT.+analyses.placefield
    BNT.+analyses.borderScore
    BNT.+analyses.borderCoverage
    opexebo.analysis.placefield
    opexebo.analysis.borderscore
        
    Copyright (C) 2019 by Simon Ball

    This program is free software; you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation; either version 3 of the License, or
    (at your option) any later version.
    '''

    # Extract keyword arguments or set defaults
    sw = kwargs.get('search_width', default.search_width)
    walls = kwargs.get('walls', default.walls)

    # Check that the wall definition is valid
    _validate_wall_definition(walls)
    walls = walls.lower()

    # Check coverage of each field in turn
    coverage = 0
    for field in fields:
        fmap = field['map'] # binary image of field: values are 1 inside field, 0 outside
        if "l" in walls:
            aux_map = fmap[:,:sw].copy()
            c = _wall_field(aux_map)
            if c > coverage:
                coverage = c

        if "r" in walls:
            aux_map = fmap[:, -sw:].copy()
            aux_map = np.fliplr(aux_map) # Mirror image to match the expectations in _wall_field, i.e. border adjacent to left-most column
            c = _wall_field(aux_map)
            if c > coverage:
                coverage = c

        # since we are dealing with data that came from a camera
        #'bottom' is actually at the top of the matrix fmap
        # i.e. in a printed array (think Excel spreadheet), [0,0] is at top-left.
        # in a figure (think graph) (0,0) is at bottom-left

        # Note: because I use rotations instead of transposition, this yields 
        # arrays that are upside-down compared to Vadim's version, 
        # BUT the left/right is correct.
        if "b" in walls:
            aux_map = fmap[:sw, :].copy()
            aux_map = np.rot90(aux_map) # Rotate counterclockwise - top of image moves to left of image
            c = _wall_field(aux_map)
            if c > coverage:
                coverage = c

        if "t" in walls:
            aux_map = fmap[-sw:, :].copy()
            aux_map = np.fliplr(np.rot90(aux_map)) # rotate 90 deg counter clockwise (bottom to right), then mirror image
            c = _wall_field(aux_map)
            if c > coverage:
                coverage = c
    return coverage


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
    if type(wfmap) != np.ma.MaskedArray:
        wfmap = np.ma.asanyarray(wfmap)
    N = wfmap.shape[0]
    wfmap[wfmap>1] = 1 # Just in case a still-labelled map has crept in
    inverted_wfmap = 1-wfmap 
    distance = distance_transform_edt(inverted_wfmap)
    distance = np.ma.masked_where(wfmap.mask, distance) # Preserve masking
    # distance_transform_edt(1-map) is the Python equivalent to (Matlab bwdist(map))
    # Cells in map with value 1 go to value 0
    # Cells in map with value 0 go to the geometric distance to the nearest value 1 in map

    adjacent_sites = distance[:,0]
    # Identify sites which are NaN, inf, or masked
    # Replace them with the next cell along the row, closest to the wall, that is not masked, nan, or inf
    adjacent_sites.mask += np.isnan(adjacent_sites.data)
    if adjacent_sites.mask.any():
        for i, rep in enumerate(adjacent_sites.mask):
            if rep:
                for j, val in enumerate(distance[i,:]):
                    if not distance.mask[i,j] and not np.isnan(val):
                        adjacent_sites[i] = val
                        adjacent_sites.mask[i] = False
                        break
    covered = np.ma.sum(adjacent_sites==0)
    contributing_cells = N - np.sum(adjacent_sites.mask) # The sum gives the number of remaining inf, nan or masked cells

    coverage = covered / contributing_cells

    return coverage


def _validate_wall_definition(walls):
    '''Parse the walls argument for invalid entry'''
    if type(walls) != str:
        raise ValueError("Wall definition must be given as a string, e.g. 'trbl'. %s is not a valid input." % str(walls))
    elif len(walls) > 4:
        raise ValueError("Wall definition may not exceed 4 characters. String '%s' contains %d characters." % (walls, len(walls)))
    elif len(walls) == 0:
        raise ValueError("Wall definition must contain at least 1 character from the set [t, r, b, l]")
    else:
        walls = walls.lower()
        for char in walls:
            if char.lower() not in ["t","r","b","l"]:
                raise ValueError("Character %s is not a valid entry in wall definition. Valid characters are [t, r, b, l]" % char)


if __name__ == '__main__':
    import matplotlib.pyplot as plt
    test_data = [[0,0,0,0,0,5,0,4],
                 [0,0,0,0,1,1,2,4],
                 [0,0,3,6,6,4,4,1],
                 [0,5,2,5,2,4,2,1],
                 [0,0,1,5,3,2,1,1],
                 [0,0,2,5,2,2,6,1],
                 [0,0,0,5,6,6,6,6]]
    test_data = np.ma.MaskedArray(test_data)
    test_data.mask = np.zeros(test_data.shape, dtype=bool)
    test_data[test_data.data>0]=1
    test_data.mask[3,5] = True
    test_data.mask[3,6] = True
    test_data.mask[0,0] = True
    
    inv_wf = 1-test_data
    distance = distance_transform_edt(inv_wf)
    distance = np.ma.masked_where(test_data.mask, distance)
    sw = 2
    fig, ((ax1, ax2), (ax3, ax4), (ax5, ax6)) = plt.subplots(3,2)
    ax1.imshow(distance)
    ax3.imshow((distance[:, :sw]))
    ax3.set_ylabel("L")
    ax4.imshow(np.fliplr(distance[:, -sw:]))
    ax4.set_ylabel("R")
    ax5.imshow(np.rot90(distance[:sw, :]))
    ax5.set_ylabel("B")
    ax6.imshow(np.fliplr(np.rot90(distance[-sw:, :])))
    ax6.set_ylabel("T")
    plt.show()