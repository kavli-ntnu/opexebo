"""Provide a function for mapping a list of positional data into a 1D or 2D space"""

import numpy as np
from opexebo.general import validatekeyword__arena_size
import opexebo.defaults as default


def accumulate_spatial(pos, **kwargs):
    """
    Accumulate repeated observations of a variable into a binned representation
    by means of a histogram.

    Parameters
    ----------
    pos : np.ndarray
        Nx1 or Nx2 array of positions associated with z
        x = pos[0,:]
        y = pos[1,:]
        This matches the simplest input creation:
            pos = np.array( [x, y] )
   kwargs
        bin_width       : float.
            Bin size (in cm). Bins are always assumed square default 2.5 cm.
        arena_size      : float or tuple of floats.
            Dimensions of arena (in cm)
            For a linear track, length
            For a circular arena, diameter
            For a square arena, length or (length, length)
            For a non-square rectangle, (length1, length2)
            In this function, a circle and a square are treated identically.
        limits : tuple or np.ndarray
            (x_min, x_max) or (x_min, x_max, y_min, y_max)
            Provide concrete limits to the range over which the histogram searches
            Any observations outside these limits are discarded
            If no limits are provided, then use np.nanmin(data), np.nanmax(data)
            to generate default limits.
            As is standard in python, acceptable values include the lower bound
            and exclude the upper bound

    Returns
    -------
    hist : np.ndarray
        1D or 2D histogram of the occurrences of the input observations
        Dimensions given by arena_size/bin_width
        Not normalised - each cell gives the integer number of occurrences of
        the observation in that cell
    edges : list-like
        x, or (x, y), where x, y are 1d np.ndarrays
        Here x, y correspond to the output histogram

    See also
    --------
    BNT.+analyses.map()

    Copyright (C) 2019 by Simon Ball

    This program is free software; you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation; either version 3 of the License, or
    (at your option) any later version.
    """

    # Check correct inputs:
    dims = pos.ndim
    if dims not in (1, 2):
        raise ValueError("pos should have either 1 or 2 dimensions. You have"\
                         " provided %d dimensions." % dims)

    # Get kwargs values
    debug = kwargs.get("debug", False)
    bin_width = kwargs.get("bin_width", default.bin_width)
    arena_size = kwargs.get("arena_size")
    limits = kwargs.get("limits", None)
    if type(limits) not in (tuple, list, np.ndarray, type(None)):
        raise ValueError("You must provide an array-like 'limits' value, e.g."\
          " (x_min, x_max, y_min, y_max). You provided type %s" % type(limits))

    arena_size, is_2d = validatekeyword__arena_size(arena_size, dims)
    num_bins = np.ceil(arena_size / bin_width).astype(int)

    # Histogram of positions
    

    if is_2d:
        x = pos[0]
        y = pos[1]
        if limits is None:
            limits = ( [np.nanmin(x), np.nanmax(x)],
                         [np.nanmin(y), np.nanmax(y)] )
            if debug:
                print("No limits found. Calculating based on min/max")
        elif len(limits) != 4:
            raise ValueError("You must provide a 4-element 'limits' value for a"\
                             " 2D map. You provided %d elements" % len(limits))
        else:
            limits = ( [limits[0], limits[1]], # change from a 4 element list to a list of lists
                             [limits[2], limits[3]] )
        in_range = np.logical_and( 
                np.logical_and(
                        np.greater_equal(x, limits[0][0]), np.less(x, limits[0][1])),
                np.logical_and(
                        np.greater_equal(y, limits[1][0]), np.less(y, limits[1][1])) )
        # the simple operator ">= doesn't respect Masked Arrays
        # As of 2019, it does actually behave correctly (NaN is invalid and so
        # is removed), but I would prefer to be explicit

        in_range_x = x[in_range]
        in_range_y = y[in_range]
        if debug:
            print(f"Limits: {limits}")
            print(f"numbins : {num_bins}")
            print(f"data points : {len(in_range_x)}")
            # Testing for invalid inputs that have made it past validation
            # in_range_* should be of non-zero, equal length
            assert len(in_range_x) == len(in_range_y)
            assert len(in_range_x) > 0
            # in_range_* should be all real
            assert np.isfinite(in_range_x).all()
            assert np.isfinite(in_range_y).all()
            # Non zero number of bins
            assert num_bins[0] > 1
            assert num_bins[1] > 1
            # Finite limits
            assert np.isfinite(np.array(limits)).all()

            
            

        hist, xedges, yedges = np.histogram2d(in_range_x, in_range_y,
                                       bins=num_bins, range=limits)
        hist = hist.transpose() # Match the format that BNT traditionally used.
        edges = np.array([yedges, xedges]) # Note that due to the tranposition
                            # the label xedge, yedge is potentially misleading
    else:
        x = pos
        if limits is None:
            limits = [np.nanmin(x), np.nanmax(x)]
        elif len(limits) != 2: 
            raise ValueError("You must provide a 2-element 'limits' value for a"\
                             " 1D map. You provided %d elements" % len(limits))
        in_range = np.logical_and(np.greater_equal(x, limits[0]),
                                  np.less(x, limits[1]))
        in_range_x = x[in_range]

        hist, edges = np.histogram(in_range_x, bins=num_bins, range=limits)

    return hist, edges
