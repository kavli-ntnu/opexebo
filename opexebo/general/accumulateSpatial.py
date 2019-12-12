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
        bin_width : float. 
            Bin size in cm. Bins are always assumed square default 2.5 cm. One
            of `bin_width`, `bin_number`, `bin_edges` must be provided
        bin_number: int
            Number of bins. The same number will be used along both axes,
            permitting rectangular bins. One of `bin_width`, `bin_number`,
            `bin_edges` must be provided
        bin_edges: array-like
            Edges of the bins. Provided either as `edges` or `(x_edges, y_edges)`. One
            of `bin_width`, `bin_number`, `bin_edges` must be provided
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
    arena_size = kwargs.get("arena_size")
    limits = kwargs.get("limits", None)
    if type(limits) not in (tuple, list, np.ndarray, type(None)):
        raise ValueError("You must provide an array-like 'limits' value, e.g."\
          " (x_min, x_max, y_min, y_max). You provided type %s" % type(limits))

    arena_size, is_2d = validatekeyword__arena_size(arena_size, dims)


    # Handle the decision of bin_edges
    # Logic:
        # If none are provided, use default bin_width
        # If more than 1 is provided, raise Exception
        # If bin_edges are provided, use them
        # Else use bin_width, if provided
        # Else use bin_number, if provided
    bin_number = kwargs.get("bin_number", False)
    bin_width = kwargs.get("bin_width", False)
    bin_edges = kwargs.get("bin_edges", False)
    if not bin_edges and not bin_width and not bin_number:
        # No bin decision was provided, so go with default
        bin_width = default.bin_width
    elif sum([bool(bin_number), bool(bin_width), bool(bin_edges)]) != 1:
        # Count the number of values where a value other than False is present
        # If there are more than 1 "True" value, then the user has provided too many keywords
        raise KeyError("You have provided more than one method for determining"\
                       " the edges of the histogram. Only zero or one methods"\
                       " can be accepted.")
    elif bool(bin_edges):
        # First priority: use predefined bin_edges
        # TODO! TODO!
        # Bear in mind that the histogram will be transposed as part of this function
        # So for the Ratemap
        if is_2d:
            if type(bin_edges) not in (tuple, list, np.ndarray):
                raise ValueError("keyword 'bin_edges' must be either a tuple or list (of np.ndarrays), or a")
        else:
            if not isinstance(bin_edges, np.ndarray):
                raise ValueError("Keyword 'bin_edges' must be a numpy array for a 1D histogram")
        bins = (bin_edges[1], bin_edges[0])
        debug_bin_type = "bin_edges"
    elif bool(bin_width):
        # Calculate the number of bins based on the requested width and arena_size
        bins = np.ceil(arena_size / bin_width).astype(int)
        debug_bin_type = "bin_width"
    elif bool(bin_number):
        if not isinstance(bin_number, int):
            raise ValueError("Keyword 'bin_number' must be an integer")
        bins = bin_number
        debug_bin_type = "bin_number"
    



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
            print(f"Binning type: {debug_bin_type}")
            print(f"bins : {bins}")
            print(f"data points : {len(in_range_x)}")

        hist, xedges, yedges = np.histogram2d(in_range_x, in_range_y,
                                       bins=bins, range=limits)
        hist = hist.transpose() # Match the format that BNT traditionally used.
        edges = [yedges, xedges] # Note that due to the tranposition
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

        hist, edges = np.histogram(in_range_x, bins=bins, range=limits)

    return hist, edges
