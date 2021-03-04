"""Provide a function for mapping a list of positional data into a 1D or 2D space"""

import numpy as np
from opexebo.general import validatekeyword__arena_size, bin_width_to_bin_number
import opexebo.defaults as default


def accumulate_spatial(pos, arena_size, **kwargs):
    """
    Given a list of positions, create a histogram of those positions. The
    resulting histogram is typically referred to as a map.
    
    The complexity in this function comes down to selecting where the edges of
    the arena are, and generating the bins within those limits.
    
    The histogram bin edges must be defined in one of 3 different ways
    
        * bin_width: based on the keyword `arena_size`, the number of bins will
          be calculated as
            `opexebo.general.bin_width_to_bin_number`
          The histogram will use `num_bins` between the minimum and maximum of the
          positions (or `limit` if provided)
        * bin_number: the histogram will use bin_number of bins between the
          minimum and maximum of the positions (or `limit` if provided)
        * bin_edges: the histogram will use the provided `bin_edg`e arrays

    Either zero or one of the three bin_* keyword arguments must be defined.
    If none are defined, then a default bin_width is used. If more than 1 is
    defined, an error is raised

    Parameters
    ----------
    pos: np.ndarray
        1D or 2D array of positions  in row-major format, i.e. `x` = pos[0],
        `y` = pos[1]. This matches the simplest input creation pos = np.array( [`x`, `y`] )
    arena_size: float or tuple of floats
        Dimensions of arena (in cm)
            * For a linear track, length
            * For a circular arena, diameter
            * For a rectangular arena, length or (length, length)
    bin_width: float
        Bin size in cm. Default 2.5cm. If bin_width is supplied, `limit` must
        also be supplied. One of `bin_width`, `bin_number`, `bin_edges` must be
        provided
    bin_number: int or tuple of int
        Number of bins. If provided as a tuple, then `(x_bins, y_bins)`. One
        of `bin_width`, `bin_number`, `bin_edges` must be provided
    bin_edges: array-like
        Edges of the bins. Provided either as `edges` or `(x_edges, y_edges)`.
        One of `bin_width`, `bin_number`, `bin_edges` must be provided
    limits: tuple or np.ndarray
        (x_min, x_max) or (x_min, x_max, y_min, y_max)
        Provide concrete limits to the range over which the histogram searches
        Any observations outside these limits are discarded
        If no limits are provided, then use np.nanmin(data), np.nanmax(data)
        to generate default limits.
        As is standard in python, acceptable values include the lower bound
        and exclude the upper bound

    Returns
    -------
    hist: np.ndarray
        1D or 2D histogram of the occurrences of the input observations
        Dimensions given by arena_size/bin_width
        Not normalised - each cell gives the integer number of occurrences of
        the observation in that cell
    edges: list-like
        `x`, or (`x`, `y`), where `x`, `y` are 1d np.ndarrays
        Here `x`, `y` correspond to the output histogram

    See Also
    --------
    opexebo.general.bin_width_to_bin_number
    
    Notes
    --------
    BNT.+analyses.map()

    Copyright (C) 2019 by Simon Ball
    """

    # Check correct inputs:
    dims = pos.ndim
    if dims not in (1, 2):
        raise ValueError("pos should have either 1 or 2 dimensions. You have"\
                         " provided %d dimensions." % dims)

    # Get kwargs values
    debug = kwargs.get("debug", False)
    limits = kwargs.get("limits", None)
    if not isinstance(limits, (tuple, list, np.ndarray, type(None))):
        raise ValueError("You must provide an array-like 'limits' value, e.g."\
          " (x_min, x_max, y_min, y_max). You provided type %s" % type(limits))

    arena_size, is_2d = validatekeyword__arena_size(arena_size, dims)

    
    ###########################################################################
    ####### Handle the decision of bin_edges
    # Logic:
        # If none are provided, use default bin_width
        # If more than 1 is provided, raise Exception
        # If bin_edges are provided, use them
        # Else use bin_width, if provided
        # Else use bin_number, if provided
    bin_number = kwargs.get("bin_number", None)
    bin_width = kwargs.get("bin_width", None)
    bin_edges = kwargs.get("bin_edges", None)
    if bin_edges is None and bin_width is None and bin_number is None:
        # No bin decision was provided, so go with default
        bin_width = default.bin_width
        debug_bin_type = "default - bin_width"
    elif sum( [x is not None for x in (bin_edges, bin_width, bin_number)] ) != 1:
        # Count the number of values where a value other than False is present
        # If there are more than 1 "True" value, then the user has provided too many keywords
        raise KeyError("You have provided more than one method for determining"\
                       " the edges of the histogram. Only zero or one methods"\
                       " can be accepted.")

    if bin_edges is not None:
        # First priority: use predefined bin_edges
        # Remember - user provides (x, y), but histogram needs (y, x)
        if is_2d:
            if not isinstance(bin_edges, (tuple, list, np.ndarray)):
                raise ValueError("keyword 'bin_edges' must be either a tuple or list"\
                                 " (of np.ndarrays), or a 2D array")
        else:
            if not isinstance(bin_edges, np.ndarray):
                raise ValueError("Keyword 'bin_edges' must be a numpy array for a 1D histogram")
        bins = (bin_edges[1], bin_edges[0])
        debug_bin_type = "bin_edges"

    elif bool(bin_width):
        # Calculate the number of bins based on the requested width and arena_size
        # Then calculate the actual bin edges that this would give, based on expanding from top left
        # Given arena size in (x, y), this also returns in (x, y) -> have to convert for Numpy
        num_bins = bin_width_to_bin_number(arena_size, bin_width)
        if limits is None:
            # Handle the case that limits is not provided, i.e. is None
            lim = (0, 0, 0, 0)
        else:
            lim = limits
        if is_2d:
            # Have to swap to (y, x)
            bins = (np.linspace(0, arena_size[1], num_bins[1]+1) + lim[2],
                    np.linspace(0, arena_size[0], num_bins[0]+1) + lim[0])
        else:
            bins = np.linspace(0, arena_size, num_bins+1) + (lim[0])
        debug_bin_type = "bin_width"

    elif bool(bin_number):
        if isinstance(bin_number, int):
            bins = bin_number
        elif isinstance(bin_number, (tuple, list, np.ndarray)):
            # If an array, we expect to recieve (x, y)
            # Have to convert for the Numpy standard of (y, x)
            bins = (bin_number[1], bin_number[0])
        else:
            raise ValueError("Keyword 'bin_number' must be an integer, or an"\
                             " array-like of integers.")
        debug_bin_type = "bin_number"

    if debug:
        print(f"Limits: {limits}")
        print(f"Binning type: {debug_bin_type}")
        print(f"bins : {bins}")

    ###########################################################################
    ###### Make the histogram
    if is_2d:
        x = pos[0]
        y = pos[1]
        if limits is None:
            limits = np.array([np.nanmin(y), np.nanmax(y)*1.001, np.nanmin(x),
                               np.nanmax(x)*1.001]).reshape(2, 2)
            # numpy convention: (y, x)
            if debug:
                print("No limits found. Calculating based on min/max")
        elif len(limits) != 4:
            raise ValueError("You must provide a 4-element 'limits' value for a"\
                             " 2D map. You provided %d elements" % len(limits))
        else:
            limits = np.flipud(np.array(limits).reshape(2, 2))
            # the flipud swaps x and y - rememebr, numpy convention that (y, x)
        in_range = np.logical_and(np.logical_and(np.greater_equal(y, limits[0, 0]), 
                                                 np.less(y, limits[0, 1])),
                                  np.logical_and(np.greater_equal(x, limits[1, 0]), 
                                                 np.less(x, limits[1, 1]))
                                  )
        # the simple operator ">= doesn't respect Masked Arrays
        # As of 2019, it does actually behave correctly (NaN is invalid and so
        # is removed), but I would prefer to be explicit

        in_range_x = x[in_range]
        in_range_y = y[in_range]
        if debug:
            print(f"data points : {len(in_range_x)}")

        '''
        A brief word on the documentation for np.histogram2d()
        https://docs.scipy.org/doc/numpy/reference/generated/numpy.histogram2d.html
        
        The documentation is subtlely misleading in the use of `x` and `y`. 
        
        The NumPy standard notation is (almost) invariably to call (y, x), and, e.g.,
        in a 2D array, you would create an array with 5 rows and 2 columns like follows:
            np.zeros((5, 2))
        
        We would normally describe this as an array with a height (corresponding to y)
        of 5, and a width (corresponding to x) of 2.
        
        The documentation for histogram2d gives the signature as:
            numpy.histogram2d(x, y, bins=10, range=None, normed=None, weights=None, density=None)
        
            Returns:
                H : ndarray, shape(nx, ny)
                    The bi-dimensional histogram of samples x and y. Values in x are 
                    histogrammed along the first dimension and values in y are histogrammed 
                    along the second dimension.
                xedges : ndarray, shape(nx+1,)
                    The bin edges along the first dimension.
                yedges : ndarray, shape(ny+1,)
                    The bin edges along the second dimension.
        
        Note the order in the Returns section: data given as `x` is histogrammed along the
        _first_ dimension, which in NumPy parlance, would _usually_ be labelled `y`.
        The use of `x`, `y` is self-consistent within this page, but misleading
        in the context of other NumPy functions. 
        
        By invoking `y` as the first axis, and then returning edges as [xedges, yedges],
        we are internally consistent with the mathematical notation that I have used throughout
        opexebo (i.e. to call x, then y)
        '''
        hist, yedges, xedges = np.histogram2d(in_range_y, in_range_x, bins=bins, range=limits)
        edges = [xedges, yedges]

    else: # is not 2d
        x = pos
        if limits is None:
            limits = [np.nanmin(x), np.nanmax(x)*1.0001]
        elif len(limits) != 2: 
            raise ValueError("You must provide a 2-element 'limits' value for a"\
                             " 1D map. You provided %d elements" % len(limits))
        in_range = np.logical_and(np.greater_equal(x, limits[0]),
                                  np.less(x, limits[1]))
        in_range_x = x[in_range]
        if debug:
            print(f"data points : {len(in_range_x)}")

        hist, edges = np.histogram(in_range_x, bins=bins, range=limits)

    return hist, edges
