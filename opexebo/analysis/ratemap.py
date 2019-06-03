"""Provides a function for calculating a Rate map from an Occupancy Map and positioned Spike Times"""

import numpy as np
from ..general import validatekeyword__arena_size, accumulatespatial
from .. import defaults as default

def ratemap(occupancy_map, spikes, **kwargs):
    ''' Calculate the spatially correlated firing rate map
    
    The rate map is calculated by the number of spikes in a bin divided by the
    time the animal spent in that bin. Bins in which the animal spent no or 
    very little time are masked such that the value is available for but typically
    excluded from future analyses.
    
    The provided arena_size and bin_width **must* provide a number of bins such 
    that the spike map has the same dimensions as the occupancy map. 
    
    Parameters
    ----------
    occupancy_map : np.ndarray or np.ma.MaskedArray
        Nx1 or Nx2 array of time spent by animal in each bin
        
    spikes : np.ndarray
        Nx2 or Nx3 array of spike positions: [t, x] or [t, x, y]. Must have the same 
        dimensionality as positions (i.e. 1d, 2d)
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
    rate_map : np.ma.MaskedArray
        2D array, masked at locations of very low occupancy (t<1ms).
        Each cell gives the rate of neuron firing at that location.
    
    See also
    --------
    BNT.+analyses.map
    
    
    Copyright (C) 2019 by Simon Ball
    
    This program is free software; you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation; either version 3 of the License, or
    (at your option) any later version.
    '''
    
    # Check correct inputs
    dims_p = occupancy_map.ndim
    dims_s, num_samples_s = spikes.shape
    if dims_s-1 != dims_p:
        raise ValueError("Spikes must have the same number of columns as positions ([t,x] or [t, x, y]). You have provided %d columns of spikes, and %d columns of positions" % (dims_s, dims_p))
    if type(occupancy_map) not in (np.ndarray, np.ma.MaskedArray) :
        raise ValueError("Occupancy Map not provided in usable format. Please provide either a Numpy ndarray or Numpy MaskedArray. You provided %s." % type(occupancy_map))
    if "arena_size" not in kwargs:
        raise ValueError("No arena dimensions provided. Please provide the dimensions of the arena by using keyword 'arena_size'.")
    
    if type(occupancy_map) == np.ndarray:
        occupancy_map = np.ma.MaskedArray(occupancy_map)
        

    # Histogram of spike positions
    spike_map = accumulatespatial(spikes[1:,:], **kwargs)[0]

        
    if spike_map.shape != occupancy_map.shape:
        raise ValueError("Rate Map and Occupancy Map must have the same dimensions. Provided: %s, %s" % (spike_map.shape, occupancy_map.shape))
    # Convert to rate map
    rate_map = spike_map / (occupancy_map + np.spacing(1)) # Ensure that DivideByZero errors do not occur
                                                            # Zero-valued cells in occupancy_map *should* have already been masked out
                                                            # but added protection is rarely a bad thing
    
    return rate_map
    
    