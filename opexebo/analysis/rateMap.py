"""Provides a function for calculating a Rate map from an 
Occupancy Map and positioned Spike Times"""

import numpy as np

import opexebo
import opexebo.defaults as default


def rate_map(occupancy_map, spikes_tracking, **kwargs):
    ''' Calculate the spatially correlated firing rate map
    
    The rate map is calculated by the number of spikes in a bin divided by the
    time the animal spent in that bin. Bins in which the animal spent no or 
    very little time are masked such that the value is available for but
    typically excluded from future analyses.
    
    The provided arena_size and bin_width **must* provide a number of bins such 
    that the spike map has the same dimensions as the occupancy map. 
    
    Parameters
    ----------
    occupancy_map : np.ndarray or np.ma.MaskedArray
        Nx1 or Nx2 array of time spent by animal in each bin, with time in bins
    spikes : np.ndarray
        Nx3 or Nx4 array of spikes tracking: [t, speed, x] or [t, speed, x, y].
        Speeds are used for the same purpose as in Spatialoccupancy - spikes
        occurring with an instaneous speed below the threshold are discarded
    **kwargs
        bin_width : float. 
            Bin size in cm. Bins are always assumed square default 2.5 cm. One
            of `bin_width`, `bin_number`, `bin_edges` must be provided
        bin_number: int or tuple of int
            Number of bins. If a tuple is provided, then (x, y). Either square
            or rectangular bins are supported. One of `bin_width`, `bin_number`,
            `bin_edges` must be provided
        bin_edges: array-like
            Edges of the bins. Provided either as `edges` or `(x_edges, y_edges)`. One
            of `bin_width`, `bin_number`, `bin_edges` must be provided
        speed_cutoff    : float. 
            Timestamps with instantaneous speed beneath this value are ignored. 
            Default 0
        arena_size : float or tuple of floats. 
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
    rmap : np.ma.MaskedArray
        2D array, masked at locations of very low occupancy (t<1ms).
        Each cell gives the rate of neuron firing at that location.
    
    Raises
    ------
    ValueError
        If input values do not match expectations
    KeyError
        Absence of arena information

    See also
    --------
    BNT.+analyses.map()
    '''

    # Check correct inputs
    if type(occupancy_map) not in (np.ndarray, np.ma.MaskedArray) :
        raise ValueError("Occupancy Map not provided in usable format. Please"\
            " provide either a Numpy ndarray or Numpy MaskedArray. You"\
            f" provided {type(occupancy_map)}.")
    if type(spikes_tracking) not in (np.ndarray, np.ma.MaskedArray) :
        raise ValueError("spikes not provided in usable format. Please"\
            " provide either a Numpy ndarray or Numpy MaskedArray. You"\
            f" provided {type(spikes_tracking)}.")
    
    dims_p = occupancy_map.ndim
    dims_s, num_samples_s = spikes_tracking.shape
    if dims_s-2 != dims_p:
        raise ValueError("Spikes must have the same number of spatial"\
            " dimensions as positions ([t, s, x] or [t, s, x, y]). You have provided"\
            f" {dims_s} columns of spikes, and {dims_p} columns of positions")
    
    if "arena_size" not in kwargs:
        raise KeyError("No arena dimensions provided. Please provide the\
                    dimensions of the arena by using keyword 'arena_size'.")

    if type(occupancy_map) == np.ndarray:
        occupancy_map = np.ma.MaskedArray(occupancy_map)

    speed_cutoff = kwargs.get("speed_cutoff", default.speed_cutoff)

    times = spikes_tracking[0, :] # never actually used
    speeds = spikes_tracking[1, :]
    good_speeds = speeds>speed_cutoff
    if dims_s == 3:
        spikes = spikes_tracking[2, :][good_speeds]
    elif dims_s == 4:
        spikes_x = spikes_tracking[2, :][good_speeds]
        spikes_y = spikes_tracking[3, :][good_speeds]
        spikes = np.array((spikes_x, spikes_y))

    # Histogram of spike positions
    spike_map, edges = opexebo.general.accumulate_spatial(spikes, **kwargs)

    if spike_map.shape != occupancy_map.shape:
        raise ValueError("Rate Map and Occupancy Map must have the same"\
                    " dimensions. Provided: %s, %s" % (spike_map.shape, 
                                                    occupancy_map.shape))
    # Convert to rate map
    rmap = spike_map / (occupancy_map + np.spacing(1)) 
    # spacing adds floating point precision to avoid DivideByZero errors
    # These should be impossible due to masking, but included nevertheless
    return rmap
    
    