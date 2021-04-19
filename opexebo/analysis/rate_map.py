"""Provides a function for calculating a Rate map from an
Occupancy Map and positioned Spike Times"""

import numpy as np

import opexebo
#import opexebo.defaults as default
from .. import errors as err, defaults as default


def rate_map(occupancy_map, spikes_tracking, arena_size, **kwargs):
    ''' Calculate the spatially correlated firing rate map

    The rate map is calculated by the number of spikes in a bin divided by the
    time the animal spent in that bin. Bins in which the animal spent no or
    very little time are masked such that the value is available for but
    typically excluded from future analyses.

    The provided arena_size and bin_width **must* provide a number of bins such
    that the spike map has the same dimensions as the occupancy map.

    Parameters
    ----------
    occupancy_map: np.ndarray or np.ma.MaskedArray
        Nx1 or Nx2 array of time spent by animal in each bin, with time in bins
    spikes: np.ndarray
        Nx3 or Nx4 array of spikes tracking: [t, speed, x] or [t, speed, x, y].
        Speeds are used for the same purpose as in Spatialoccupancy - spikes
        occurring with an instaneous speed below the threshold are discarded
    arena_size: float or tuple of floats
        Dimensions of arena (in cm)
            * For a linear track, length
            * For a circular arena, diameter
            * For a rectangular arena, length or (length, length)
    speed_cutoff: float
        Timestamps with instantaneous speed beneath this value are ignored. Default 0
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
    rmap: np.ma.MaskedArray
        2D array, masked at locations of very low occupancy (t<1ms).
        Each cell gives the rate of neuron firing at that location.

    See Also
    --------
    opexebo.general.accumulate_spatial
    opexebo.general.bin_width_to_bin_number

    Notes
    --------
    BNT.+analyses.map()

    Copyright (C) 2019 by Simon Ball
    '''

    # Check correct inputs
    if not isinstance(occupancy_map, (np.ndarray, np.ma.MaskedArray)):
        raise err.ArgumentError("Occupancy Map not provided in usable format. Please"\
            " provide either a Numpy ndarray or Numpy MaskedArray. You"\
            f" provided {type(occupancy_map)}.")
    if not isinstance(spikes_tracking, (np.ndarray, np.ma.MaskedArray)):
        raise err.ArgumentError("spikes not provided in usable format. Please"\
            " provide either a Numpy ndarray or Numpy MaskedArray. You"\
            f" provided {type(spikes_tracking)}.")

    dims_p = occupancy_map.ndim
    dims_s, _ = spikes_tracking.shape
    if dims_s-2 != dims_p:
        raise err.DimensionMismatchError("Spikes must have the same number of spatial"\
            " dimensions as positions ([t, s, x] or [t, s, x, y]). You have provided"\
            f" {dims_s} columns of spikes, and {dims_p} columns of positions")

    if isinstance(occupancy_map, np.ndarray):
        occupancy_map = np.ma.MaskedArray(occupancy_map)

    speed_cutoff = kwargs.get("speed_cutoff", default.speed_cutoff)

    speeds = spikes_tracking[1, :]
    good_speeds = speeds > speed_cutoff
    if dims_s == 3:
        spikes = spikes_tracking[2, :][good_speeds]
    elif dims_s == 4:
        spikes_x = spikes_tracking[2, :][good_speeds]
        spikes_y = spikes_tracking[3, :][good_speeds]
        spikes = np.array((spikes_x, spikes_y))

    # Histogram of spike positions
    spike_map, _ = opexebo.general.accumulate_spatial(spikes, arena_size, **kwargs)

    if spike_map.shape != occupancy_map.shape:
        raise err.DimensionMismatchError("Rate Map and Occupancy Map must have the same"\
                         f" dimensions. Provided: {spike_map.shape},"\
                         f" {occupancy_map.shape}")
    # Convert to rate map
    rmap = spike_map / (occupancy_map + np.spacing(1))
    # spacing adds floating point precision to avoid DivideByZero errors
    # These should be impossible due to masking, but included nevertheless
    return rmap
