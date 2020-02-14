'''
Provides function to calculate the spatial occupancy of the arena
'''
import numpy as np
import opexebo
from opexebo import defaults as default



def spatial_occupancy(time, position, speed, **kwargs):
    '''
    Generate an occpuancy map: how much time the animal spent in each location
    in the arena.

    NOTES: This assumes that the positions have already been aligned and curated
    to remove NaNs. This is based on the expectation that it will primarily be
    used within the DataJoint framework, where the curation takes place at a
    much earlier stage.

    Parameters
    ----------
    time: np.ndarray
        timestamps of position and speed data
    position: np.ndarray (x, [y])
        1d or 2d array of positions at timestamps. If 2d, then row major, such
        that `position[0]` corresponds to all `x`; and `position[1]` to all `y`
    speed: np.ndarray
        1d array of speeds at timestamps
    speed_cutoff: float
        Timestamps with instantaneous speed beneath this value are ignored. Default 0
    arena_shape: {"square", "rect", "circle", "line"}
        Rectangular and square are equivalent. Elliptical or n!=4 polygons
        not currently supported. Defaults to Rectangular
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
    arena_size: float or tuple of floats
        Dimensions of arena (in cm)
            * For a linear track, length
            * For a circular arena, diameter
            * For a rectangular arena, length or (length, length)
    debug: bool, optional
        If `true`, print out debugging information throughout the function.
        Default `False`

    Returns
    -------
    masked_map: np.ma.MaskedArray
        Unsmoothed map of time the animal spent in each bin.
        Bins which the animal never visited are masked (i.e. the mask value is
        `True` at these locations)
    coverage: float
        Fraction of the bins that the animal visited. In range [0, 1]
    bin_edges: ndarray or tuple of ndarray
        x, or (x, y), where x, y are 1d np.ndarrays
        Here x, y correspond to the output histogram

    See Also
    --------
    opexebo.general.accumulate_spatial
    opexebo.general.bin_width_to_bin_number

    Notes
    --------
    BNT.+analyses.map

    Copyright (C) 2019 by Simon Ball
    '''
    # Check for correct shapes.
    dimensionality, num_samples = position.shape
    if dimensionality not in (1, 2):
        raise ValueError("Positions array has the wrong number of columns (%d)" % dimensionality)
    if speed.ndim != 1:
        raise ValueError("Speed array has the wrong number of columns")
    if speed.size != num_samples:
        raise ValueError("Speed array does not have the same number of samples as Positions")
    if "arena_size" not in kwargs:
        raise KeyError("Arena dimensions not provided. Please provide dimensions"\
                       " using keyword 'arena_size'.")

    # Handle NaN positions by converting to a Masked Array
    position = np.ma.masked_invalid(position)

    speed_cutoff = kwargs.get("speed_cutoff", default.speed_cutoff)
    debug = kwargs.get("debug", False)

    if debug:
        print("Number of time stamps: %d" % len(time))
        print("Maximum time stamp value: %.2f" % time[-1])
        print("Time stamp delta: %f" % np.min(np.diff(time)))

    good = np.ma.greater_equal(speed, speed_cutoff)
    pos = np.array([position[0, :][good],
                    position[1, :][good]])

    occupancy_map, bin_edges = opexebo.general.accumulate_spatial(pos, **kwargs)
    if debug:
        print(f"Frames included in histogram: {np.sum(occupancy_map)}"\
              f" ({np.sum(occupancy_map)/len(time):3})")

    # So far, times are expressed in units of tracking frames
    # Convert to seconds:
    frame_duration = np.min(np.diff(time))
    occupancy_map_time = occupancy_map * frame_duration

    if debug:
        print(f"Time length included in histogram: {np.sum(occupancy_map_time):2}"\
              f"({np.sum(occupancy_map_time)/time[-1]:3})")

    masked_map = np.ma.masked_where(occupancy_map < 0.001, occupancy_map_time)

    # Calculate the fractional coverage based on the mask. The occupancy_map is
    # zero where the animal has not gone, and therefore non-zero where the animal
    # HAS gone. . Coverage is 1.0 when the animal has visited every location
    # Does not take account of a circular arena, where not all locations are
    # accessible

    arena_size = kwargs.get("arena_size")
    shape = kwargs.get("arena_shape", default.shape)

    if shape.lower() in default.shapes_square:
        coverage = np.count_nonzero(occupancy_map) / occupancy_map.size
    elif shape.lower() in default.shapes_circle:
        if isinstance(arena_size, (float, int)):
            diameter = arena_size
        elif isinstance(arena_size, (tuple, list, np.ndarray)):
            diameter = arena_size[0]
        in_field, _ = opexebo.general.circular_mask(bin_edges, diameter)
        coverage = np.count_nonzero(occupancy_map) / (np.sum(in_field))
        coverage = min(1.0, coverage)
        # Due to the thresholding, coverage might be calculated to be  > 1
        # In this case, cut off to a maximum value of 1.
    elif shape.lower() in default.shapes_linear:
        raise NotImplementedError("Spatial Occupancy does not currently"\
                                  " support linear arenas")
    else:
        raise NotImplementedError(f"Arena shape '{shape}' not understood")

    return masked_map, coverage, bin_edges
