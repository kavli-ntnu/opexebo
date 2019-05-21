'''
Provides function to calculate the spatial occupancy of the arena
'''
import numpy as np
from opexebo import defaults as default
from opexebo.general import validatekeyword__arena_size


def spatialoccupancy(positions, speed, **kwargs):
    '''
    Generate an occpuancy map: how much time the animal spent in each location
    in the arena.    
    
    NOTES: This assumes that the positions have already been aligned and curated
    to remove NaNs. This is based on the expectation that it will primarily be 
    used within the DataJoint framework, where the curation takes place at a 
    much earlier stage.
    
    Parameters
    ----------
    positions   : ndarray. 
        [t, x, y] or [t, x]. [0,:] -> timestamps. This matches array creation 
        with positions=np.array([time_stamps, x, y])
    speed       : ndarray. 
        [t, s]
    
    kwargs
        arena_size      : float or tuple of floats. 
            Dimensions of arena (in cm)
            For a linear track, length
            For a circular arena, diameter
            For a square arena, length
            For a non-square rectangle, (length1, length2)
            In this function, a circle and a square are treated identically.
        bin_width       : float. 
            Bin size (in cm). Bins are always assumed square default 2.5 cm.
        speed_cutoff    : float. 
            Timestamps with instantaneous speed beneath this value are ignored. 
            Default 0
        debug           : bool
            If true, print out debugging information throughout the function.
            Default False
    
    Returns
    -------
    masked_map : np.ma.MaskedArray
    
    
    See also
    --------
    BNT.+analyses.map
    
    Testing vs BNT
    --------------
    Given the same list of positions, the resulting 2D array from BNT.+analyses.map.timeRaw
    and the output of spatialoccupancy() are compared:    
        The mean difference between the two arrays, and the mean ratio between the two is evaluated
        Over many possible sessions, the meanDiff and meanRatio are tabulated
    Over 100 sessions, a difference of (0.8+-4)e-3 and a ratio of (1+)(2+-5)e-2 is observed
    
    maximum differences do not exceed single digit milliseconds. 

    '''
    # Check for correct shapes. 
    dimensionality, num_samples = positions.shape
    if dimensionality not in (2, 3):
        raise ValueError("Positions array has the wrong number of columns (%d)" % dimensionality)
    if speed.shape[1] != num_samples:
        raise ValueError("Speed array does not have the same number of samples as Positions")
    if speed.shape[0] != 2:
        raise ValueError("Speed array has the wrong number of columns")
    if "arena_size" not in kwargs:
        raise ValueError("Arena dimensions not provided. Please provide dimensions using keyword 'arena_size'.")
    
    
    # Handle NaN positions by converting to a Masked Array
    positions = np.ma.masked_invalid(positions)
    
    # Get default kwargs values
    bin_width = kwargs.get("bin_width", default.bin_width)
    arena_size = kwargs.get("arena_size")
    speed_cutoff = kwargs.get("speed_cutoff", default.speed_cutoff)
    debug = kwargs.get("debug", False)
    
    # Handle the distinctions with arena sizes: 1d or 2d, different kinds of 2d
    # If a single arena dimensions is provided, it treats the arena as either circular/square (len x len)
    # or linear (len x 1).. In both cases, num_bins is an int
    # If two arena dimensions are provided, it treats it as a rectangle (len1 x len2)
    # in this case, num_bins is an array.
    arena_size, is_2d = validatekeyword__arena_size(arena_size, dimensionality-1)
    num_bins = np.ceil(arena_size / bin_width)
    
    if debug:
        print("Bin width: %.2f" % bin_width)
        print("Arena size : %s" % str(arena_size))
        print("Bin number : %s" % str(num_bins))
    # Calculate occupancy
    # This takes a histogram of frame positions: each position gets the number of frames found there
    # The below "range" approach means that it is assumed that the bounding box 
    # of the animal's movements is equal to the arena walls
    time_stamps = positions[0,:]
    x = positions[1,:]
    speeds = speed[1,:]
    
    if debug:
        print("Number of time stamps: %d" % len(time_stamps))
        print("Maximum time stamp value: %.2f" % time_stamps[-1])
        print("Time stamp delta: %f" % np.min(np.diff(time_stamps)))
    
    if is_2d:
        y = positions[2,:]
        
        range_2Dhist = [ [np.min(x), np.max(x)],
                    [np.min(y), np.max(y)] ]
        occupancy_map, xedges_t, yedges_t = np.histogram2d(x[(speeds>speed_cutoff)],
                                       y[(speed[1]>speed_cutoff)],
                                       bins=num_bins, range=range_2Dhist)
    else:
        range_1Dhist = [np.min(x), np.max(x)]
        occupancy_map, xedges_t = np.histogram(x[(speeds>speed_cutoff)],
                                       bins=num_bins, range=range_1Dhist)
    occupancy_map = np.array(occupancy_map, dtype=int)
    
    # Transpose the map to match the expected directions provided by BNT
    occupancy_map = np.transpose(occupancy_map)
    
    if debug:
        print("Frames included in histogram: %d (%.3f)" % (np.sum(occupancy_map), np.sum(occupancy_map)/len(time_stamps)) )
    
    # So far, times are expressed in units of tracking frames
    # Convert to seconds:
    frame_duration = np.min(np.diff(time_stamps))    
    occupancy_map_time = occupancy_map * frame_duration
    
    if debug:
        print("Time length included in histogram: %.2f (%.3f)" % (np.sum(occupancy_map_time), np.sum(occupancy_map_time)/time_stamps[-1]) )
    

    masked_map = np.ma.masked_where(occupancy_map < 0.001, occupancy_map_time)
    
    return masked_map
    
    
     
