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
            Timestamps with instantaneous speed exceeding this value are ignored. 
            Default infinity

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

    
    # Get default kwargs values

    
    bin_width = kwargs.get("bin_width", default.bin_width)
    arena_size = kwargs.get("arena_size")
    speed_cutoff = kwargs.get("speed_cutoff", default.speed_cutoff)
    
    # Handle the distinctions with arena sizes: 1d or 2d, different kinds of 2d
    # If a single arena dimensions is provided, it treats the arena as either circular/square (len x len)
    # or linear (len x 1).. In both cases, num_bins is an int
    # If two arena dimensions are provided, it treats it as a rectangle (len1 x len2)
    # in this case, num_bins is an array.
    arena_size, is_2d = validatekeyword__arena_size(arena_size, dimensionality-1)
    num_bins = np.ceil(arena_size / bin_width)
    
    # Calculate occupancy
    # This takes a histogram of frame positions: each position gets the number of frames found there
    # The below "range" approach means that it is assumed that the bounding box 
    # of the animal's movements is equal to the arena walls
    time_stamps = positions[0,:]
    x = positions[1,:]
    speeds = speed[1,:]
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
    occupancy_map = np.array(occupancy_map, dtype=float)
    
    # So far, times are expressed in units of tracking frames
    # Convert to seconds:
    frame_duration = np.min(np.diff(time_stamps))    
    occupancy_map_time = occupancy_map / frame_duration
    

    masked_map = np.ma.masked_where(occupancy_map == 0, occupancy_map_time)
    
    return masked_map
    
    
     
