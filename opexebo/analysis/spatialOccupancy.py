'''
Provides function to calculate the spatial occupancy of the arena
'''
import numpy as np
from scipy.ndimage.filters import gaussian_filter

def spatialOccupancy(positions, speed, **kwargs):
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
        sigma           : float.  
            Parameter for gaussian smoothing, in units of bins. Default 1
    '''
    # Check for correct shapes. 
    dimensionality, num_samples = positions.shape
    if dimensionality not in (2, 3):
        raise ValueError("Positions array has the wrong number of columns (%d)" % dimensionality)
    if speed.shape[1] != num_samples:
        raise ValueError("Speed array does not have the same number of samples as Positions")
    if speed.shape[0] != 2:
        raise ValueError("Speed array has the wrong number of columns")

    
    # Get default kwargs values
    default_bin_width = 2.5
    default_arena_size = 80
    default_speed_cutoff = np.inf
    default_sigma = 1
    
    bin_width = kwargs.get("bin_width", default_bin_width)
    arena_size = kwargs.get("arena_size", default_arena_size)
    speed_cutoff = kwargs.get("speed_cutoff", default_speed_cutoff)
    sigma = kwargs.get("sigma", default_sigma)
    
    # Handle the distinctions with arena sizes: 1d or 2d, different kinds of 2d
    # If a single arena dimensions is provided, it treats the arena as either circular/square (len x len)
    # or linear (len x 1).. In both cases, num_bins is an int
    # If two arena dimensions are provided, it treats it as a rectangle (len1 x len2)
    # in this case, num_bins is an array.
    is_2d = bool(dimensionality - 2) # i.e. if positions has 2 columns [t,x], this will evaluate to False
    if type(arena_size) in (list, tuple, np.ndarray):
        if len(arena_size) == 1:
            arena_size = float(arena_size)            
        elif len(arena_size) > 2:
            raise ValueError("Keyword 'arena_size' value is invalid. Provide either a float or a 2-element tuple")
        elif len(arena_size) == 2 and not is_2d:
            raise ValueError("Mismatch in dimensions: 1d position data but 2d arena specified")
        else:
            arena_size = np.array(arena_size)
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
    
    # Apply smoothing
    occupancy_map_time = gaussian_filter(occupancy_map_time, sigma=sigma, mode='nearest')
    masked_map = np.masked_where(occupancy_map == 0, occupancy_map_time)
    
    return masked_map
    
    
     
    
