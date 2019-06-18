'''
Provides function to calculate the spatial occupancy of the arena
'''
import numpy as np
import opexebo
from opexebo import defaults as default



def spatial_occupancy(positions, speed, **kwargs):
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
            For a square arena, length or (x_length, y_length)
            For a non-square rectangle, (x_length, y_length)
            In this function, a circle and a square are treated identically.
        bin_width       : float. 
            Bin size (in cm). Bins are always assumed square default 2.5 cm.
        speed_cutoff    : float. 
            Timestamps with instantaneous speed beneath this value are ignored. 
            Default 0
        limits : tuple or np.ndarray
            (x_min, x_max) or (x_min, x_max, y_min, y_max)
            Provide concrete limits to the range over which the histogram searches
            Any observations outside these limits are discarded
            If no limits are provided, then use np.nanmin(data), np.nanmax(data)
            to generate default limits. 
            As is standard in python, acceptable values include the lower bound
            and exclude the upper bound
        debug           : bool
            If true, print out debugging information throughout the function.
            Default False
    
    Returns
    -------
    masked_map : np.ma.MaskedArray
        Unsmoothed map of time the animal spent in each bin.
        Bins which the animal never visited are masked (i.e. the mask value is 
        True at these locations)
    coverage : float
        Fraction of the bins that the animal visited. In range [0, 1]
    bin_edges : list-like
        x, or (x, y), where x, y are 1d np.ndarrays
        Here x, y correspond to the output histogram
    
    
    See also
    --------
    BNT.+analyses.map
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
        raise KeyError("Arena dimensions not provided. Please provide dimensions using keyword 'arena_size'.")
    
    
    # Handle NaN positions by converting to a Masked Array
    positions = np.ma.masked_invalid(positions)
    

    speed_cutoff = kwargs.get("speed_cutoff", default.speed_cutoff)
    debug = kwargs.get("debug", False)
    
    time_stamps = positions[0,:]
    speeds = speed[1,:]
    
    if debug:
        print("Number of time stamps: %d" % len(time_stamps))
        print("Maximum time stamp value: %.2f" % time_stamps[-1])
        print("Time stamp delta: %f" % np.min(np.diff(time_stamps)))
   

    good = speeds>speed_cutoff
    x = positions[1,:][good]
    y = positions[2,:][good]
    pos = np.array([x,y])
    
    occupancy_map, bin_edges = opexebo.general.accumulatespatial(pos, **kwargs)
    if debug:
        print("Frames included in histogram: %d (%.3f)" % (np.sum(occupancy_map), np.sum(occupancy_map)/len(time_stamps)) )

    # So far, times are expressed in units of tracking frames
    # Convert to seconds:
    frame_duration = np.min(np.diff(time_stamps))    
    occupancy_map_time = occupancy_map * frame_duration
    
    if debug:
        print("Time length included in histogram: %.2f (%.3f)" % (np.sum(occupancy_map_time), np.sum(occupancy_map_time)/time_stamps[-1]) )
    

    masked_map = np.ma.masked_where(occupancy_map < 0.001, occupancy_map_time)
    
    # Calculate the fractional coverage based on the mask. Since the mask is 
    # False where the animal HAS gone, invert it first (just for this calculation)
    coverage = np.sum(np.logical_not(masked_map.mask)) / masked_map.size
    
    return masked_map, coverage, bin_edges
    
    
     
