"""Provides a function for calculating a Rate map from an Occupancy Map and positioned Spike Times"""

import numpy as np
from opexebo.general import validatekeyword__arena_size
import opexebo.defaults as default

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
        bin_width
        arena_size

        
    
    Returns
    -------
    rate_map : np.
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
        
    # Get kwargs values
    bin_width = kwargs.get("bin_width", default.bin_width)
    arena_size = kwargs.get("arena_size")
    
    arena_size, is_2d = validatekeyword__arena_size(arena_size, dims_p)
    num_bins = np.ceil(arena_size / bin_width)
    
    
    

    
    # Histogram of spike positions
    spike_x = spikes[1,:]
    if is_2d:
        spike_y = spikes[2,:]
        
        range_2Dhist = [ [np.min(spike_x), np.max(spike_x)],
                    [np.min(spike_y), np.max(spike_y)] ]
        spike_map, xedges_t, yedges_t = np.histogram2d(spike_x, spike_y,
                                       bins=num_bins, range=range_2Dhist)
    else:
        range_1Dhist = [np.min(spike_x), np.max(spike_x)]
        spike_map, xedges_t = np.histogram(spike_x, bins=num_bins, range=range_1Dhist)
    spike_map = np.array(spike_map, dtype=float)
        
    if spike_map.shape != occupancy_map.shape:
        raise ValueError("Rate Map and Occupancy Map must have the same dimensions. Provided: %s, %s" % (spike_map.shape, occupancy_map.shape))
    # Convert to rate map
    rate_map = spike_map / occupancy_map
    
    return rate_map
    
    