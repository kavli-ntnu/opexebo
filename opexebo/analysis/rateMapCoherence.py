"""Provide function for calculating coherence of a Rate Map"""


import numpy as np
from astropy.convolution import convolve


def rate_map_coherence(rate_map_raw):
    '''
    Calculate coherence of a rate map
    
    Coherence is calculated based on RU Muller, JL Kubie "The firing of hippocampal place
    cells predicts the future position of freely moving rats", Journal of Neuroscience, 1 December 1989,
    9(12):4101-4110. The paper doesn't provide information about how to deal with border values
    which do not have 8 well-defined neighbours. This function uses zero-padding technique.
    
    Parameters
    ---
    rate_map _raw       : np.ma.MaskedArray
        Non-smoothed rate map: n x m array where cell value is the firing rate, masked at locations with low occupancy

    
    Returns
    ---
    coherence           : float
        see relevant literature (above)
    See:
    ---
    BNT.+analyses.coherence(map)
    '''
    
    

    
    kernel = np.array([[0.125, 0.125, 0.125],
                      [0.125, 0,     0.125],
                      [0.125, 0.125, 0.125]])
    
    avg_map = convolve(rate_map_raw, kernel, 'fill', fill_value=0)
    avg_map = avg_map.ravel()
    avg_map = np.nan_to_num(avg_map)
    
    rmap = np.copy(rate_map_raw)
    rmap = rmap.ravel()
    rmap = np.nan_to_num(rmap)
    
    coherence = np.corrcoef(avg_map, rmap)[0,1]
    
    return coherence