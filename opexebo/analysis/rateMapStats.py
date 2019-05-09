"""Provide function for calculating statistics of a Rate Map"""

import numpy as np


def ratemapstats(rate_map, time_map, debug=False):
    '''
    Calculate statistics of a rate map that depend on probability distribution function (PDF)
    
    Calculates information, sparsity and selectivity of a rate map. Calculations are done
    according to 1993 Skaggs et al. "An Information-Theoretic Approach to Deciphering the Hippocampal Code"
    paper. Another source of information is 1996 Skaggs et al. paper called
    "Theta phase precession in hippocampal neuronal populations and the compression of temporal sequences".
    
    Parameters
    ---
    rate_map        : np.ma.MaskedArray
        rate map: n x m array where cell value is the firing rate, masked at locations with low occupancy
    time_map   : np.ma.MaskedArray
        time map: n x m array where the cell value is the time the animal spent in each cell, masked at locations with low occupancy
    
    Returns
    ---
    information_rate    : float
        information rate [bits/sec]
    information_content : float
        spatial information content [bits/spike]
    sparsity            : float
        see relevant literature (above)
    selectivity         : float
        see relevant literature (above)
     
    See:
    ---
    BNT.+analyses.mapStatsPDF(map)
    '''
     
    rm = rate_map.copy()
    duration = np.ma.sum(time_map)
    position_PDF = time_map / (duration + np.spacing(1)) # Probability distribution of where the animal spent its time. 
    
    if debug:
        print("Duration = %ds" % duration)
        print("Masked locations: %d" % np.sum(rate_map.mask))
        print("Masked values: %s" % rate_map.data[rate_map.mask])

    sparsity = np.nan
    selectivity = np.nan
    information_rate = np.nan
    information_content = np.nan
    
    mean_rate = np.ma.sum( rate_map * position_PDF )
    mean_rate_sq = np.ma.sum( np.ma.power(rate_map, 2) * position_PDF )
    
    max_rate = np.max(rate_map)
    
    if debug:
        print("mean rate: %.2fHz" % mean_rate)
        print("mean rate squared: %.2fHz^2" % mean_rate_sq)
        print("max rate: %.2fHz" % max_rate)
        
        rm.shape = rm.size
        print(np.sort(rm))
    
    if mean_rate_sq != 0:
        sparsity = mean_rate * mean_rate / mean_rate_sq
    
    if mean_rate != 0:
        selectivity = max_rate / mean_rate
        
        log_argument = rate_map / mean_rate
        log_argument[log_argument < 1] = 1
        if debug:
            print(log_argument.shape)
            #print("log argument: %.4f" % log_argument)
        information_rate = np.ma.sum(position_PDF * rate_map * np.ma.log2(log_argument))
        information_content = information_rate / mean_rate
    
    
    return information_rate, information_content, sparsity, selectivity