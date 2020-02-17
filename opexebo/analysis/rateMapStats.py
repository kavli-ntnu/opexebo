"""Provide function for calculating statistics of a Rate Map"""

import numpy as np


def rate_map_stats(rate_map, time_map, debug=False):
    '''
    Calculate statistics of a rate map that depend on probability distribution
    function (PDF)
    
    Calculates information, sparsity and selectivity of a rate map. Calculations
    are done according to 1993 Skaggs et al. "An Information-Theoretic Approach
    to Deciphering the Hippocampal Code" paper. Another source of information is
    1996 Skaggs et al. paper called "Theta phase precession in hippocampal
    neuronal populations and the compression of temporal sequences".

    Coherence is calculated based on RU Muller, JL Kubie "The firing of
    hippocampal place cells predicts the future position of freely moving rats",
    Journal of Neuroscience, 1 December 1989, 9(12):4101-4110. The paper doesn't
    provide information about how to deal with border values which do not have
    8 well-defined neighbours. This function uses zero-padding technique.

    Parameters
    ----------
    rate_map: np.ma.MaskedArray
        Smoothed rate map: n x m array where cell value is the firing rate,
        masked at locations with low occupancy

    time_map: np.ma.MaskedArray
        time map: n x m array where the cell value is the time the animal spent
        in each cell, masked at locations with low occupancy
        Already smoothed

    Returns
    -------
    rms: dict
        spatial_information_rate: float
            information rate [bits/sec]
        spatial_information_content: float
            spatial information content [bits/spike]
        sparsity: float
            see relevant literature (above)
        selectivity: float
            see relevant literature (above)
        peak_rate: float
            peak firing rate of smoothed map [Hz]
        mean_rate: float
            mean firing rate of smoothed map [Hz]

    Notes
    -----
    BNT.+analyses.mapStatsPDF(map)
    
    BNT.+analyses.coherence(map)

    Copyright (C) 2019 by Simon Ball
    '''

    if type(rate_map) != np.ma.MaskedArray:
        rate_map = np.ma.masked_invalid(rate_map, copy=True)
    if type(time_map) != np.ma.MaskedArray:
        time_map = np.ma.masked_invalid(time_map, copy=True)

    duration = np.ma.sum(time_map)
    position_PDF = time_map / (duration + np.spacing(1)) 
    # Probability distribution of where the animal spent its time. 

    if debug:
        print("Duration = %ds" % duration)
        print("Masked locations: %d" % np.sum(rate_map.mask))
        print("Masked values: %s" % rate_map.data[rate_map.mask])

    sparsity = np.nan
    selectivity = np.nan
    inf_rate = np.nan
    inf_content = np.nan
    if rate_map.mask.all():
        # Currently, there is a bug in numpy that causes nanmean to fail
        # on fully masked arrays. This should be a pretty rare occurrence, though!
        rmap_mean = np.nan
        rmap_peak = np.nan
    else:
        rmap_mean = np.nanmean(rate_map)
        rmap_peak = np.nanmax(rate_map)

    mean_rate = np.ma.sum( rate_map * position_PDF )
    mean_rate_sq = np.ma.sum( np.ma.power(rate_map, 2) * position_PDF )

    max_rate = np.max(rate_map)

    if debug:
        print("mean rate: %.2fHz" % mean_rate)
        print("mean rate squared: %.2fHz^2" % mean_rate_sq)
        print("max rate: %.2fHz" % max_rate)


    if mean_rate_sq != 0:
        sparsity = mean_rate * mean_rate / mean_rate_sq

    if mean_rate != 0:
        selectivity = max_rate / mean_rate

        log_argument = rate_map / mean_rate
        log_argument[log_argument < 1] = 1
        if debug:
            print(log_argument.shape)
            #print("log argument: %.4f" % log_argument)
        inf_rate = np.ma.sum(position_PDF * rate_map * np.ma.log2(log_argument))
        inf_content = inf_rate / mean_rate



    return {"spatial_information_rate":inf_rate, "spatial_information_content":inf_content,
            "sparsity":sparsity, "selectivity":selectivity, "peak_rate":rmap_peak,
            "mean_rate":rmap_mean}
