

import numpy as np
import astropy.stats.circstats as cs
import opexebo.defaults as default

def turningcurvestats(turning_curve, **kwargs):
    """ Calculate statistics about a turning curve
    
    Calculates various statistics for a turning curve.
    1. Mean vector length of a head direction rate map.
    The value will range from 0 to 1. 0 means that there are so much dispersion
    that a mean angle cannot be described. 1 means that all data are
    concentrated at the same direction. Note that 0 does not necessarily
    indicate a uniform distribution.
    Calculation is based on Section 26.4, J.H Zar - Biostatistical Analysis 5th edition,
    see eq. 26.13, 26.14.
    
    Parameters
    ----------
    turning_curve : np.ma.MaskedArray
        Smoothed turning curve of firing rate as a function of angle
        Nx1 array
    kwargs
        percentile : float
            Percentile value for the head direction arc calculation
            Arc is between two points with values around
            globalPeak * percentile. Value should be in range [0, 1]
        
    
    Returns
    -------
    tcstat : dict
        'score'         : float
            Score for how strongly modulated by angle the cell is
        'mvl'           : float
            mean vector length
        'stdev'         : float
            Circular standard deviation [degrees]
        'peak_rate'     : float
            ' Peak firing rate  [Hz]
        'mean_rate'     : float
            Mean firing rate [Hz]
        'peak_direction': float
            Direction of peak firing rate [degrees]
        'mean_direction': float
            Direction of mean firing rate [degrees]
        
    
    See also
    --------
    BNT.+analyses.tcStatistics
    opexebo.analysis.turningcurve
    
    Copyright (C) 2019 by Simon Ball
    
    This program is free software; you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation; either version 3 of the License, or
    (at your option) any later version.
    """
    
    percentile = kwargs.get('percentile', default.hd_percentile)
    ndim = turning_curve.ndim
    if ndim != 1:
        raise ValueError("turning_curve should be a 1D array. You have provided \
                         %d dimensions" % ndim)
    if not 0 <= percentile <= 1:
        raise ValueError("Keyword 'percentile' should be in the range [0, 1]. \
                         You provided  %.2f. " % percentile)
    
    
    
    num_bin = turning_curve.size
    bin_width = 2*np.pi / num_bin
    hb = bin_width/2
    
    
    
    # Calculate the simple values
    tcstat = {}
    tcstat['mean_direction'] = np.degrees(cs.circmean(turning_curve))
    tcstat['stdev'] = np.degrees(np.sqrt(cs.circvar(turning_curve)))
    tcstat['peak_rate'] = np.nanmax(turning_curve)
    tcstat['mean_rate'] = np.nanmean(turning_curve)
    tcstat['peak_direction'] = np.degrees( (0.5+np.argmax(turning_curve)) * bin_width )
    
    
    # Calculate the more complex ones:
    # mvl
    # circ_r : alpha = bin_centres
    # w = turning_curve
    bin_centres = np.linspace(hb, (2*np.pi)-hb, num_bin)
    mvl = np.sum(turning_curve * np.exp(1j*bin_centres))
    mvl = np.abs(mvl)/np.sum(turning_curve)    
    tcstat['mvl'] = mvl
    
    # Direction score
    tcstat['score'] = np.nan
    # TODO!
    
    return tcstat