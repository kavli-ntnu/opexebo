import numpy as np
import matplotlib.pyplot as plt
import astropy.stats.circstats as cs
import opexebo.defaults as default


def tuning_curve_stats(tuning_curve, **kwargs):
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
    tuning_curve : np.ma.MaskedArray
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
        'peak_direction_deg' : float
        'peak_direction_rad' : float
            Direction of peak firing rate
        'mean_direction_deg': float
        'mean_direction_rad': float
            Direction of mean firing rate [degrees]
        'halfCwInd'  : int
        'halfCcwInd' : int
            Indicies of at the start, end of the range defined by percentile
            (Clockwise, counter-clockwise).
        'halfCwRad' : float
        'halfCcwRad'  : float
            Angle of the start, end of the range defined by percentile
        'arc_angle_rad' : float
        'arc_angle_rad' : float
            Angle of the arc defined by percentile


    See also
    --------
    BNT.+analyses.tcStatistics
    opexebo.analysis.tuningcurve

    Copyright (C) 2019 by Simon Ball

    This program is free software; you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation; either version 3 of the License, or
    (at your option) any later version.
    """

    debug = kwargs.get("debug", False)
    percentile = kwargs.get('percentile', default.hd_percentile)
    ndim = tuning_curve.ndim
    if ndim != 1:
        raise ValueError("tuning_curve should be a 1D array. You have provided\
                         %d dimensions" % ndim)
    if not 0 <= percentile <= 1:
        raise ValueError("Keyword 'percentile' should be in the range [0, 1]. \
                         You provided  %.2f. " % percentile)

    num_bin = tuning_curve.size
    bin_width = 2 * np.pi / num_bin
    hb = bin_width / 2

    if debug:
        print("Num_bin: %d" % num_bin)
        print("Therefore, bin_width = %.3g deg = %.3g rad"
              % (np.degrees(bin_width), bin_width))

    # Calculate the simple values
    tcstat = {}
    mean_dir = cs.circmean(tuning_curve)
    peak_dir_index = np.argmax(tuning_curve)
    tcstat['mean_direction_rad'] = mean_dir
    tcstat['peak_direction_rad'] = _index_to_angle(peak_dir_index, bin_width)
    tcstat['mean_direction_deg'] = np.degrees(mean_dir)
    tcstat['peak_direction_deg'] = np.degrees(_index_to_angle(peak_dir_index,
                                                              bin_width))
    tcstat['peak_rate'] = np.nanmax(tuning_curve)
    tcstat['mean_rate'] = np.nanmean(tuning_curve)

    # Calculate the more complex ones:
    # mvl
    bin_centres = np.linspace(hb, (2*np.pi)-hb, num_bin)
    mvl = np.sum(tuning_curve * np.exp(1j*bin_centres))
    mvl = np.abs(mvl)/np.sum(tuning_curve)
    tcstat['mvl'] = mvl

    # stdev
    # Eq. 26.20 from J. H. Zar
    tcstat['stdev'] = np.sqrt(2*(1-mvl))

    # Percentile arc
    peak_index = np.nanargmax(tuning_curve)
    peak = tcstat['peak_rate']
    half_peak = peak * percentile

    # Because Python doesn't natively handle circular arrays, reshape such that
    # the peak rate occurs at the centre of the array - thendon't have to worry
    # about whether the arc goes off one edge of the array or not
    # Must be careful to keep track of the array to which the indicies point
    tuning_curve_re = np.zeros_like(tuning_curve)
    centre_index = int(num_bin / 2)
    offset = centre_index - peak_index
    tuning_curve_re[offset:] = tuning_curve[:-offset]
    tuning_curve_re[:offset] = tuning_curve[-offset:]
    # A positive offset means that the peak angle was in the range [0, pi], and
    # is now at the central index. Therefore, to get the "proper" index,
    # subtract offset from index in tuning_curve_re

    if debug:
        print("Centre index: %d, value" % centre_index)
        print("Peak index: %d" % peak_index)
        print("Offset: %d" % offset)

    # Clockwise and counter-clockwise edges of arc around peak defined by
    # percentile. ccw index +1 to account for width of central peak
    cw_hp_index = np.where(tuning_curve_re >= (half_peak))[0][0] - offset
    ccw_hp_index = np.where(tuning_curve_re >= (half_peak))[0][-1] - offset+1

    cw_hp_ang = _index_to_angle(cw_hp_index, bin_width)
    ccw_hp_ang = _index_to_angle(ccw_hp_index, bin_width)
    arc_angle = ccw_hp_ang - cw_hp_ang

    if debug:
        print("CW: %d, %.3g rad" % (cw_hp_index, cw_hp_ang))
        print("CCW: %d, %.3g rad" % (ccw_hp_index, ccw_hp_ang))
        print("Arc: %.3g rad" % arc_angle)

    score = 1 - (arc_angle / np.pi)
    tcstat['halfCwInd'] = cw_hp_index
    tcstat['halfCcwInd'] = ccw_hp_index
    tcstat['halfCwRad'] = cw_hp_ang
    tcstat['halfCcwRad'] = ccw_hp_ang
    tcstat['arc_angle_rad'] = arc_angle
    tcstat['arc_angle_deg'] = np.degrees(arc_angle)
    tcstat['score'] = score

    return tcstat


def _index_to_angle(index, bin_width):
    '''Given an index in the turning curve, return the angle at the centre of
    the bin that it indexes. The angle is deliberately not wrapped into [0,2pi]
    '''
    return (0.5 + index) * bin_width


if __name__=='__main__':
    plt.close('all')
    samples = 100
    x0 = -0.5
    w = 0.5
    x = np.linspace(0, 2*np.pi * (samples-1)/samples, samples)
    tc = np.exp(-1*np.square(x-x0)/w) + np.exp(-1*np.square(x-x0-(2*np.pi))/w)

    perc = 0.8
    stats = tuningcurvestats(tc, percentile=perc, debug=True)
    print(stats['score'])
