"""Provide a thead-direction turning curve analysis"""

import numpy as np
import opexebo
import opexebo.defaults as default


def tuning_curve(angular_occupancy, spike_angles, **kwargs):
    """Analogous to a RateMap - i.e. mapping spike activity to spatial position
    map spike rate as a function of angle

    Parameters
    ----------
    angular_occupancy : np.ma.MaskedArray
        unsmoothed histogram of time spent at each angular range
        Nx1 array, covering the range [0, 2pi] radians
        Masked at angles of zero occupancy
    spike_angles : np.ndarray
        Mx1 array, where the m'th value is the angle of the animal (in radians)
        associated with the m'th spike
    kwargs
        bin_width : float
            width of histogram bin in DEGREES
            Must match that used in calculating angular_occupancy
            In the case of a non-exact divisor of 360 deg, the bin size will be 
            shrunk to yield an integer bin number. 


    Returns
    -------
    tuning_curve : np.ma.MaskedArray
        unsmoothed array of firing rate as a function of angle
        Nx1 array

    Notes
    --------
    BNT.+analyses.turningcurve

    Copyright (C) 2019 by Simon Ball

    This program is free software; you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation; either version 3 of the License, or
    (at your option) any later version.
    """

    occ_ndim = angular_occupancy.ndim
    spk_ndim = spike_angles.ndim
    if occ_ndim != 1:
        raise ValueError("angular_occupancy must be a 1D array. You provided a"\
                         " %d dimensional array" % occ_ndim)
    if spk_ndim != 1:
        raise ValueError("spike_angles must be a 1D array. You provided a %d"\
                         " dimensional array" % spk_ndim)
    if np.nanmax(spike_angles) > 2*np.pi:
        raise Warning("Angles higher than 2pi detected. Please check that your"\
                      " spike_angle array is in radians. If it is in degrees,"\
                      " you can convert with 'np.radians(array)'")

    bin_width = kwargs.get("bin_width", default.bin_angle) # in degrees
    num_bins = opexebo.general.bin_width_to_bin_number(360., bin_width) # This is for validation ONLY, the value num_bins here is not passed onwards
    if num_bins != angular_occupancy.size:
        raise ValueError("Keyword 'bin_width' must match the value used to"\
                         " generate angular_occupancy")
    #UNITS!!!
    # bin_width and arena_size need to be in the same units. 
    # As it happens, I hardcoded arena_size as 2pi -> convert bin_width to radians
    # limits and spike_angles need to be in the same units
    
    bin_width = np.radians(bin_width)

    spike_histogram, bin_edges = opexebo.general.accumulate_spatial(spike_angles,
                arena_size=2*np.pi, limits=(0, 2*np.pi), bin_width=bin_width)

    tuning_curve = spike_histogram / (angular_occupancy + np.spacing(1))

    return tuning_curve
