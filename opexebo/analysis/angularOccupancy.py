'''
Provides function to calculate the angular occupancy, i.e. the frequency with 
which the animal looked in each direction
'''
import numpy as np
import opexebo
import opexebo.defaults as default


def angular_occupancy(time, angle, **kwargs):
    '''
    Calculate angular occupancy from tracking angle and kwargs over (0,2*pi)

    Parameters
    ----------
    time : numpy.ndarray
        time stamps of angles in seconds
    angle : numpy array
        Head angle in radians
        Nx1 array
    bin_width : float, optional
        Width of histogram bin in degrees

    Returns
    -------
    masked_histogram : numpy masked array
        Angular histogram, masked at angles at which the animal was never 
        observed. A mask value of True means that the animal never occupied
        that angle. 
    coverage : float
        Fraction of the bins that the animal visited. In range [0, 1]
    bin_edges : list-like
        x, or (x, y), where x, y are 1d np.ndarrays
        Here x, y correspond to the output histogram
    
    Notes
    --------
    Copyright (C) 2019 by Simon Ball, Horst Obenhaus

    This program is free software; you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation; either version 3 of the License, or
    (at your option) any later version.
    '''
    if time.ndim != 1:
        raise ValueError("time must be provided as a 1D array. You provided %d"\
                         " dimensions" % time.ndim)
    if angle.ndim != 1:
        raise ValueError("angle must be provided as a 1D array. You provided %d"\
                         " dimensions" % angle.ndim)
    if time.size != angle.size:
        raise ValueError("Arrays 'time' and 'angle' must have the same number"\
                         f" of elements. You provided {time.size} and {angle.size}")
    if np.nanmax(angle) > 2*np.pi:
        raise Warning("Angles greater than 2pi detected. Please check that your"\
                      " angle array is in radians. If it is in degrees, you can"\
                      " convert with 'np.radians(array)'")

    bin_width = kwargs.get('bin_width', default.bin_angle)
    bin_width = np.radians(bin_width)
    arena_size = 2*np.pi
    limits = (0, arena_size)

    angle_histogram, bin_edges = opexebo.general.accumulate_spatial(angle, bin_width=bin_width, 
                                                arena_size=arena_size, limits=limits)
    masked_angle_histogram = np.ma.masked_where(angle_histogram==0, angle_histogram)
    
    # masked_angle_histogram is in units of frames. It needs to be converted to units of seconds
    frame_duration = np.mean(np.diff(time))
    masked_angle_seconds = masked_angle_histogram * frame_duration
    
    # Calculate the fractional coverage based on locations where the histogram
    # is zero. If all locations are  non-zero, then coverage is 1.0
    coverage = np.count_nonzero(angle_histogram) / masked_angle_seconds.size
    
    return masked_angle_seconds, coverage, bin_edges
