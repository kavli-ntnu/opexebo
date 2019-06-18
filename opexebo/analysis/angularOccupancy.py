'''
Provides function to calculate the angular occupancy, i.e. the frequency with 
which the animal looked in each direction

Closely based on code from Horst's initial imaging pipeline, tidied up and
pythonised by Simon

(C) 2019 Horst Oberhaus, Simon Ball
'''
import numpy as np
import opexebo
import opexebo.defaults as default


def angular_occupancy(angle,**kwargs):
    '''
    Calculate angular occupancy from tracking angle and kwargs over (0,2*pi)

    Parameters
    ----------
    angle : numpy array
        Head angle in radians
        Nx1 array
    **kwargs : 
        bin_width : float
            Width of histogram bin in degrees

    Returns
    -------
    masked_histogram : numpy masked array
        Angular histogram, masked at angles at which the animal was never 
        observed. Masked means that the mask value is True
    coverage : float
        Fraction of the bins that the animal visited. In range [0, 1]
    bin_edges : list-like
        x, or (x, y), where x, y are 1d np.ndarrays
        Here x, y correspond to the output histogram
    '''
    ndim = angle.ndim
    if ndim != 1:
        raise ValueError("angle must be provided as a 1D array. You provided %d \
                         dimensions" % ndim)
    if np.nanmax(angle) > 2*np.pi:
        raise Warning("Angles greater than 2pi detected. Please check that your \
                      angle array is in radians. If it is in degrees, you can \
                      convert with 'np.radians(array)'")

    bin_width = kwargs.get('bin_width', default.bin_angle)
    bin_width = np.radians(bin_width)
    arena_size = 2*np.pi
    limits = (0, arena_size)

    angle_histogram, bin_edges = opexebo.general.accumulate_spatial(angle, bin_width=bin_width, 
                                                arena_size=arena_size, limits=limits)
    masked_angle_histogram = np.ma.masked_where(angle_histogram==0, angle_histogram)
    
    # Calculate the fractional coverage based on the mask. Since the mask is 
    # False where the animal HAS gone, invert it first (just for this calculation)
    coverage = np.sum(np.logical_not(masked_angle_histogram.mask)) / masked_angle_histogram.size
    
    return masked_angle_histogram, coverage, bin_edges
