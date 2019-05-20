'''
Provides function to calculate the angular occupancy, i.e. the frequency with 
which the animal looked in each direction

Closely based on code from Horst's initial imaging pipeline, tidied up and pythonised by Simon

(C) 2019 Horst Oberhaus, Simon Ball
'''
import numpy as np
import opexebo.defaults as default

def angularoccupancy(head_angle,**kwargs):
    '''
    Calculate angular occupancy from tracking angle and kwargs over (0,2*pi)
    
    Parameters
    ----------
    head_angle : numpy array
        Head angle in degrees
    **kwargs : 
        bin_width : int
            Width of histogram bin in degrees
        sigma_angle : float
            Sigma of Gaussian smoothing kernel
            If this keyword is ommitted, then no smoothing
        
    Returns
    -------
    masked_histogram : numpy masked array
        Angular histogram, masked (where no tracking data) and smoothed (with gaussian kernel)
    '''
    bin_width = kwargs.get('bin_width', default.bin_angle)
    num_bins = int(360. / bin_width)
    


    
    angle_histogram, bins_angle = np.histogram(head_angle, bins = num_bins, range=(0,360))
    angle_histogram = np.array(angle_histogram,dtype=float)
    angle_histogram_unfiltered = angle_histogram
    masked_angle_histogram = np.ma.masked_where(angle_histogram_unfiltered==0, angle_histogram_unfiltered)

    return masked_angle_histogram