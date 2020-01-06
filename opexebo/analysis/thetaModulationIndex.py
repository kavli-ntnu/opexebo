# -*- coding: utf-8 -*-
"""
Created on Wed Dec 18 11:14:37 2019

@author: simoba
"""

import numpy as np

def theta_modulation_index(spike_times, **kwargs):
    '''
    Calculate the 1-dimensional autocorrelation of the spike train and use it
    as a proxy score for how how reliably a cell fires at the same phase of
    the theta frequency.
    
    The measure is calculated as described in publication (with small modifications)
    Cacucci et al, Theta-Modulated Place-by-Direction Cellsin the Hippocampal
    Formation in the Rat, Journal of Neuroscience, 2004.
    doi: 10.1523/jneurosci.2635-04.2004
    
    The calculation is very similar in practice, although somewhat less 
    sophisicated, than the standard g(2) autocorrelation calculation.
    
    For each spike, consider all spikes that follow within some of length P
        Calculate the time delta to all spikes within P and bin as a histogram
    Convert from number of occurrences to rate of occurrences (optional)
    Calculate Index as the contrast of the autocorrelation at Tau=(50,70ms) to
    Tau=(100,140ms) (assumed theta trough and peak respectively)
    
    This indicator fails in two key areas:
        * It indicates nothing about the phase of theta, i.e. this is a scalar rather than vector calculation
        * It _assumes_ the value of theta for the animal in question, rather than calculating it directly
    
    Parameters
    ----------
    spike_times : np.ndarray
        Array of times at which spikes for a cell are detected, in [seconds]
    
    Returns
    -------
    theta_modulation_index : float
        Contrast of autocorrelation
    hist : np.ndarray
        Histogram of time_delta to subsequent spikes within the next 0.5s
    bins : np.ndarray
        Bin edges of the histogram in [s]
    
    '''
    
    if spike_times.ndim != 1:
        raise ValueError("spike_times must be a 1D array")
    
    
    # We start with 1D array spike_times
    # The neatest solution would be to calculate a 2D array of time_deltas, and
    # then take a 1D histogram of that cutoff at tau=0.
    # However, that will rapidly run into memory constraints, so we probably
    # need to iterate over each spike in turn
    
    lower = 0
    upper = 0.5
    bin_size = 5e-3
    bins = np.linspace(lower, upper, int(1/bin_size))
    hist = np.zeros(bins.size-1)
    
    for i, t in enumerate(spike_times):
        # Identify the section of spike_times that is relevant to us here, i.e.  i:j
        try:
            j = i+np.where(spike_times[i:] > t+upper)[0][0]
            time_delta = spike_times[i+1:j] - t
            
            hist_i, _ = np.histogram(time_delta, bins=bins, range=(lower, upper))
            hist += hist_i
        except IndexError:
            # When we are within <upper> seconds of the end of the recording,
            # the above will fail (because j will be meaningless)
            # In any case, we lose information in the final, tiny section
            # So discard it completely
            break
        
    # hist is now the sum of all time_deltas (i.e. tau) in the range (lower, upper)
    # theta_modulation_index is defined as the contrast between two specific, 
    # arbitrary, time windows
    # Arbitrary values taken from the paper cited above. 
    
    arbitrary_trough = np.where((50e-3 <= bins) & (bins < 70e-3))
    arbitrary_peak = np.where((100e-3 <= bins) & (bins < 140e-3))
    
    contrast = ( np.sum(hist[arbitrary_peak]) - np.sum(hist[arbitrary_trough]) ) / ( np.sum(hist[arbitrary_peak]) + np.sum(hist[arbitrary_trough]) )
    
    return contrast, hist, bins