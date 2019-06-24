""" Provides functions for calculating Speed Score """

import numpy as np
import opexebo.defaults as default
import opexebo


def speed_score(spike_times, tracking_speeds, **kwargs):
    '''
    Calculate Speed score.
    
    STATUS : EXPERIMENTAL

    Speed score is a correlation between cell firing *rate* and animal speed. 
    The Python version is based on BNT.+scripts.speedScore. At Edvard's request, 
    both the 2015 and 2016 scores are calculated. The primary difference is how 
    the speed smoothing is implemented

    Speed score originates in the following paper in Nature from Emiliano et al
    doi:10.1038/nature14622

    The original Matlab script implemented - but as far as I can tell, did not 
    (by default) use, a Kalman filter for smoothing the animal speed. Since it is 
    not the default behaviour, I have not (yet) added that Kalman filter to opexebo.
    Its addition is contingent on the score similarity to BNT. 

    Discussion on Python equivalent to matlab corr() is found here
    https://stackoverflow.com/questions/16698811/what-is-the-difference-between-matlab-octave-corr-and-python-numpy-correlate

    Parameters
    ----------
    spike_times : np.ndarray
        Nx1 array listing the times at which spikes occurred. [s]

    tracking_speeds : np.ndarray
        Nx2 array [time, speed] calculated from each tracking frame
        tracking_speeds[0, :] = time_stamps
        tracking_speeds[1, :] = speeds
        This matches the grouping tracking_speeds = np.array([time_stamps, speeds])

    kwargs:
        'bin_width' : float
            Width of bins for calculating speed tuning (histogram of speeds)
        'sigma' : float
            Gaussian width for smoothing firing rate, default 0.4 [s]
        'lower_bound_speed' : float
            Speed in [cm/s] used as the lower edge of the speed bandpass filter
        upper_bound_time : float
            Duration in [s] used for determining the upper edge of the speed 
            bandpass filter
        'debug' : bool

    Returns
    -------
    scores : dict
        '2015' : float
        '2016' : float
            Variations on the speed score. '2015' is based on the code in the paper
            above, but additionally including an upper speed filter
            '2016' is a modification involving a slightly different approach to 
            smoothing the firing rate data

    See also
    --------
    BNT.+scripts.speedScore
        
    Copyright (C) 2019 by Simon Ball

    This program is free software; you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation; either version 3 of the License, or
    (at your option) any later version.
    '''
    # Check that the provided functions have correct dimensions
    stn = spike_times.ndim
    if stn != 1:
        raise ValueError("Spike Times must be an Nx1 array. You have provided\
                         %d dimensions" % stn)

    # Get kwargs values
    bin_width = kwargs.get('bin_width', default.bin_speed)
    sigma = kwargs.get('sigma', default.sigma_speed)
    upper_bound_time = kwargs.get('upper_bound_time', default.upper_bound_time)
    lower_bound_speed = kwargs.get('lower_bound_speed', default.lower_bound_speed)
    debug = kwargs.get('debug', False)

    tracking_speeds = np.ma.masked_invalid(tracking_speeds, copy=True)
    spike_times = np.ma.masked_invalid(spike_times, copy=True)

    t_times = tracking_speeds[0,:]
    t_speeds = tracking_speeds[1,:]
    
    # Convert spike_times to spike firing rate
    firing_rate = _spiketimes_to_spikerate(spike_times, t_times)
    firing_rate_smoothed = opexebo.general.smooth(firing_rate, sigma)
    
    # Calculate the speed tuning curve. 
    # This is only used to determine the upper edge of the speed bandpass filter. 
    # Note, because we have to relate this back to time spent, this must use the 
    # speeds from tracking frames, which ahs a consistent sampling rate
    # The spike-speeds does NOT have a consistent sampling rate, and therefore
    # cannot be used to calculate the time spent at a given speed.     
    num_bins = int(( np.nanmax(t_speeds) - np.nanmin(t_speeds)) / bin_width) + 1
    range_bins = ( np.min(t_speeds), np.min(t_speeds) + (num_bins*bin_width) )

    hist, bin_edges = np.histogram(t_speeds, bins=num_bins, range=range_bins)

    # Calculate the upper bandpass edge from the speed tunng curve
    # The upper bound is chosen as the centre of the final (i.e. fastest) bin 
    # in which the animal spends at least upper_bound_time
    sampling_rate = 1 / np.min(np.diff(t_times))    
    upper_bound_samples = upper_bound_time * sampling_rate
    index = np.max(np.where(hist>upper_bound_samples))
    upper_bound_speed = bin_edges[index] + (bin_width/2)

    if debug:
        print(index)
        print(lower_bound_speed)
        print(upper_bound_speed)
        
    good_speeds = (t_speeds < upper_bound_speed) & (t_speeds > lower_bound_speed)
    bad_speeds = np.invert(good_speeds)

    # Score 2016: apply bandpass filter to already-smoothed rate and then correlate
    filtered_speeds = t_speeds[good_speeds]
    filtered_rate = firing_rate_smoothed[good_speeds]
    speed_score_2016 = np.corrcoef(filtered_speeds, filtered_rate.T, rowvar=0)[0,1]

    # Score 2015: Filter rates first (by setting to NaN), and then smooth and correlate
    # Reuse the same filtered_speeds as for 2016, but redefine filtered_rate
    aux_rate = firing_rate[:]
    aux_rate[bad_speeds] = np.nan
    filtered_rate = opexebo.general.smooth(aux_rate, sigma)
    filtered_rate = filtered_rate[good_speeds]
    speed_score_2015 = np.corrcoef(filtered_speeds, filtered_rate.T, rowvar=0)[0,1]

    scores = {'2015': speed_score_2015, '2016': speed_score_2016}
    return scores


def _spiketimes_to_spikerate(spike_times, tracking_times):
    '''Convert a list of spike times to a list of spike rates
    parameters
    ----------
    spike_times : np.ndarray
        Nx1 array of times at which spikes occur in [s]
    tracking times : np.ndarray
        Nx1 array of times at which tracking information is known - e.g. time 
        stamp of camera frames. Also in [s]

    returns
    -------
    spike_rate : np.adarray
        Nx1 array of spike rate [Hz], with the i'th value being the spike rate 
        during the i'th tracking frame.
    '''
    frame_length = np.min(np.diff(tracking_times))
    bin_edges = np.append(tracking_times, tracking_times[-1]+frame_length)

    spikes_per_frame, be = np.histogram(spike_times, bin_edges)
    spike_rate = spikes_per_frame / np.append(np.diff(tracking_times), frame_length)

    return spike_rate
