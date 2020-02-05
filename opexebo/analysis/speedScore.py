""" Provides functions for calculating Speed Score """

import numpy as np
import opexebo.defaults as default
import opexebo


def speed_score(spike_times, tracking_times, tracking_speeds, **kwargs):
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

    Summary:
        * The intention is to correlate (spike firing rate) with (animal speed)
        * Convert an N-length array of spike firing times into an M-length array
        of spike firing rates, where M is the same length as tracking times
        * Optional: smooth firing rates
        * Optional: smooth speeds
        * Optional: apply a bandpass filter to speeds
        * calculate the Pearson correlation coefficient between (speed), (firing rate)

    Parameters
    ----------
    spike_times : np.ndarray
        N-length array listing the times at which spikes occurred. [s]

    tracking_times : np.ndarray
        M-length array of time stamps of tracking frames

    tracking_speeds : np.ndarray
        M-length array of animal speeds at time stamps given in `tracking_times`

    kwargs:
        bandpass : str
            Type of bandpass filter applied to animal speeds. Acceptable values
            are:
                * "none" - No speed based filtering is applied
                * "fixed" - a fixed lower and upper speed bound are used, based on keywords "lower_bound_speed", "upper_bound_speed"
                * "adaptive" - a fixed lower speed bound is used, based on keyword "lower_bound_speed"
                    An upper speed bound is determined based on keywords "upper_bound_time" and "speed_bandwidth"
        lower_bound_speed' : float
            Speed in [cm/s] used as the lower edge of the speed bandpass filter("fixed" and "adaptive")
        lower_bound_speed' : float
            Speed in [cm/s] used as the upper edge of the speed bandpass filter ("fixed" only)
        upper_bound_time : float
            Duration in [s] used for determining the upper edge of the speed 
            bandpass filter ("adaptive" only)
        speed_bandwidth : float
            Range of speeds in [cm/s] used for determining the upper edge of the
            speed bandpass filter ("adaptive" only)
        sigma: float
            Standard deviation in [s] of Gaussian smoothing kernel for smoothing
            both speed and firing rate data. 
        debug : bool

    Returns
    -------
    scores : dict
        '2015' : float
        '2016' : float
            Variations on the speed score. '2015' is based on the code in the paper
            above, but additionally including an upper speed filter
            '2016' is a modification involving a slightly different approach to 
            smoothing the firing rate data
    (lower_speed, upper_speed): list of floats
        Speed thresholds using in the bandpass filter
        Most useful in the case of the adaptive filter, because there is no
        other way to find out what was actually used. 
        

    See also
    --------
    BNT.+scripts.speedScore
        
    Copyright (C) 2019 by Simon Ball

    This program is free software; you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation; either version 3 of the License, or
    (at your option) any later version.
    '''
    # Check that the provided arrays have correct dimensions
    if spike_times.ndim != 1:
        raise ValueError("spike_times must be an Nx1 array. You have provided"\
                         f" {spike_times.ndim} dimensions")
    elif tracking_times.ndim != 1:
        raise ValueError("tracking_times must be an Nx1 array. You have provided"\
                         f" {tracking_times.ndim} dimensions")
    elif tracking_speeds.ndim != 1:
        raise ValueError("tracking_speeds must be an Nx1 array. You have provided"\
                         f" {tracking_speeds.ndim} dimensions")
    if tracking_times.size != tracking_speeds.size:
        raise ValueError("tracking_times and tracking_speeds must be the same length")
    
    if np.isnan(tracking_speeds).any():
        raise ValueError("tracking_speed cannot have NaN values")

    # Get kwargs values
    speed_bandwidth = kwargs.get('speed_bandwidth', default.speed_bandwidth)
    sigma_time = kwargs.get('sigma', default.sigma_time)
    upper_bound_time = kwargs.get('upper_bound_time', default.upper_bound_time) # Only used in "adaptive" bandpass filter
    lower_bound_speed = kwargs.get('lower_bound_speed', default.lower_bound_speed)
    upper_bound_speed = kwargs.get("upper_bound_speed", default.upper_bound_speed) # Only used in "fixed" bandpass filter
    bandpass_type = kwargs.get("bandpass", "none").lower()
    available_filters = ("none", "fixed", "adaptive")
    if bandpass_type not in available_filters: 
        raise NotImplementedError(f"Bandpass tpye '{bandpass_type}' is not implemented."\
                                  f" Available types are {available_filters}.")
    debug = kwargs.get('debug', False)


    
    # Convert spike_times to spike firing rate   
    sampling_rate = 1 / np.mean(np.diff(tracking_times))
    firing_rate = _spiketimes_to_spikerate(spike_times, tracking_times, sampling_rate)
    
    # Apply smoothing
    # Smoothing expects to be given a sigma in units [bins], so convert from real units to bins
    tracking_speeds_smoothed = opexebo.general.smooth(tracking_speeds, sigma_time * sampling_rate)
    firing_rate_smoothed = opexebo.general.smooth(firing_rate, sigma_time * sampling_rate)

    # Calculate the bandpass filter
    if debug:
        print(bandpass_type)
    if bandpass_type == "none":
        _filter = _bandpass_none(tracking_speeds_smoothed, **kwargs)
        lower_bound_speed = 0
        upper_bound_speed = np.inf
    elif bandpass_type == "fixed":
        _filter = _bandpass_fixed(tracking_speeds_smoothed, lower_bound_speed, 
                                  upper_bound_speed, **kwargs)
    elif bandpass_type == "adaptive":
        _filter, upper_bound_speed = _bandpass_adaptive(tracking_speeds_smoothed, sampling_rate, 
                                     lower_bound_speed, upper_bound_time, speed_bandwidth, **kwargs)
    
    # Apply the filter to speeds
    speeds = tracking_speeds_smoothed[_filter]
        # The filter will be applied differently to rate based on which score version is wanted


    # Score 2016: apply bandpass filter to already-smoothed rate and then correlate
    rate = firing_rate_smoothed[_filter]
    speed_score_2016 = np.corrcoef(speeds, rate)[0, 1]

    # Score 2015: Filter rates first (by setting to NaN), and then smooth and correlate
    # Reuse the same filtered_speeds as for 2016, but redefine filtered_rate
    rate = firing_rate[_filter]
    rate = opexebo.general.smooth(rate, sigma_time * sampling_rate)
    speed_score_2015 = np.corrcoef(speeds, rate)[0,1]

    scores = {'2015': speed_score_2015, '2016': speed_score_2016}
    return scores, (lower_bound_speed, upper_bound_speed)






def _bandpass_adaptive(speed, sampling_rate, lower_speed, upper_time, speed_bw, **kwargs):
    '''Create a filter list that allows through values based on a defined lower
    value, and an upper value determined by the highest 2cm/s bandwidth at which
    the animal spent at least X time
    
    Calculating the upper speed: 
        * Histogram the speed array with a resolution 10x higher than the 
        desired speed-bandwidth
        * Iterate over the resulting histogram to identify the highest speed range
        at which the animal spends at least upper_time
        * select the centre of this speed range as the upper threshold
    
    parameters
    ----------
    speed : np.ndarray
        1d M-length array of animal speeds at fixed sample rate, in [cm/s]
    sampling_rate : float
        Sampling rate of the tracking system, in [Hz]
    lower_speed : float
        Lower threshold of bandpass filter, in [cm/s]
    upper_time : float
        Time for calculating upper_speed in [s]. The upper_speed is calculated
        as the highest [2cm/s] speed bandwidth that the animal spends at least
        this long.
    speed_bw : float
        Range of speeds for calculating the upper_speed, in [cm/s]
        
    returns
    -------
    _filter : np.ndarray
        1d M-length array of booleans for indexing the speed array. True where
        the speed PASSES the filter
    '''
    m = 10 # multiplier on resolution - the histogram will be done with bin_widths this factor narrower than the speed_bw
    hist_resolution = speed_bw / m          # this is bin_width
    bins = np.arange(np.min(speed), np.max(speed), hist_resolution)
    hist, bin_edges = np.histogram(speed, bins=bins)
    
    required_frames = upper_time * sampling_rate
    upper_speed = None
    
    for i in np.arange(-1, -(bins.size - m), -1):
        # Iterate backwards through the histogram, i.e. from highest speeds
        total_frames = np.sum(hist[i:i+m])
        if total_frames >= required_frames:
            # go to the centre of the bandwidth. minus because of reverse direction
            upper_speed = bins[i-int(m/2)]
            break
    if kwargs.get("debug", False):
        print(f"Upper speed determined as {upper_speed} cm/s")
    if upper_speed is not None:
        _filter = _bandpass_fixed(speed, lower_speed, upper_speed, **kwargs)
    else:
        raise ValueError(f"The animal did not speed {upper_time}s within a speed"\
                         f" bandwidth of {speed_bw} cm/s. Try using a"\
                         " larger speed-bandwidth")
    return _filter, upper_speed

def _bandpass_fixed(speed, lower_speed, upper_speed, **kwargs):
    '''Create a filter list that allows through values between the defined upper
    and lower defined values
    
    parameters
    ----------
    speed : np.ndarray
        1d M-length array of animal speeds at fixed sample rate, in [cm/s]
    sampling_rate : float
        Sampling rate of the tracking system, in [Hz]
    lower_speed : float
        Lower threshold of bandpass filter, in [cm/s]
    upper_speed : float
        Upper threshold of bandpass filter, in [cm/s]
        Required upper_speed > lower_speed
    
    returns
    -------
    _filter : np.ndarray
        1d M-length array of booleans for indexing the speed array. True where
        the speed PASSES the filter
    '''
    if lower_speed >= upper_speed:
        raise ValueError(f"Your lower bound ({lower_speed}) is higher than your"\
                         f" upper bound ({upper_speed}). Check your argument order.")
    _filter = (lower_speed <= speed) & (speed <= upper_speed)
    passed = np.sum(_filter)
    if kwargs.get("debug", False):
        print(f"{passed:,} survived filter out of {speed.size:,} ({passed/speed.size:3})")
    if passed <= 5:
        raise ValueError("Your filter has excluded nearly all values, only"\
                         f" {passed} remaining. Check your filter criteria")
    return _filter

def _bandpass_none(speed, **kwargs):
    '''Create a filter list that allows all values through'''
    _filter = np.ones(speed.size, dtype=bool)
    return _filter

def _spiketimes_to_spikerate(spike_times, tracking_times, sampling_rate):
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
    frame_length = 1/sampling_rate
    bin_edges = np.append(tracking_times, tracking_times[-1]+frame_length)

    spikes_per_frame, be = np.histogram(spike_times, bins=bin_edges)
    spike_rate = spikes_per_frame *sampling_rate

    return spike_rate
