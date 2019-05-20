""" Provides functions for calculating Speed Score """


import numpy as np
import opexebo.defaults as default
import opexebo

def speedscore(spike_times, tracking_speeds, **kwargs):
    '''
    Calculate Speed score.
    
    Speed score is a correlation between cell firing *rate* and animal speed. 
    The Python version is based on "score16" from BNT.+scripts.speedScore
    
    
    General structure:
        Speed limits:
            Original code (Emiliano) uses a low speed filter
            Score16 introduces an upper speed filter, defined as maximum speed 
            at which the animal spent no more than <period_of_time=10s>
        Kalman: (155-193)
            Optionally, smooth the binned speedtuning curve with a Kalman filter
            uses BNT.externals.kalman.doKalmanProcessing (I think)
            Emiliano's Kalman filtering used a filter based on 
            http://www.cs.ubc.ca/~murphyk/Software/Kalman/kalman.html (dated 2004)
        Calculation: (243 - )
            scores :
                2015: (278-285)
                2016: (285-300)
                    Both operate on the table firingRateSmoothed
                
                
    A whole bunch of the script is not very relevant - because it's a script 
    rather than ana analysis it does a bunch of other stuff, like iterating over 
    many cells / sessions, plotting data, etc, which this file doesn't need to 
    handle
                    
        
    
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
    
    See also
    --------
    BNT.+scripts.speedScore
    '''
    # Check that the provided functions have correct dimensions
    stn = spike_times.ndim
    if stn != 1:
        raise ValueError("Spike Times must be an Nx1 array. You have provided %d dimensions" % stn)
        
    # Get kwargs values
    bin_width = kwargs.get('bin_width', default.bin_speed)
    sigma = kwargs.get('sigma', default.sigma_speed)
    upper_bound_time = kwargs.get('upper_bound_time', default.upper_bound_time)
    lower_bound_speed = kwargs.get('lower_bound_speed', default.lower_bound_speed)
    debug = kwargs.get('debug', False)
    
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
    num_bins = int(( np.max(t_speeds) - np.min(t_speeds)) / bin_width) + 1
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
    if debug:
        print(filtered_speeds.shape)
        print(filtered_rate.shape)
    speed_score_2016 = np.corrcoef(filtered_speeds, filtered_rate.T, rowvar=0)[0,1]
    if debug:
        print(speed_score_2016)
    
    
    # Score 2015: Filter rates first (by setting to NaN), and then smooth and correlate
    # Reuse the same filtered_speeds as for 2016, but redefine filtered_rate
    aux_rate = firing_rate[:]
    aux_rate[bad_speeds] = np.nan
    filtered_rate = opexebo.general.smooth(aux_rate, sigma)
    filtered_rate = filtered_rate[good_speeds]
    speed_score_2015 = np.corrcoef(filtered_speeds, filtered_rate.T, rowvar=0)[0,1]
    
    return (speed_score_2015, speed_score_2016)
    
    # (177) posForSpeed - get te list of positions from tracking 
    # Calculate speed
    # Copy into the HalmanResults structure so that the same code can run even though the Kalman filter wasn't applied
    # (227) Calculate speed distribution, what I had been calling SpeedTuning
    # Get the upper speed threshold from this distribution
    # Bandpass filter speeds
    # For each cell
        # (270) Correlate spike times to speeds
        # (273) Calculate and smooth firing rate
        # (282) Calculate score 2015 : matlab corr(speed, smooth(firing_rate))
            # firing rate -> aux. aux[badspeed] = nan. aux_s = smooth(aux). Use aux_s
        # Also calculate shuffle
        # (286) Calculate score 2016
            # use firing_rate-smoothed[goodSpeed] - i.e. the bad speeds will nevertheless have contributed because they were used as part of the smoothing
        
    
  
    
def _spiketimes_to_spikerate(spike_times, tracking_times):
    '''Convert a list of spike times to a list of spike rates
    parameters
    ----------
    spike_times : np.ndarray
        Nx1 array of times at which spikes occur in [s]
    tracking times : np.ndarray
        Nx1 array of times at which tracking information is known - e.g. time stamp of camera frames
        Also in [s]
    
    returns
    -------
    spike_rate : np.adarray
        Nx1 array of spike rate [Hz], with the i'th value being the spike rate during the i'th tracking frame.
        
    
    '''
    frame_length = np.min(np.diff(tracking_times))
    bin_edges = np.append(tracking_times, tracking_times[-1]+frame_length)
    
    spikes_per_frame, be = np.histogram(spike_times, bin_edges)
    spike_rate = spikes_per_frame / np.append(np.diff(tracking_times), frame_length)
    
    return spike_rate
    