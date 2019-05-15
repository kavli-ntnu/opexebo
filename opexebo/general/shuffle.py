"""Provide function for shuffling"""

import numpy as np
import time

def shuffle(times, t_min, t_max, iterations, **kwargs):
    '''
    Increment the provided time series by a random period of time in order to 
    destroy the correlation between spike times and animal behaviour.
    
    The increment behaves circularly: 
        * initially, we have two time series, Tracking and SpikeTimes, both with 
        values in the range [t0, t1]. 
        * after incrementing by T, we have Tracking in [t0, t1] and SpikeTimes
        in [t0+T, t1+T]
        * Take all spike times in the range [t1, t1+T] and map back to [t0, t0+T]
        by subtracting (t1-t0)
    
    The end result should be a series of times also in the range [t0, t1], 
    with the same intervals between times**, but the exact value of those times
    has changed. 
    
    ** with 1 exception, at the gap between where the oldest times end and
    the first times begin
    
    The random numbers are drawn from a pseudorandom, uniform, distribution in 
    the range [t_min, t_max], seeded 
        
    
    Parameters
    ----------
    times : np.ndarray
        Nx1 array of times
    t_min : float
        The minimum value that the time series can be incremented by
    t_max : float
        the maximum value that the time series can be incremented by
    iterations : int
        How many copies of times should be returned (each copy incremented by a random offset)
    kwargs
        'debug' : bool
            Enable additional debugging output
        'tracking_range' : array_like
            The time range over which tracking behaviour exists, defining the
            times at which spike indexes are looped back on themselves
            This can be provided either as a 2-element tuple (t0, t1), or as 
            the entire list of tracking timestamps. In each case, the min and 
            max values are used as t0, t1
            If no values are provided, then the first and last spike times
            are used. This is not desirable behaviour, since it will then guarantee 
            that in the shuffled output, two spikes will occur simultaneously
    
    Returns
    -------
    output : np.ndarray
        N x iterations array of times. To access a single iteration, output[:, i]
    
    See also
    --------
    BNT.+scripts.shuffling (around line 350)
    '''
    
    # Check values
    if type(times) != np.ndarray:
        times = np.array(times)
    if times.ndim != 1:
        raise ValueError("You must provide a 1D array of times. You provided a %d-dimensional array" % times.ndim)
    
    debug = kwargs.get('debug', False)
    tr = kwargs.get('tracking_range', None)
    if tr:
        t0 = np.min(tr)
        t1 = np.max(tr)
    else:
        t0 = np.min(times)
        t1 = np.max(times)

    
    # Initialise Numpy's random number generator with a new seed based on the 
    # user's local computer OS methods
    
    
    increments = np.random.RandomState().rand(iterations) # Uniformmly distrbuted in [0, 1]
    increments = t_min + (increments * (t_max-t_min))
    
    num_spikes = np.size(times)
    
    if debug:
        print("Number of spikes provided: %d" % num_spikes)
        print("Spikes in time range [%d, %d]" % (t0, t1))
        print("Iterations requested: %d" % iterations)
        print("Increments in range [%.2f, %.2f]" % (np.min(increments), np.max(increments)))
    
    
    # Generate 2D arrays by repeating over the correct axis
    # Want to duplicate the whole list of times for each repeat
    # Want to duplicate the *same* increment for each repeat. Hence the transpose. 
    all_times = np.repeat(times[:,np.newaxis], iterations, axis=1)              
    all_inc = np.repeat(increments[:, np.newaxis], times.size, axis=1).transpose()
    new_times = all_times + all_inc
    
    
    
    to_shift = (new_times > t1) # Binary array of the elements that need to be moved to the beginning. Each column is different
    time_idx = num_spikes - np.count_nonzero(to_shift, axis=0) # indicies of the first time in each iteration which must be moved
    if debug:
        print("New time range: [%d, %d]" % (np.min(new_times), np.max(new_times)))
        print("Indexes exceeding value t1: %s" % to_shift)
        
    output = np.zeros(new_times.shape)

    a0 = time.time()
    for i in range(iterations):
        # Circular buffer. We have alread identified the index at which elements 
        # need to be moved (time_idx[i]). So take those end elements, and move 
        # them to the start
        # Turns out, this is actually faster than fancy array indexing with np.ix_(time_idx, iter_idx), by nearly 100x (!) in testing
        
        # Elements that are still within the time range[t0+T, t1]
        a = new_times[0:time_idx[i], i]
        
        # Elements that are in the time range [t1, t1+T]
        # len(b) == num_spikes-time_idx[i]
        b = new_times[time_idx[i]:, i]
        
        # Elements in b should have their values reduced to fit in the range [t0, t0+T]        
        b = b - t1 + t0
        
        # Insert elements into output in the correct order: b first, then a filling in the blanks after
        output[0:num_spikes-time_idx[i], i] = b
        output[num_spikes-time_idx[i]:, i] = a

    a1 = time.time()
    if debug:
        print("Final time range: [%d, %d]" % (np.min(output), np.max(output)))
        print("method 1 took %dms" % ((a1-a0)*1000))
        print(np.diff(times))
        print(np.diff(output[:,0]))
    
    return output
    
if __name__ == '__main__':
    times = [2, 12, 27, 54, 82, 113, 115, 207, 300]
    t_min = 20
    t_max = max(times)-t_min
    iterations = 1
    s = shuffle(times, t_min, t_max, iterations, debug=1, tracking_range=(0,349))
    #print(s)

    