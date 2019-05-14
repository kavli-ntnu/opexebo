"""Provide function for shuffling"""

import numpy as np

def shuffle(times, t_min, t_max, iterations):
    '''
    Increment the provided time series by a random period of time in order to 
    destroy the correlation between spike times and animal behaviour.
    
    The increment behaves circularly: 
        * initially, we have two time series, Tracking and SpikeTimes, both with 
        values in the range [t0, t1]. 
        * after incrementing by T, we have Tracking in [t0, t1] and SpikeTimes
        in [t0+T, T1+T]
        * Take all spike times in the range [t1, t1+T] and map back to [t0, t0+T]
    
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
    
    Returns
    -------
    output : np.ndarray
        N x iterations array of times. To access a single iteration, output[:, i]
    
    See also
    --------
    BNT.+scripts.shuffling
    '''
    
    # Check values
    if type(times) != np.ndarray:
        times = np.array(times)
    if times.ndim != 1:
        raise ValueError("You must provide a 1D array of times. You provided a %d-dimensional array" % times.ndim)
    
    # Initialise Numpy's random number generator with a new seed based on the 
    # user's local computer OS methods
    
    
    increments = np.random.RandomState().rand(iterations) # Uniformmly distrbuted in [0, 1]
    increments = t_min + (increments * (t_max-t_min))
    
    t0 = np.min(times)
    t1 = np.max(times)
    T = np.max(increments)
    num_spikes = np.size(times)
    
    
    # Generate 2D arrays by repeating over the correct axis
    # Want to duplicate the whole list of times for each repeat
    # Want to duplicate the *same* increment for each repeat. Hence the transpose. 
    all_times = np.repeat(times[:,np.newaxis], iterations, axis=1)              
    all_inc = np.repeat(increments[:, np.newaxis], times.size, axis=1).transpose()
    new_times = all_times + all_inc
    
    to_shift = (new_times > (t1-increments)) # Binary array of the elements that need to be moved to the beginning. Each column is different
    time_idx = num_spikes - np.count_nonzero(to_shift, axis=0) # indicies of the first time in each iteration which must be moved
    iter_idx = np.arange(iterations)
    output = np.zeros(new_times.shape)

    for i in iter_idx:
        # Circular buffer. We have alread identified the index at which elements 
        # need to be moved (time_idx[i]). So take those end elements, and move 
        # them to the start
        # Turns out, this is actually faster than fancy array indexing with np.ix_(time_idx, iter_idx), by nearly 100x (!) in testing
        output[0:num_spikes-time_idx[i], i] = new_times[time_idx[i]:, i] - increments[i]
        output[time_idx[i]:, i] = new_times[:num_spikes-time_idx[i], i]

    
    return output
    
if __name__ == '__main__':
    times = np.arange(5000)
    t_min = 17
    t_max = 42
    iterations = 200
    s = shuffle(times, t_min, t_max, iterations)

    