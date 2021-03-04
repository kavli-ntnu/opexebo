import numpy as np
import opexebo
import opexebo.defaults as default


array_like = (np.ndarray, np.ma.MaskedArray, list, tuple)

def speed_occupancy(time, speed, **kwargs):
    '''
    Calculate speed occupancy from speed
    
    Parameters
    ----------
    time : numpy.ndarray
        time stamps of angles in seconds
    speed : numpy.ndarray
        animal speed
    bin_width : float, optional
        Width of speed histogram bins
    
    Returns
    -------
    speed_histogram : numpy.ma.MaskedArray
        Speed histogram: masked array of time spent (in seconds) in specific
        speed ranges
    coverage: float
        Fraction of the speed ranges that the animal occupied. Range [0, 1]
    bin_edges: numpy.ndarray
        Bin edges of the speed histogram. len(bin_edges) == len(speed_histogram)+1
    '''
    for arg in (time, speed):
        if not isinstance(arg, array_like):
            raise ValueError("Time and Speed must be array-like")
        if not arg.ndim == 1:
            raise ValueError("Time and Speed must be 1d")
    if not time.size == speed.size:
        raise ValueError("Time and Speed must be the same size")
    
    bin_width = kwargs.get("bin_width", default.bin_speed)
    arena_size = max(speed)
    limits = (0, arena_size)
    
    speed_histogram, bin_edges = opexebo.general.accumulate_spatial(speed,
                                                                    bin_width=bin_width,
                                                                    arena_size=arena_size,
                                                                    limits=limits)
    masked_speed_histogram = np.ma.masked_where(speed_histogram==0, speed_histogram)
    
    frame_duration = np.mean(np.diff(time))
    masked_speed_histogram_seconds = masked_speed_histogram * frame_duration
    
    coverage = np.count_nonzero(speed_histogram) / masked_speed_histogram_seconds.size
    
    return masked_speed_histogram_seconds, coverage, bin_edges


def speed_map(time, speed, **kwargs):
    '''Calculate the firing rate of the animal as a function of its speed
    '''
    raise NotImplementedError
    
    
    