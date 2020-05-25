import numpy as np

def bin_width_to_bin_number(length, bin_width):
    '''This conversion is done in several separate places, and must be done the
    same in every case. Therefore, refactor into a quick helper function.
    
    In cases where the bin_width is not a perfect divisor of length, the actual
    bins will be slightly smaller
    
    Parameters
    ----------
    length: float or np.ndarray
        Length of a side to be divided into equally spaced bins
    bin_width: float
        Dimension of a square bin or pixel
    
    Returns
    -------
    num_bins: int or np.ndarray
        Same type as `length`. Integer number of bins.
    '''
    if type(length) in (list, tuple):
        length = np.array(length)
    num_bins = np.ceil(length / bin_width).astype(int)
    return num_bins