import numpy as np

from opexebo import errors

def walk_filter(speed:np.ndarray, speed_cutoff:float, *args, fmt="remove"):
    """
    It is common practice when studying a freely moving subject to exclude data
    from periods when the subject was stationary, or nearly stationary. This
    method is described as a "walk-filter" - a high-pass filter on subject speed.
    
    This function allows an arbitrary number of arrays to be filtered in parallel
    to the speed (or whatever other filtering criteria are used). Filters can be
    performed either by removing the unwanted elements, or by masking them and
    retunring a MaskedArray.
    
    Example
    -------
    Filter speed only
    >>> speed = np.arange(500)
    >>> cutoff = 200
    >>> speed = walk_filter(speed, cutoff, fmt="remove")
    >>> speed.size
    300
    
    Filter other arrays as well
    >>> speed = np.arange(500)
    >>> cutoff = 200
    >>> pos_x = np.linspace(-25, 73, speed.size)
    >>> pos_y = np.linspace(0, 98, speed.size)
    >>> speed, pos_x, pos_y = walk_filter(speed, cutoff, pos_x, pos_y, fmt="remove")
    >>> speed.size
    300
    >>> pos_x.size
    300
    

    Parameters
    ----------
    speed : np.ndarray
        Array of speeds for other data points
    speed_cutoff : float
        The cutoff, below which values in ``speed`` will be excluded.
    *args : np.ndarray, optional
        Any other arrays that should be filtered in parallel with speed
        Optional arguments here _must_ be np.ndarrays with size equal to that of
        ``speed``
    fmt : str, optional
        Either "remove" or "mask". Determines how the values are returned
        "remove" (default) - the invalid valaues are removed from the array
        "mask" - the original array is returned as a MaskedArray, with the invalid
        values masked out.
    

    Returns
    -------
    np.ndarray
        Filtered copy of ``speed``
    [np.ndarray]
        Arbitrary other filtered arrays, if any other arrays were provided as *args
    """
    
    if not isinstance(speed, np.ndarray):
        raise errors.ArgumentError("`speed` should be an ndarray, not ({})".format(type(speed)))
    
    if not isinstance(speed_cutoff, (float, int)):
        raise errors.ArgumentError("`speed_cutoff` should be a numeric value ({})".format(type(speed_cutoff)))
    if speed_cutoff <= 0 or not np.isfinite(speed_cutoff):
        raise errors.ArgumentError("\speed_cutoff` should be a finite positive value")
    
    if fmt.lower() not in ("remove", "mask"):
        raise errors.ArgumentError("`fmt` should be either 'remove' or 'mask'")
    
    if len(args):
        for i, arg in enumerate(args):
            if not isinstance(arg, np.ndarray):
                raise errors.ArgumentError(f"`arg {i} is not a Numpy array ({arg})")
            if not arg.shape == speed.shape:
                raise errors.ArgumentError(f"`arg {i} is a different size to `speed`")
            
    good = speed >= speed_cutoff
    
    if fmt.lower() == "mask":
        bad = np.logical_not(good)
        speed = np.ma.masked_where(bad, speed)
        out_args = [np.ma.masked_where(bad, arg) for arg in args]
    elif fmt.lower() == "remove":
        speed = speed[good]
        out_args = [arg[good] for arg in args]
        
    if out_args:
        out_args.insert(0, speed)
        return out_args
    else:
        return speed
