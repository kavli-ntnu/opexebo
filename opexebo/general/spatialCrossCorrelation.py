import numpy as np

def spatial_cross_correlation(arr_0, arr_1, **kwargs):
    '''
    Calculate the Pearson cross correlation between two arrays. 
    
    Arrays may be either 1d or 2d. If MaskedArrays are supplied, all locations
    that are masked are excluded from the correlation. Invalid numbers (NaN) 
    are also excluded from the correlation.

    If 2d arrays are provided, then calculates a 1d array of row(or column)-wise
    correlations, controlled by the `row_major` keyword.

    Parameters
    ----------
    arr_0: np.ndarray
        1d or 2d array. Must either contain no NaNs, or be an np.ma.MaskedArray
        with NaN values masked. 
    
    arr_1: np.ndarray
        2nd array to correlate with. Must have the same dimensions as `arr_0`
    
    Other Parameters
    ----------------
    row_major: bool, optional
        If the arrays are 2D, process first by row, or first by column.
        Default `True`

    Returns
    -------
    output_single: float
        Pearson correlation coefficient between entire arrays. 2d input arrays
        are flattened to 1d to calculate
    output_array: np.ndarray
        1D array of pearson correlation coefficients, where the i'th value is
        the coefficient between the i'th rows (if row_major), or the i'th columns
        of the two arrays
        Return value is np.nan if 1d arrays are supplied

    Notes
    --------
    BNT.+analyses.spatialCrossCorrelation()

    Copyright (C) 2019 by Simon Ball
    '''
    debug = kwargs.get("debug", False)
    arr_0 = _check_single_array(arr_0, 0, **kwargs)
    arr_1 = _check_single_array(arr_1, 1, **kwargs)        
    
    if arr_0.shape != arr_1.shape:
        raise ValueError(f"Your arrays are different shapes: {arr_0.shape} and"\
                         " {arr_0.shape}. You must provide the same shape arrays")

    if arr_0.ndim not in (1, 2):
        raise ValueError(f"Your arrays have the wrong number of dimensions. Only"\
                         "1d and 2d arrays are supported. You provided {arr_0.ndim}"\
                         "dimensions")

    is_2d = arr_0.ndim==2

    joint_mask = np.logical_or(arr_0.mask, arr_1.mask)
    good = np.logical_not(joint_mask)
    if debug:
        print(f"is_2d: {is_2d}")
        print(f"{np.sum(joint_mask)} values excluded")

    output_single = np.corrcoef(arr_0.data[good].flatten(), arr_1.data[good].flatten())[0,1]
    output_array = np.nan

    if is_2d:
        row_major = kwargs.get("row_major", True)
        y_bins, x_bins = arr_0.shape
        if row_major:
            iterator = y_bins
        else:
            iterator = x_bins
        output_array = np.zeros(iterator)

        for i in range(iterator):
            if row_major:
                left = arr_0[i, :][good[i, :]]
                right = arr_1[i, :][good[i, :]]
            else:
                left = arr_0[:, i][good[:, i]]
                right = arr_1[:, i][good[:, i]]
            output_array[i] = np.corrcoef(left, right)[0,1]

    return output_single, output_array


def _check_single_array(arr, i, **kwargs):
    '''input checking on a single array'''
    if isinstance(arr, np.ndarray):
        arr = np.ma.masked_invalid(arr)
    elif not isinstance(arr, np.ma.MaskedArray):
        raise ValueError(f"arr_{i} is type {type(arr)}. You must provide a Numpy"\
                         "array or MaskedArray")
    return arr