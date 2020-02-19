import numpy as np
import skimage.transform


def upsample(array, upscale, **kwargs):
    '''A MaskedArray aware upsampling routine. 
    
    `skimage` provides more reliable methods for interpolated upsampling of simple
    ndarrays. However, this is not appropriate to MaskedArrays, i.e. where
    certain locations have invalid values that should not expand.
    
    For non-masked arrays, this function can optionally rely on the `skimage`
    routine allowing fractional upscaling. For MaskedArrays, integer upscaling
    is enforced, with no interpolation
    
    Parameters
    ----------
    array: array-like
        Supports `list`, `tuple`, `np.ndarray` or `np.ma.MaskedArray`. Supports
        1d and 2d (2d for true arrays only). If the array contains any non-finite
        values, or is a MaskedArray, then integer upscaling is enforced
    upscale: int or float
        Upscaling factor. The output array shape will be this factor larger.
        Method of upscaling depends on type(upscale). 
            * `int` -> integer upscaling with no interpolation
            * `float` -> upscaling with interpolation
        Note the type distinction: `type(1) == int`; `type(1.) == float`
    debug: bool, optional
        Print out debugging information while running. Default False

    Returns
    -------
    new_array : np.ndarray or np.ma.MaskedArray
        Upsampled array
    '''
    ### General handling of the input array
    # If a tuple or list was provided, convert to an ndarray
    if isinstance(array, (tuple, list)):
        array = np.array(array)
    # Check for invalid values
    if isinstance(array, np.ndarray):
        if np.logical_not(np.isfinite(array)).any():
            # Check for any non-finite elements -> convert to masked array if so
            array = np.ma.masked_invalid(array)
        else:
            pass
    elif isinstance(array, np.ma.MaskedArray):
        pass
    else:
        raise NotImplementedError(f"Upsampling is not supported for array type {type(array)}")

    debug = kwargs.get("debug", False)
    
    if debug:
        print(f"Array type: {type(array)}")
        print(f"Upscaling: {upscale}, type {type(upscale)}")
    
    #####           Based on the inputs, decide how to proceed
    # Is the array Masked (then use integer scaling), or not masked (then allow non-integer scaling
    if isinstance(array, np.ma.MaskedArray):
        if isinstance(upscale, int):
            new_array = _integer_upsampling(array, upscale, **kwargs)
        else:
            raise NotImplementedError(f"Fractional interpolation is not supported"\
                                      f" with MaskedArrays. You requested upscaling {upscale}")
    else:
        if isinstance(upscale, int):
            new_array = _integer_upsampling(array, upscale, **kwargs)
        else:
            new_array = _fractional_upsampling(array, upscale, **kwargs)
    
    return new_array

###############################################################################
####            Actual calculation functions here

    
def _integer_upsampling(array, upscale, **kwargs):
    '''
    Perform integer upscaling on the provided array. No interpolation

    Each pixel in `array` is replaced by `upscale`x`upscale` pixels of the same
    value in `new_array`.

    Parameters
    ----------
    array: np.ndarray or np.ma.MaskedArray
        1D or 2D, the array to be scaled
    upscale: int
        upscaling factor, >= 1. 1 is an acceptable, though pointless, value -
        the original array is returned with no modifications

    Returns
    -------
    new_array : np.ndarray or np.ma.MaskedArray
        Same array type as input. 
    '''
    if upscale < 1:
        raise NotImplementedError("upscale must be 1 or higher")
    elif upscale == 1:
        new_array = array
    elif isinstance(array, np.ma.MaskedArray):
        #In the case of a MaskedArray, upscale the mask and data separately
        ndata = _integer_upsampling(array.data, upscale, **kwargs)
        nmask = _integer_upsampling(np.ma.getmaskarray(array), upscale, **kwargs)
        new_array = np.ma.masked_where(nmask, ndata)
    else:
        new_shape = np.array(array.shape) * upscale
        new_array = np.zeros(new_shape, dtype=array.dtype)

        # 1D
        if array.ndim == 1:
            for i in range(upscale):
                new_array[i:new_shape[0]+i+1-upscale:upscale] = array[:]

        # 2D
        elif array.ndim == 2:
            for i in range(upscale):
                for j in range(upscale):
                    new_array[i:new_shape[0]+i+1-upscale:upscale,
                              j:new_shape[1]+j+1-upscale:upscale] = array[:,:]
        else:
            raise NotImplementedError(f"Integer upscaling only supports 1D and 2D arrays."\
                                      f" You provided {array.ndim}D")
    return new_array


def _fractional_upsampling(array, upscale, **kwargs):
    '''Perform fractional upscaling on the provided array. Since there is no
    direct mapping from one to many locations, use interpolation to calculate
    the new values
    
    Parameters
    ----------
    array: np.ndarray
        nD array to be scaled
    upscale: float
        Scaling factor. Must be > 0
    
    Returns
    -------
    new_array : np.ndarray
        Upscaled, interpolated array. Same type and dimensionality as input array
    '''
    if upscale == 1:
        new_array = array
    elif upscale <= 0:
        raise ValueError(f"Upscaling not supported for values <= 0 ({upscale})")
    elif not isinstance(array, np.ndarray):
        raise ValueError(f"Upscaling is only supported for np.ndarrays. You provided {type(array)}")
    else:
        new_array = skimage.transform.rescale(array, upscale)
    return new_array