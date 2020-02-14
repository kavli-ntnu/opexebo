"""
Calculate 2D spatial autocorrelation

Calculates 2D autocorrelation (autocorrelogram) of a firing map.

"""

import numpy as np
import opexebo


def autocorrelation(firing_map):
    """Calculate 2D spatial autocorrelation of a firing map.

    Parameters
    ----------
    firing_map: np.ndarray
        NxM matrix, smoothed firing map. map is not necessary a numpy array. 
        May contain NaNs.

    Returns
    -------
    acorr: np.ndarray
        Resulting correlation matrix, which is a 2D numpy array.

    See Also
    --------
    opexebo.general.normxcorr2_general
        
    Notes
    -----
    BNT.+analyses.autocorrelation

    Copyright (C) 2018 by Vadim Frolov
    """

    # overlap_amount is a parameter that is intentionally not exposed to
    # the outside world. This is because too many users depend on it and we
    # do not what everyone to use their own overlap value.
    # Should be a value in range [0, 1]
    overlap_amount = 0.8
    slices = []

    if type(firing_map) != np.ndarray:
        firing_map = np.array(firing_map)

    if firing_map.size == 0:
        return firing_map

    # make sure there are no NaNs in the firing_map
    firing_map = np.nan_to_num(firing_map)

    # get full autocorrelgramn
    aCorr = opexebo.general.normxcorr2_general(firing_map)

    # we are only interested in a portion of the autocorrelogram. Since the values
    # on edges are too noise (due to the fact that very small amount of elements
    # are correlated).
    for i in range(firing_map.ndim):
        new_size = np.round(firing_map.shape[i] + firing_map.shape[i] * overlap_amount)
        if new_size % 2 == 0:
            new_size = new_size - 1
        offset = aCorr.shape[i] - new_size
        offset = np.round(offset/2 + 1)
        d0 = int(offset-1)
        d1 = int(aCorr.shape[i] - offset + 1)
        slices.append(slice(d0, d1))

    return aCorr[tuple(slices)]
