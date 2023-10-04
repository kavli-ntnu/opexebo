""" Provides a function for Gaussian smoothing """
# Hack to speed up the borked Astropy configuration search
import os
import pathlib
os.environ["HOMESHARE"] = str(pathlib.Path.home())

import numpy as np
import warnings
with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    # Suppress the ConfigurationMissingWarning that Astropy triggers
    from astropy.convolution import convolve, Gaussian2DKernel, Gaussian1DKernel

import opexebo.defaults as default
#http://docs.astropy.org/en/stable/convolution/index.html
#
#Astropy appears to have an import problem - it routinely takes about 3 minutes
#to import, apparently because it is looking for a file with a stupidly long
#timeout. It typically results in a warning:
#        ConfigurationMissingWarning : Configuration defaults will be used due
#           to FileNotFoundError:2 on None
#
#The root cause seems to be that Astropy expects to find a configuration file
#in a very specific location - in thise case, a network location
#       (\\home.ansatt.ntnu.no\.astropy)
#Which doesn't exist. But the timeout for checking the network is very long
#
# A tempoary workaround is to create the environment variables
#    XDG_CACHE_HOME
#    XDG_CONFIG_HOME
# pointing to local locations that do exist. Since we only use a very minor
# part of Astropy here, the fact that the config file may be ephemeral is not a
# problem. See here for discussion of these variables
# https://github.com/astropy/astropy/issues/6511


def smooth(data, sigma, **kwargs):
    '''Smooth provided data with a Gaussian kernel

    The smoothing is done with a routine from the astronomical package astropy
    Like scipy.ndimage.gaussian_filter, this does not handle MaskedArrays - but
    it handles NaNs much better. Specifically, astropy.convolution.convolve
    replaces NaN values with an interpolation across the void region.

    Therefore, to handle masked arrays, the data at masked positions *is
    replaced by np.nan* prior to smoothing, and thus avoids influencing nearby,
    unmasked cells. The masked cells are then returned to their original values
    prior to return.

    The package is discussed at
    http://docs.astropy.org/en/stable/convolution/index.html

    Parameters
    ----------
    data: np.ndarray or np.ma.MaskedArray
        Data that will be smoothed
    sigma: float
        Standard deviations for Gaussian kernel in units of pixels/bins
    mask_fill: float, optional
        The value that masked locations should be treated as
        This can either be provided as an absolute number (e.g. 0), or nan
        If nan, then each masked location will get a value by interpolating
        from nearby cells. This will only apply if the input is a
        MaskedArray to start with. If the input is a standard np.ndarray,
        then no values will be substituted, even if there are nans present.
    circular: bool, optional
        If True, then smoothing at the edge of the array will be handled in
        a circular manner, i.e. the value to the left of data[0] will be
        data[-1]. If False, the edge will be handled by padding with values
        equal to the boundary value. Default False

    Returns
    -------
    smoothed_data: np.ndarray or np.ma.MaskedArray
        Smoothed data

    Notes
    --------
    BNT.+general.smooth
    http://docs.astropy.org/en/stable/convolution/index.html
    https://github.com/astropy/astropy/issues/6511

    Copyright (C) 2019 by Simon Ball
    '''
    d = data.ndim
    if  d == 2:
        kernel = Gaussian2DKernel(x_stddev=sigma)
    elif d == 1:
        kernel = Gaussian1DKernel(stddev=sigma)
    else:
        raise NotImplementedError("This function currently supports smoothing"\
                f" 1D, 2D data. You have provided {d} dimensional data")

    mask_fill = kwargs.get('mask_fill', default.mask_fill)
    circular = kwargs.get("circular", False)
    if not isinstance(circular, bool):
        raise ValueError("You must provide a boolean (True/False) value for"\
                         f" keyword 'circular'. You provided {circular}, which"\
                         f" is type {type(circular)}")

    working_data = data.copy()
    if type(data) == np.ma.MaskedArray:
        working_data[data.mask] = mask_fill

    width = int(4*sigma)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        if circular:
            smoothed_data = convolve(working_data, kernel, boundary="wrap")
            # Don't bother with padding. Use the values from the other end of the
            # array, i.e. imagine the array wrapped around a cylinder
        elif not circular:
            working_data = np.pad(working_data, pad_width=width, mode='symmetric')
            # pad the outer boundary to depth "width
            # The padding values are based on reflecting at the border
            # mode='symmetrical' results in
            # [0, 1, 2, 3, 4] -> [1,0  ,0,1,2,3,4,  4,3]
            # mode='reflect' results in
            # [0, 1, 2, 3, 4] -> [2,1  ,0,1,2,3,4,  3,2]
            # i.e. changing whether the reflection axis is outside the original data
            # or overlaid on the outermost row

            smoothed_data = convolve(working_data, kernel, boundary='extend')
            # Because of the padding, the boundary mode isn't really relevant
            # By choosing a large width, the edge effects arising from this additional
            # padding (boundary='extend') is minimised

            if d == 2:
                smoothed_data = smoothed_data[width:-width, width:-width]
            elif d == 1:
                smoothed_data = smoothed_data[width:-width]
            else: # This condition should never happen, due to checking above
                raise NotImplementedError("This function currently supports smoothing"\
                        f" 1D, 2D data. You have provided {d} dimensional data")
            # We have to get rid of the padding that we previously added, and the only
            # way to do that is slicing, which is NOT dimensional-agnostic
            # There may be a more elegant solution than if/else, but this will do now

    if type(data) == np.ma.MaskedArray:
        smoothed_data = np.ma.masked_where(data.mask, smoothed_data)
        smoothed_data.data[data.mask] = data.data[data.mask]
    
    assert smoothed_data.shape == data.shape, "Output array is a different shape to input array"

    return smoothed_data
