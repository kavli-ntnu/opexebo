import numpy as np
from astropy.convolution import convolve


def calc_speed(t, x, y=None, moving_average=None):
    """
    Calculate the speed of an animal given a list of times and positions

    Speed is calculated element-wise, with an optional moving-average filter.
    Speed is not checked for non

    Parameters
    ----------
    t : np.ndarray
        1d array containing time stamps
    x, y : np.ndarray
        1d arrays containing positions. `y` is optional and may be excluded for
        1d datasets
    moving_average: int, optional
        Apply a moving-average filter of this size to the speed data
        To exclude this filter, leave blank or set [None] (default)
    
    Returns
    -------
    np.ndarray
        1d array containing speed in units [position_unit / time_unit]
    """
    if y is None:
        y = np.zeros(t.size)
    arg_names = ["t", "x", "y"]
    for i, arg in enumerate((t, x, y)):
        if not isinstance(arg, (list, tuple, np.ndarray, np.ma.MaskedArray)):
            raise ValueError(
                "{} must be an iterable, not  {}".format(arg_names[i], type(arg))
            )
        if not isinstance(arg, (np.ndarray, np.ma.MaskedArray)):
            arg = np.array(arg)
        if not arg.ndim == 1:
            raise ValueError(
                "{} must be 1-dimensional, not {}-dimensional".format(
                    arg_names[i], arg.ndim
                )
            )
        if not arg.size == t.size:
            raise ValueError(
                "{} must be the same length as `t` ({}, {})".format(
                    arg_names[i], arg.size, t.size
                )
            )

    if moving_average is not None:
        if not isinstance(moving_average, int):
            raise ValueError(
                "`moving_average` must be None or type `int`, not {}".format(
                    moving_average
                )
            )
        if moving_average < 3 or moving_average % 2 != 1:
            raise ValueError(
                "`moving_average` must be a positive odd integer greater than 1"
            )

    d_x = np.diff(x)
    d_y = np.diff(y)
    d_t = np.diff(t)
    speed = abs(np.sqrt(np.square(d_x) + np.square(d_y)) / d_t)
    new_speed = np.zeros(speed.size + 1)
    new_speed[1:] = speed
    new_speed[0] = speed[0]

    if moving_average is not None:
        kernel = np.ones(moving_average) / moving_average
        new_speed = convolve(new_speed, kernel, boundary="extend")
    return new_speed
