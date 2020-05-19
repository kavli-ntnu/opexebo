"""Interpret the 'arena_size' keyword"""

import numpy as np

import opexebo.defaults as default
from opexebo.errors import ArgumentError

def validatekeyword__arena_size(kwv, provided_dimensions):
    '''
    Decipher the possible meanings of the keyword "arena_size".
    
    "arena_size" is given to describe the arena in which the animal is moving
    It should be either a float, or an array-like of 2 floats (x, y)
    
    Parameters:
    ----------
    kw: float or array-like of floats
        The value given for the keyword `arena_size`
    provided_dimensions : int
        the number of spatial dimensions provided to the original function.
        Acceptable values are 1 or 2
        E.g. if the original function was provided with positions = [t, x, y], then
        provided_dimensions=2 (x and y)

    Returns
    -------
    arena_size : float or np.ndarray of floats
    
    Raises
    ------
    ValueError
    IndexError
    '''
    if provided_dimensions == 1:
        is_2d = False
    elif provided_dimensions == 2:
        is_2d = True
    else:
        raise NotImplementedError("Only 1d and 2d arenas are supported. You"\
                                  " provided %dd" % provided_dimensions)
    if type(kwv) in (float, int, str):
        kwv = float(kwv)
        if kwv <= 0: 
            raise ValueError("Keyword 'arena_size' value must be greater than"\
                             " zero (value given %f)" % kwv)
        if is_2d:
            arena_size = np.array((kwv, kwv))
        else:
            arena_size = kwv
    elif type(kwv) in (list, tuple, np.ndarray):
        if len(kwv) == 1:
            if is_2d:
                
                arena_size = np.array(kwv[0], kwv[0])
            else:
                arena_size = kwv[0]
        elif len(kwv) == 2 and not is_2d:
            raise IndexError("Mismatch in dimensions: 1d position data but 2d"\
                             " arena specified")
        elif len(kwv) not in [1, 2]:
            raise IndexError("Keyword 'arena_size' value is invalid. Provide"\
                             " either a float or a 2-element tuple")
        else:
            arena_size = np.array(kwv)
    else:
        raise ValueError("Keyword 'arena_size' value not understood. Please"\
                         " provide either a float or a tuple of 2 floats. Value"\
                         " provided: '%s'" % str(kwv))
    return arena_size, is_2d


def validate_keyword_arena_shape(arena_shape):
    '''
    Ensure that the arena_shape is a meaningful value
    
    Parameters
    ----------
    arena_shape : str
        the value given for the keyword `arena_shape`
    
    Returns
    -------
    arena_shape : str
        A value that is guaranteed to be an acceptable member of one of the
        recognised groups of arena_shapes
    '''
    if not isinstance(arena_shape, str):
        raise ArgumentError("Keyword `arena_shape` must be a string, not type `{type(arena_shape)}`")
    else:
        arena_shape = arena_shape.lower()
    
    if arena_shape in default.shapes_square:
        # this is ok
        pass
    elif arena_shape in default.shapes_circle:
        # this is ok
        pass
    elif arena_shape in default.shapes_linear:
        # this is ok
        pass
    else:
        raise NotImplementedError(f"Arena shape '{arena_shape}' not implemented")
    
    return arena_shape
    
    