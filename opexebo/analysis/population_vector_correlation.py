"""
Provide function for population vector correlation calculation
"""
import numpy as np

from .. import errors as err


def population_vector_correlation(stack_0, stack_1, **kwargs):
    """Calculates the bin-wise correlation between two stacks of rate maps
    
    Each stack corresponds to a separate Task, or trial. Each layer is the 
    ratemap for a single cell from that Task. The same units should be given in
    the same order in each stack.
    
    Take a single column through the stack (i.e. 1 single bin/location in
    arena, with a firing rate for each cell), from each stack
    
    In the original MatLab implementation, three output modes were supported
        * 1D: (`numYbins`) - iterate over `i`
            1) Take a 2D slice from each stack - all cells at all `X` positions at a
              single `Y` position `i`
            2) Reshape from 2D to 1D 
            3) Calculate the Pearson correlation coefficient between the two 1D
              arrays
            4) The value of `pv_corr_1d[i]` is the Pearson correlation coefficient
              arising from `Y` position `i`
        * 2D (`numXbins` x `numYbins`) - iterate over `i`
            1) Take a 2D slice from each stack - all cells at all `X` positions at a
              single `Y` position `i`
            2) Calculate the 2D array (`numXbins` x `numYbins`) where the `[j,k]`th
              value is the Pearson correlation coefficient between all
              observations at the `j`'th `X` location in `stack_left` and the `k`'th
              location in `stack_right`
            3) The `i`'th row of `pv_corr_2d` is the DIAGONAL of the correlation matrix
              i.e. where `j==k` i.e. the correlation of the the SAME location in
              each stack for all observations (`numCells`)
        * 3D (`numXbins` x `numYbins` x iteration(=`numYbins`))
            Same as 2D BUT take the whole correlation matrix, not the diagonal
            i.e. the full [j,k] correlatio between all X locations
    
    A note on correlation in Numpy vs Matlab
    
    Matlab's `corr(a, b)` function returns the correlation of ab
    Numpy's `corrcoef` function returns the normalised covariance matrix,
    which is:
            aa  ab
            ba  aa
    The normalised covariance matrix *should* be hermitian, but due to
    floating point accuracy, this is not actually guaranteed
    the MatLab function can be reproduced by taking either [0, 1] or [1,0]
    of the normalised covariance matrix. 

    If `a`, `b` are 2D matricies, then they should have shape `(num_variables, num_observations)`
    In the case of this function, where the iterator is over the `Y` values
    of the rate map, that means: `(x_bins, num_cells)`

    Parameters
    ----------
    stack_0: 3D array -or- list of 2D arrays
    stack_1: 3D array -or- list of 2D arrays
        `stack_x[i]` should return the `i`'th ratemap. This corresponds to a 
        constructor like:
            `np.zeros(num_layers, y_bins, x_bins)`
            
        Alternatively, a list or tuple of 2D arrays may be supplied:
            `stack_x` = (`ratemap_0`, `ratemap_1`, `ratemap_2`, ...)
    row_major: bool
        Direction of iteration. If `True`, then each row is iterated over in turn
        and correlation is calculated per row. 
        If `False`, then each column is iterated over in turn, and correlation is 
        calculated per column. 
        Default True (same behavior as in BNT)

    Returns
    -------
    (p1, p2, p3)
    p1: np.ndarray (1D, iterator x 1)
        Array of Pearson correlation coefficients. i'th value is given by the 
        correlation of the i'th flattened slice of stack_0 to the i'th
        flattened slice  of stack_1
    p2: np.ndarray (2D, iterator x non-iterator)
        i'th row is the diagonal of the correlation matrix, i.e. the correlation
        of the same location (location i) in each stack, i.e. where j==k
    p3: np.ndarray(3D, iterator x non-iterator x non-iterator)
        i'th array is the entire correlation matrix, rather than just the diagonal

    Notes
    --------
    BNT.+analyses.populationVectorCorrelation

    Copyright (C) 2019 by Simon Ball
    """
    debug = kwargs.get("debug", False)
    row_major = kwargs.get("row_major", True)
    
    # Perform input validation and ensure we have a pair of 3D arrays
    stack_0, stack_1 = _handle_both_inputs(stack_0, stack_1)
    
    # _handle_ has ensured that both arrays meet the shape/type requirements
    # Hardcode iterating over Y for now. 
    num_cells, y_bins, x_bins = stack_0.shape
    if row_major:
        iterator = y_bins
        non_iterator = x_bins
    else:
        iterator = x_bins
        non_iterator = y_bins
    
    if debug:
        print(f"Number of ratemaps: {num_cells}")
        print(f"Ratemap dimensions: {y_bins} x {x_bins}")
        print(f"Iterating over axis length {iterator} (row_major is {row_major})")

    p1 = np.zeros(iterator)
    p2 = np.zeros((iterator, non_iterator))
    p3 = np.zeros((iterator, non_iterator, non_iterator))

    for i in range(iterator):
        if row_major:
            left = stack_0[:, i, :].transpose()
            right = stack_1[:, i, :].transpose()
        else:
            left = stack_0[:, :, i].transpose()
            right = stack_1[:, :, i].transpose()
        
        # 1D
        # Reshape 2D array to a 1D array
        correlation_value = np.corrcoef(left.flatten(), right.flatten())[0,1]
        p1[i] = correlation_value
        
        # 2D, 3D
        correlation_matrix = np.corrcoef(left, right)[0:non_iterator, non_iterator:]
        p2[i, :] = np.diagonal(correlation_matrix)
        p3[i, :, :] = correlation_matrix

    return (p1, p2, p3)












###############################################################################
#############
#############           Error checking
#############


def _handle_both_inputs(stack_0, stack_1):
    '''Handle error checking across both main inputs'''
    stack_0 = _handle_single_input(stack_0, 0)
    stack_1 = _handle_single_input(stack_1, 1)
    if stack_0.shape[0] != stack_1.shape[0]:
        raise err.ArgumentError("You have a different number of rate maps in each stack.")
    if stack_0.shape[1:] != stack_1.shape[1:]:
        raise err.ArgumentError("Your rate maps do not have matching dimensions")
    return stack_0, stack_1

def _handle_single_input(stack, i):
    '''Handle the input stack(s) and provide a correctly formatted 3D array
    
    Handle error checking for a variety of conditions for a single stack
    If not already a MaskedArray, then convert to that
    
    Parameters
    ----------
    stack : array-like
        One of main inputs to population_vector_correlation.
        Should be either a 3D array, where each layer (stack[j]) is a RateMap,
        OR a list of 2D arrays, where each array is a 2D RateMap. 
        If a list of arrays, all arrays must be the same dimension
    i : int
        Index of stack input, solely used for providing more meaningful error
        message

    Returns
    -------
    stack : np.ma.MaskedArray
        3D array of RateMaps, masked at invalid values
    '''
    dims = None
    t = type(stack)
    if t not in (list, tuple, np.ndarray, np.ma.MaskedArray):
        raise ValueError(f"Stack_{i} must be array-like. You provided {t}")
    elif t in (tuple, list):
        for element in stack:
            e = type(element)
            if e not in (np.ndarray, np.ma.MaskedArray):
                raise err.ArgumentError(f"The elements of the list stack_{i} must be"\
                                 f" NumPy arrays. You provided {e}")
            if dims is None:
                dims = element.shape
            else:
                if element.shape != dims:
                    raise err.ArgumentError(f"Your ratemaps are not a consistent"\
                                     f" shape in stack_{i}")
        # Passes error handling, now convert from list to masked array
        stack = np.ma.masked_invalid(stack)
    elif isinstance(stack, np.ndarray):
        # Ok, but convert to masked array
        stack = np.ma.masked_invalid(stack)
        dims = stack.shape[1:]
    else:
        # Instance is already a Masked Array
        dims = stack.shape[1:]
    return stack


    