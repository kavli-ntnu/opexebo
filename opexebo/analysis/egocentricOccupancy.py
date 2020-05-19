'''
Provies a function to generate egocentric maps from tracking/spike data
'''
import numpy as np

from opexebo import defaults as default



def egocentric_occupancy(positions, head_angles, num_angles, num_bins, boundaries, arena_shape):
    '''Generate a 2D histogram of (angle, border_distance) data . Can calculate either allocentric
    or egocentric values. To calculate allocentric (=fixed reference), set head_angles to None
    
    Implementation is based on, but indepdent of, https://doi.org/10.1016/j.cub.2019.07.007
    
    This is a moderately general function: it will work for either the tracking
    or spike data. However, it is sensible to enforce that the `boundaries` keyword
    uses the same value for both sides if the goal is to generate a meaningful ratemap
    
    Parameters
    ----------
    position: np.ndarray
        array of (x, y) positions of animal
    head_angle: np.ndarray -or- None
        Angle [RADIANS] of the animal's head with respect to the X axis
        Zero radians points along the x axis. 
        Set to None to calculate the allocentric values
    num_angles: int
        Number of angles at which to calculate the wall distance
    num_bins:
        Number of bins to consider wall distance
    boundaries: tuple
        Some way of defining where the boundary of the arena is
        if `arena_shape` is `square`, then boundaries should be:
            ( (min_x, max_x), (min_y, max_y) )
        elif `arena_shape` is `circ`, then boundaries should be:
            ( (centre_x, centre_y), radius)
    arena_shape: str
        definition of the shape of the arena, ties in to format of `boundaries`
        Acceptable values are given in `opexebo.defaults.shapes_*
    
    Returns
    -------
    hist: np.ndarray
        2D histogram of (wall_distance, angle) data
        hist[0, :] are all the distance bins for a single angle
        hist[:, 0] are all the angles for a single distance bin
    distance_bins: np.ndarray
        Distance bins, +1 element longer than hist[0, :].shape
    angle_bins: np.ndarray
        Angular bins, +1 element longer than hist[:, 0].shape   
    '''
    if positions.shape[0] == 2:
        # The easiest way to generate positions is `np.array((pos_x, pos_y))`, but that is the
        # transpose of what we want - so handle this automatically
        positions = positions.T
    
    # Create our output data structure
    num_positions = positions.shape[0]
    distances = np.zeros((num_positions, num_angles))
    
    # Create our target angle centres
    target_angles = (np.arange(num_angles) * 2 * np.pi / num_angles)
    target_angles += 0.5*np.diff(target_angles)
    
    if not isinstance(head_angles, np.ndarray):
        if isinstance(head_angles, (tuple, list)):
            head_angles = np.array(head_angles)
        elif head_angles is None:
        # Shortcut to let the user specify allocentric calculation
        # In effect, we do every calculation with a head_angle of 0
            head_angles = np.zeros(num_positions)
    
    # Calculate the wall distances
    for i, pos in enumerate(positions):
        distances[i] = _wall_distance(pos, head_angles[i], target_angles, boundaries, arena_shape)
    
    # Generating a histogram is a bit weird here, because our data is semi-histogrammed
    # already, i.e. the target angles have already been discretised. 
    # Thus, we call 1D histogram repeatedly, once for each target_angle
    
    # Calculate our histogram limits:
    # distance can be from 0 to the maximum diagonal length (diameter for a circle)
    min_bin = 0
    if arena_shape in default.shapes_square:
        max_bin = np.sqrt(np.square(boundaries[0][1] - boundaries[0][0]) + 
                          np.square(boundaries[1][1] - boundaries[1][0]))
    elif arena_shape in default.shapes_circle:
        max_bin = boundaries[1] * 2
    
    distance_bins = np.linspace(min_bin, max_bin, num_bins+1) # 1 longer because it lists the (start, end) of each bin, not just the centre
    angle_bins = np.linspace(0, 2*np.pi, num_angles+1)
    
    data = np.zeros((num_angles, num_bins))
    for i, tad in enumerate(distances.T):
        # we wrote this with each row being a single frame
        # but now we want to iterate over the tracking angles
        # hence transposing
        data[i, :], distance_bins = np.histogram(tad, distance_bins)
    return data, distance_bins, angle_bins
        
    
    
        


def _wall_distance(position, head_angle, target_angles, boundaries, arena_shape):
    '''For a single position and head angle, generate an array of border-distances
    for a given list of target angles. 
    
    Parameters
    ----------
    position: tuple
        (x, y) position of animal
    head_angle: float
        Angle [RADIANS] of the animal's head with respect to the X axis
        Zero radians points towards positive x values
        To calculate the allocentric values, set to zero
    target_angles: np.ndarray
        List of angles [RADIANS] at which to calculate the distance-to-wall
    boundaries: ???
        Some way of defining where the boundary of the arena is
        if `arena_shape` is `square`, then boundaries should be:
            ( (min_x, max_x), (min_y, max_y) )
        elif `arena_shape` is `circ`, then boundaries should be:
            ( (centre_x, centre_y), radius)
    arena_shape: str
        definition of the shape of the arena, ties in to format of `boundaries`
    
    Returns
    -------
    np.ndarray
        Distances to bounding wall at egocentric angles defined by `target_angles`
    '''    
    distances = np.zeros(target_angles.size)
    x, y = position
    target_angles = (target_angles + head_angle)%(2*np.pi)
    
    if arena_shape in default.shapes_square:
        '''Optimised array operation for square arenas
        Expects to recieve `boundaries` as a tuple of tuples
        ( ( min(x), max(x)), (min(y), max(y)) )
        Ca. 75x speedup over the for-loop version'''
        bounds_x, bounds_y = boundaries
        
        # Calculate the relevant boundary in x and y for each angle
        x_bounds = np.zeros(target_angles.shape)
        x_criteria = (np.abs(target_angles-np.pi)) < (np.pi*0.5)
        x_bounds[x_criteria] = bounds_x[0]
        x_bounds[np.logical_not(x_criteria)] = bounds_x[1]
        
        y_bounds = np.zeros(target_angles.shape)
        y_criteria = target_angles > np.pi
        y_bounds[y_criteria] = bounds_y[0]
        y_bounds[np.logical_not(y_criteria)] = bounds_y[1]
        
        # Calculate the distance to the vertical [0] and horizontal [1] boundaries
        temp_distances = np.zeros((2, target_angles.size))
        temp_distances[0] = (x_bounds - x) / np.cos(target_angles)
        temp_distances[1] = (y_bounds - y) / np.sin(target_angles)
        
        # Take the shortest value as the actual distance
        distances = np.min(temp_distances, axis=0)
        
    elif arena_shape in default.shapes_circle:
        raise NotImplementedError("Circular arenas are not currently supported")
    else:
        raise NotImplementedError(f"Arena Shape '{arena_shape}' are not currently supported")
    return distances