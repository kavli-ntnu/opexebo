
import numpy as np

def circular_mask(axes, diameter, **kwargs):
    '''Given a set of axes, construct a circular mask defining pixels as within
    or without a circular arena.
    
    A pixel is assumed to be inside the circle iff the centre-point is within
    the radius of the circle. For small radii, this will result in including
    pixels with <50% of their area inside. 
    
    Parameters
    ----------
        axes: list of np.ndarray
            [`x`, `y`] - pair of nd-arrays defining the edges of the pixels. This
            is the return value from np.histogram2d, for instance. 
            Each one is size+1 relative to the number of pixels
            This function assumes that each dimension has a CONSTANT bin width,
            although each dimension may have its own bin width. 
       diameter: float
           diameter of the circle
       origin: list of floats
           [`x`, `y`] co-ordinates of the centre of the arena. Optional,
           defaults to `(0,0)`
    
    Returns
    -------
    in_field: np.ndarray
        2D boolean array of dimensions `(axes[0].size-1, axs[1].size-1)`
        Values are `true` if INSIDE circle and `false` if OUTSIDE
    distance_map: np.ndarray
        2D float array of same size. Values are the distance of the centre of
        the pixel from the origin
    '''
    origin = kwargs.get("origin", (0,0))
    radius = diameter / 2
    bin_width = [np.mean(np.diff(ax)) for ax in axes]
    x_centres = axes[0][:-1] + (bin_width[0] / 2) - origin[0]
    y_centres = axes[1][:-1] + (bin_width[1] / 2) - origin[1]
    X, Y = np.meshgrid(x_centres, y_centres)
    distance_map = np.sqrt(np.power(X,2) + np.power(Y,2))
    in_field = distance_map<=radius
    return in_field, distance_map