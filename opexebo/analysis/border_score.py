"""Provide function for calculating a Border Score"""

import numpy as np
from scipy.ndimage import distance_transform_cdt
try:
    import cv2
    HAS_CV2 = True
except:
    HAS_CV2 = False


from opexebo.analysis import border_coverage
from opexebo.general import validate_keyword_arena_shape, circular_mask
import opexebo.defaults as default


def border_score(rate_map, fields_map, fields, arena_shape, **kwargs):
    '''
    Calculate border score for a firing map
    
    STATUS : EXPERIMENTAL
    
    TODO: Update to account for the use of Vadim's placefield code
    TODO: Handle circular arenas (How?)
    

    Calculates a border score for a firing rate map according to the article 
    "Representation of Geometric Borders in the Entorhinal Cortex" by Solstad 
    et. al. (Science 2008).
    
    Border score is either -1 (no firing fields provided) or in the range [0-1].
    The score reflects both the width of a field (what fraction of a single wall
    it touches), and the depth of a field (how far away from the field it extends)
    The highest score is returned for a field with maximum width and infinitesimal
    depth.
    
    Consequently, a border score of 1 can only **ever** be returned given an 
    infinite resolution. A perfect field in a typical 40x40 bin map has a maximum
    value of around 0.91.
    
    The score is only evaluated for the firing field that has the greatest wall
    coverage. All other fields are ignored. 
    
    Parameters
    ----------
    rate_map:   np.ma.MaskedArray
        rate map: N x M array where each cell value is the firing rate of the cell
    fields_map:   np.ma.MaskedArray
        Integer array of labelled fields. Each cell value is a positive integer. 
        Cells that are members of field have the value corresponding to field_id 
        (non-zero positive  unique integers. Not necessarily contiguous 
        (e.g. 1, 2, 3, 5)). Cells that are not members of fields have value zero
    fields:  list of  dict
        List of dictionaries of firing fields. 
        Each dictionary must, at least, contain the keyword "field_map", yielding a 
        binary map of that field within the overall arena
    arena_shape: {"square", "rect", "circle", "line"}
        Rectangular and square are equivalent. Elliptical or n!=4 polygons
        not currently supported.
    search_width: int, optional
        rate_map and fields_map have masked values, which may occur within the
        region of border pixels. To mitigate this, we check rows/columns within
        search_width pixels of the border Default `8`
    walls: str or list of tuple, optional
        Definition of which walls to consider for border coverage. Behaviour is
        different for different arena shapes
        * Rectangular arenas
          type str. The four walls are referred to as `T`, `B`, `R`, `L` for the
          standard cardinals. Case insensitive. You may include or exclude any
        * Circular arenas
          Walls should be specified as a list of inclusive angular start-stop pairs. Angles
          given in degrees clockwise from top. Both `0` and `360` are permissible
          values. Start-stop may wrap through zero and may be overlapping with
          other pairs. Negative angles may be used to express counter-clockwise
          rotation, and will be automatically wrapped into the [0, 360] range.
          Example: [(0, 15), (180, 270), (315, 45), (30, 60)]
          Walls may ALTERNATIVELY be expressed as a string, equivalent to rectangular
          arenas. In this case, walls will be interpreted as follows
              * `T` - (315, 45)
              * `B` - (135, 225)
              * `L` - (45, 135)
              * `R` - (225, 315)
        In either case, defaults `TRBL`. Converting this (or other string) for
        circluar arrays is handled internally

    Returns
    -------
    score: float
        Border score in the range [-1, 1].
        A value of -1 is given when no fields are provided
    coverage: float
        Border coverage in the range [0, 1]
        The coverage is given for ONE border. Coverage for a specific wall is
        only available if a single wall is searched

    See Also
    --------
    opexebo.analysis.placefield
    opexebo.analysis.bordercoverage

    Notes
    -----
    * BNT.+analyses.placefield
    * BNT.+analyses.borderScore
    * BNT.+analyses.borderCoverage
        
    Copyright (C) 2019 by Simon Ball
    '''
    arena_shape = validate_keyword_arena_shape(arena_shape)

    # Check that fields exist
    if np.ma.max(fields_map) == 0:
        # No fields exist
        return -1, 0
    
    # 1 or more fields exist
    fields_map_unlabelled = np.copy(fields_map)
    fields_map_unlabelled[fields_map> 1 ] = 1

    coverage = border_coverage(fields, arena_shape, **kwargs)

    fields_rate_map = fields_map_unlabelled * rate_map
    fields_rate_map = np.ma.masked_invalid(fields_rate_map)

    wfd = _weighted_firing_distance(fields_rate_map, arena_shape)

    score = (coverage - wfd)/(coverage + wfd)
    return score, coverage



###############################################################################
####            Border Score helper functions
###############################################################################


def _weighted_firing_distance(rmap, arena_shape):
    '''
    Calculate the weighted firing distance
    
    A perfect border cell is one that fires only (and always) when the animal is
    imemdiately adjacent to a border. This function weights the firing rate
    reported in the ratemap by its proximity to the border.

    Parameters
    ----------
    rmap: np.ma.MaskedArray
        Rate map of cells inside fields. NxM array where cells inside fields
        give the firing rate, and cells outside fields are zero
    Returns
    -------
    wfd: float
        Weighted firing distance. 
    '''
    # Check that the provided array is properly masked
    if type(rmap != np.ma.MaskedArray):
        rmap = np.ma.masked_invalid(rmap)

    # Normalise firing map by the sum of the firing map, such that it resembles a PDF
    pdf = rmap / np.ma.sum(rmap)

    # Create an array, same size as rmap, where the cell value is the distance (in bins) from the edge of the array:
    x = np.ones(rmap.shape) # define area same size as ratemap
    x = np.pad(x, 1, mode='constant') # Pad outside with zeros, we calculate distance from nearest zero
    dist = distance_transform_cdt(x, metric='taxicab') # taxicab metric: distance along axes, not diagonal distance. 
    #wfd = np.ma.sum( np.ma.dot(dist[1:-1, 1:-1], rmap) )
    wfd = np.ma.sum( dist[1:-1, 1:-1] * pdf )

    # Normalise by half of the smallest arena dimensions
    wfd = 2 * wfd / np.min(rmap.shape)
    return wfd



def _identify_border(rmap, arena_shape, debug=False):
    '''
    Identify the border based on the arena_shape. This is needed to cope with
    circular arenas
    
    Parameters
    ----------
    rmap: np.ndarray or np.ma.MaskedArray
        Rate map of the unit
    arena_shape: {"square", "rect", "circle", "line"}
        Rectangular and square are equivalent. Elliptical or n!=4 polygons
        not currently supported.
    
    Returns
    -------
    dict
        mask: np.ndarray
            Boolean array. `True` inside arena, `False` outside
        distance: np.ndarray, optional
            [Circular arenas only] Array of distances from the border
        angle: np.ndarray, optional
            [Circular arenas only] Array of angles from the centre. 
    '''
    out = {}
    if arena_shape in default.shapes_square:
        mask = np.ones(rmap.shape, dtype=bool)
        out["mask"] = mask
    elif arena_shape in default.shapes_circle:        
        if not HAS_CV2:
            raise ImportError("You must install OpenCV to use this function: `pip install opencv-python`")
        rmap_threshold = np.uint8(rmap>0)
        centres = cv2.findContours(rmap_threshold, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[-2]
        x = y = radius = 0
        for c in centres:
            (x_, y_), r_ = cv2.minEnclosingCircle(c)
            if r_ > radius:
                x = x_
                y = y_
                radius = r_
        axes = [np.arange(0, rmap.shape[i]+1, 1) for i in [0,1]]
        mask, distance, angle = circular_mask(axes, radius*2, origin=(x, y))
        out["mask"] = mask
        out["distance"] = radius - distance
        out["angle"] = angle
        if debug:
            print(f"radius (bins): {radius}")
    else:
        raise NotImplementedError("arena shape '{}' not supported".format("arena_shape"))
    return out
        