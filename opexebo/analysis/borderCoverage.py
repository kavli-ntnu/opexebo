"""Provide function for calculating a Border Score"""

import numpy as np
from scipy.ndimage import distance_transform_edt
import cv2
# TODO TODO TODO
import matplotlib.pyplot as plt

import opexebo.defaults as default
from opexebo.general import validate_keyword_arena_shape, circular_mask
from opexebo.errors import ArgumentError


def border_coverage(fields, arena_shape, **kwargs):
    '''
    Calculate border coverage for detected fields.
    
    STATUS : EXPERIMENTAL
    
    This function calculates firing map border coverage that is further
    used in calculation of a border score. Coverage is calculated on a per-field
    basis, and is therefore dependent on the exact details of the place-field
    detection algorithm. 
    
    Additionally, please be aware of what `top` and `bottom` mean. This function
    treats `top` as the location where the `y` indicies are greatest. This is
    the standard convention in a GRAPH ((0,0) at bottom left, `x` increases to
    right and `y` increases to top). Images are conventionally plotted UPSIDE-
    DOWN, with `y` increasing towards the bottom.

    Parameters
    ----------
    fields: dict or list of dicts
        One dictionary per field. Each dictionary must contain the keyword "map"
    arena_shape: {"square", "rect", "circle", "line"}
        Rectangular and square are equivalent. Elliptical or n!=4 polygons
        not currently supported. Defaults to Rectangular
    search_width: int
        rate_map and fields_map have masked values, which may occur within the
        region of border pixels. To mitigate this, we check rows/columns within
        search_width pixels of the border Default `8`
    walls: str or list of tuple
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
    coverage: float
        Border coverage, ranges from 0 to 1.

    See Also
    --------
    opexebo.analysis.placefield
    opexebo.analysis.borderscore

    Notes
    --------
    * BNT.+analyses.placefield
    * BNT.+analyses.borderScore
    * BNT.+analyses.borderCoverage

    Copyright (C) 2019 by Simon Ball
    '''
    # Process input arguments
    debug= kwargs.get("debug", False)
    arena_shape = validate_keyword_arena_shape(arena_shape)
    
    if arena_shape in default.shapes_square:
        calculate_coverage = _calculate_coverage_rect
    elif arena_shape in default.shapes_circle:
        calculate_coverage = _calculate_coverage_circ
    else:
        raise NotImplementedError(f"Border coverage is not implemented for arena shape `{arena_shape}`")
    
    if isinstance(fields, dict):
        # Deal with the case of being passed a single field, instead of a list of fields
        fields = [fields]
    elif not isinstance(fields, (list, tuple, np.ndarray)):
        raise ValueError(f"You must supply either a dictionary, or list of dictionaries, of fields. You provided type '{type(fields)}'")
    for i, field in enumerate(fields):
        if "map" not in field.keys():
            raise KeyError(f"field dictionary {i} does not have keyword 'map'.")

    # Extract keyword arguments or set defaults
    search_width = kwargs.get('search_width', default.search_width)
    walls = kwargs.get('walls', default.walls)

    # Perform validation of the `walls` argument
    walls = _validate_keyword_walls(walls, arena_shape)

    if debug:
        print("===== border_coverage =====")
        print(f"arena_shape: {arena_shape}")
        print(f"coverage method: {calculate_coverage}")
        print(f"walls: {walls}")
        print(f"num fields: {len(fields)}")

    coverage = 0
    for field in fields:
        fmap = field["map"]
        for wall in walls:
            c = calculate_coverage(fmap, wall, search_width, debug)
            coverage = max(coverage, c)
    
    return coverage


###############################################################################
####            Helper functions : CIRCULAR arenas
###############################################################################



def _calculate_coverage_circ(fmap, wall, search_width, debug):
    '''
    Evaluate what fraction of the border area is covered by a single field in a
    circular arena
    
    Problems to solve:
        * Identify where the border actually is, since we can't just use the
          edge of the array
        * Convert the cartesian co-ordinates of the field map to polar co-ordinates
        * Isolate the relevant arc
    First point: we can _start_ by assuming that the circle is centred on the
    centre of the array. I'm not convinced that this is a good approximation, but
    it's a place to start
    
    Parameters
    ----------
    fmap : np.ndarray or np.ma.MaskedArray
        Binary map of specific firing field
    '''
    if not isinstance(fmap, np.ma.MaskedArray):
        fmap = np.ma.asanyarray(fmap)
    fmap = fmap.copy() # create a separate copy to work on, to avoid propagating changes back outside this function
    
    # Create a border wall mask, and a distance map
    # use cv2 to 
    raise NotImplementedError
    # Below - using cv2 to find the minimum enclosing circular contour
    # The trouble is this only works on the _entire_ ratemap (and it seems to work fairly well)
    # It won't work at all on single fields. 
    # At least it would need to be run on the entire, summed, fields_map. 
    # Better still, it would be run on the ratemap. 
    '''
    fmap_threshold = np.uint8(fmap>0)
    centres = cv2.findContours(fmap_threshold, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[-2]
    x = y = radius = 0
    for c in centres:
        (x_, y_), r_ = cv2.minEnclosingCircle(c)
        if r_ > radius:
            x = x_
            y = y_
            radius = r_
    axes = [np.arange(0, fmap.shape[i]+1, 1) for i in [0,1]]
    circle_mask, distance, angle = circular_mask(axes, radius*2, origin=(x, y))
    
    if debug:
        print(f"radius (bins): {radius}")
    '''
    
    

    
    # Circular_mask gives us nearly everything we need: an array of distances (in floating-point bins)
    # and a separate array of angles (in degrees, with 0 pointing to y_max, and 90° pointing to x_max)
    # We can use these as lookups to convert to polar co-ordinates
    
    
    # Construct a mask for locations we do care about, based on angle, 
    # distance_from_border, and whether it is inside the circle
    if wall[0] > wall[1]:
        good_arc = np.logical_or(np.logical_and(wall[0] <= angle,
                                                angle <= 360),
                                 np.logical_and(0 <= angle,
                                                angle <= wall[1]))
    else:
        good_arc = np.logical_and(wall[0] <= angle,
                                  angle <= wall[1])
    
    border_distance = radius - distance
    good_distance = np.logical_and(0 <= border_distance,
                                   border_distance <= search_width)
    
    in_region = np.logical_and(circle_mask, good_arc, good_distance)
    not_in_region = np.logical_not(in_region)
    
    # Using this mask, blank out all the locations that are not relevant
    # "Blanking out" the border_distance array , means set it to high values, not low
    fmap[not_in_region] = 0
    border_distance[not_in_region] = np.max(border_distance)
    
    # Plan
        # Iterate across all locations within `in_region`, low to high, sorted by `border_distance`
        # Locations which round (floor) down to zero will potentially count towards coverage
            # To count towards coverage, they must be part of a field
            # Alternatively, if they are NaN in `fmap` (never visited), then find the next nearest in along that angle, and repeat the check




    def _circular_inwards_search(idx, i):
        '''
        If an unvisited location is found, recursively search inwards until either
        search_width is reached, or a visited location is found. If a visited location
        is found, at nearly the same angle, then swap the values in field_map
        
        This recursive loop is only started where `border_distance_int`==0, so start
        searching where `border_distance_int` == 1, and recursively allow it to rise, 
        until the value at the closest angle is NOT a NaN
        '''
        nonlocal fmap
        nonlocal border_distance
        nonlocal border_distance_int
        nonlocal angle
        nonlocal in_region
        nonlocal search_width
        
        if i == search_width:
            # Recurse at most `search_width` times
            return None
        
        locations_to_search = (border_distance_int == i)
        try:
            idx_2 = np.argmin(np.abs(angle[locations_to_search] - angle[idx]))
        except ValueError:
            # if the sequence is empty, try the next i
            _circular_inwards_search(idx, i+1)
        if np.isnan(fmap[idx_2]):
            # If this location is NaN as well, try the next i
            _circular_inwards_search(idx, i+1)
        else:
            # If not NaN, then swap the values, and return to the main loop
            fmap[idx] = fmap[idx_2]


    
    border_distance_int = np.floor(border_distance)
    # sort by border_distance and then work with co-ordinates to index between the three arrays (fmap, border_distance, angle)
    locations_to_search =  np.argwhere(border_distance_int == 0)
    if debug:
        fig, ax = plt.subplots(ncols=2)
        ax[0].imshow(fmap)
        ax[1].imshow(border_distance_int)

    for row, col in locations_to_search:
        idx = (row, col)
        if np.isnan(fmap[idx]):
            # Do some shuffling
            # TODO
            _circular_inwards_search(idx, 1)
            pass
        if fmap[idx]:   # i.e. is part of the field, after shuffling
            # do nothing, leave it as a zero for the final count
            pass
        else:           # i.e. not part of the field
            # Set the value in border_distance_int to non-zero, so it will not be counted
            border_distance_int[idx] = np.max(border_distance_int)
    
    # Coverage is determined by the ratio of locations with distance 0 to the number covered by the field in question    
    coverage = np.count_nonzero(border_distance_int==0) / np.max(locations_to_search.shape)
            
    return coverage
    
    
    
###############################################################################
####            Helper functions : RECTANGULAR arenas
###############################################################################


def _calculate_coverage_rect(fmap, wall, search_width, debug):
    '''
    Evaluate what fraction of the border area is covered by a single field in a
    rectangular or square arena

    Border coverage is provided as two values: 
        covered : the sum of the values across all sites immediately adjacent 
            to the border, where the values are calculated from the distance of
            those sites to the firing field, excluding NaN, inf, and masked values
        norm    : the number of non nan, inf, masked values considered in the 
            above sum

    The border area is defined by wfmap - this is the subsection of the binary 
    firing map of a  single field that lies within search_width of a border. 
    wfmap is must be of size NxM where
        N : arena_size / bin_size
        M : search_width
    and where the 0th column (wfmap[:,0], len()=N) repsents the sites closest 
    to the border

    fmap: has value 1 inside the field and 0 outside the field
    
    Parameters
    ----------
    fmap : np.ndarray or np.ma.MaskedArray
        Binary map of a sngle firing field (as supplied by `opexebo.analysis.place_field`).
    wall : str
        Which wall to consider for coverage. Call this function multiple times
        to consider multiple walls
    search_width : int
        Width of the region from the border inwards to consider for calculation
        
    
    Returns
    -------
    coverage : float
        Coverage of the relevant border. Coverage is maximised for an infinitely
        narrow field extending over the entire width of the border. Thicker fields
        that cover less of the border width have reduced coverage. 
    '''
    # fmap is the input field
    # wfmap is the isolated and rotated section to search
    wfmap = _get_aux_map_rect(fmap, wall, search_width)
    
    if type(wfmap) != np.ma.MaskedArray:
        wfmap = np.ma.asanyarray(wfmap)
    wfmap[np.ma.greater(wfmap, 1)] = 1 # Just in case a still-labelled map has crept in
    inverted_wfmap = 1 - wfmap 
    distance = distance_transform_edt(np.nan_to_num(inverted_wfmap.data, copy=True)) # scipy doesn't recognise masks
    distance = np.ma.masked_where(np.ma.getmaskarray(wfmap), distance) # Preserve masking
    # distance_transform_edt(1-map) is the Python equivalent to (Matlab bwdist(map))
    # Cells in map with value 1 go to value 0
    # Cells in map with value 0 go to the geometric distance to the nearest value 1 in map

    adjacent_sites = distance[:,0]
    # Identify sites which are NaN, inf, or masked
    # Replace them with the next cell along the row, closest to the wall, that is not masked, nan, or inf
    adjacent_sites.mask += np.isnan(adjacent_sites.data)
    if adjacent_sites.mask.any():
        for i, rep in enumerate(adjacent_sites.mask):
            if rep:
                for j, val in enumerate(distance[i,:]):
                    if not distance.mask[i,j] and not np.isnan(val):
                        adjacent_sites[i] = val
                        adjacent_sites.mask[i] = False
                        break
    covered = np.ma.sum(adjacent_sites==0)
    contributing_cells = wfmap.shape[0] - np.sum(adjacent_sites.mask) # The sum gives the number of remaining inf, nan or masked cells
    coverage = covered / contributing_cells
    return coverage



def _get_aux_map_rect(fmap, wall, search_width):
    '''
    Isolate the appropriate section of a map (determined by `search_width`), and
    rotate it so that the relevant portion, and the border are at left most 
    (i.e. minimum `x` index)
    
    !! WARNING !! This function treats "top" as "the values where the `y` index is greatest.
    THIS DOES NOT NECESSARILY MEAN THE TOP OF THE IMAGE!
    
    Conventionally, a graph is plotted with (x=0, y=0) at BOTTOM left. X increases
    towards the right, and Y increases towards the top
    
    Conventionally an image is plotted flipped up/down, with (0,0) at TOP left,
    and Y increases towards the BOTTOM
    '''
    # Get our "auxiliary" map from the provided firing map
    # That means pick out the particular section determined by `search_width`
    # And rotate as determined by `wall`, so that the relevant border and pixels 
    # are always at the left-most of the image
    
    if wall == "l":
        aux_map = fmap[:,:search_width].copy()
    elif wall == "r":
        # Mirror image to match the expectations in _wall_field, i.e. border adjacent to left-most column
        aux_map = fmap[:, -search_width:].copy()
        aux_map = np.fliplr(aux_map)
    elif wall == "t":
        # rotate 90 deg counter clockwise (bottom to right), then mirror image
        aux_map = fmap[-search_width:, :].copy()
        aux_map = np.rot90(aux_map)
        aux_map = np.fliplr(aux_map)
    elif wall == "b":
        # Rotate counterclockwise - top of image moves to left of image
        aux_map = fmap[:search_width, :].copy()
        aux_map = np.rot90(aux_map)
    else:
        # This should never happen
        raise ValueError
    return aux_map





###############################################################################
####            Helper functions : Misc
###############################################################################

def _validate_keyword_walls(walls, shape):
    '''Parse the walls keyword to be sure that it is valid and conforms to requirements.
    Logic is complicated due to the wildly diverging approach of the square and circular arena shapes
    
    Parameters
    ----------
    walls : keyword value given to border_coverage `walls`
    shape : keyword value given to border_coverage `arena_shape`
    
    Returns
    -------
    walls : str or list of np.ndarrays
        Appropriate form given the `shape` parameter
    
    '''
    
    if isinstance(walls, str):
        # do string-related stuff
        # Check that only acceptable characters are included
        # Remove duplicates and set to lowercase
        walls = "".join(sorted(set(walls.lower())))
        # Note: set() can break ordering. sorting still breaks ordering, but makes it consistent
        if not (0 < len(walls) <= 4):
            raise ValueError("keyword `walls` as a string must contain 1-4 characters."\
                             f" You provided {len(walls)} characters")
        for char in walls:
            if char not in ["t","r","b","l"]:
                 raise ValueError("Character %s is not a valid entry in wall"\
                                  " definition. Valid characters are [t, r, b, l]" % char)
        # If the arena_shape is circular, convert this to the requisite angular form
        if shape in default.shapes_circle:
            walls_new = []
            for char in walls:
                if char == "t":
                    walls_new.append((315, 45))
                elif char == "b":
                    walls_new.append((135, 225))
                elif char == "l":
                    walls_new.append((45, 135))
                elif char == "r":
                    walls_new.append((225, 315))
            walls = walls_new

    elif isinstance(walls, (list, tuple, np.ndarray)):
        if shape in default.shapes_square:
            raise ArgumentError("Keyword `walls` is the wrong type for a rectangular"\
                                " arena. You must provide a string definition")
        for i, element in enumerate(walls):
            if not isinstance(element, (list, tuple, np.adarray)):
                raise ValueError("Keyword `walls` as an array-like must contain array-like"\
                                 f" start-stop pairs. Element {i} has type `{type(element)}`")
            if not len(element) == 2:
                raise ValueError("Keyword `walls` as an array-like must contain array-like"\
                                 f" start-stop pairs. Element {i} has length `{len(element)}`")
            # Ensure that values are wrapped into [0, 360]
            # Ensure that values are ordered (smaller, larger)
            new_element = np.array(sorted(element)) % 360
            walls[i] = new_element

    else:
        raise ArgumentError("Keyword `wall` definition is the wrong type. You"\
                            "must provide a string (for rectangular or circular arrays),"\
                            " or a list of angular start-stop lists for circular arrays")

    return walls