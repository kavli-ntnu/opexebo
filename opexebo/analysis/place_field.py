"""
Provide function for 2D placefield detection.
"""

import numpy as np
from scipy import ndimage
from skimage import measure, morphology

import opexebo
import opexebo.defaults as default
import opexebo.errors as err


def place_field(firing_map, **kwargs):
    '''
    Locate place fields on a firing map.

    Identifies place fields in 2D firing map. Placefields are identified by
    using an adaptive threshold. The idea is that we start with a peak value as
    the threshold. Then we gradually decrease the threshold until the field
    area doesn't change any more or the area explodes (this means the threshold
    is too low).

    Parameters
    ----------
    firing_map: np.ndarray or np.ma.MaskedArray
        smoothed rate map.
        If supplied as an np.ndarray, it is assumed that the map takes values
        of np.nan at locations of zero occupancy. If supplied as an np.ma.MaskedArray,
        it is assumed that the map is masked at locations of zero occupancy
    
    Other Parameters
    ----------------
    min_bins: int
        Fields containing fewer than this many bins will be discarded. Default 9
    min_peak: float
        Fields with a peak firing rate lower than this absolute value will
        be discarded. Default 1 Hz
    min_mean: float
        Fields with a mean firing rate lower than this absolute value will
        be discarded. Default 0 Hz
    init_thresh: float
        Initial threshold to search for fields from. Must be in the range [0, 1].
        Default 0.96
    search_method: str
        Peak detection finding method. By default, use `skimage.morphology.local_maxima`
        Acceptable values are defined in `opexebo.defaults`. Not required if 
        peak_coords are provided
    peak_coords: array-like
        List of peak co-ordinates to consider instead of auto detection. [y, x].
        Default None

    Returns
    -------
    fields: list of dict
        coords: np.ndarray
            Coordinates of all bins in the firing field
        peak_coords: np.ndarray
            Coordinates peak firing rate [y,x]
        centroid_coords: np.ndarray
            Coordinates of centroid (decimal) [y,x]
        area: int
            Number of bins in firing field. [bins]
        bbox: tuple
            Coordinates of bounding box including the firing field
            (y_min, x_min, y_max, y_max)
        mean_rate: float
            mean firing rate [Hz]
        peak_rate: float
            peak firing rate [Hz]
        map: np.ndarray
            Binary map of arena. Cells inside firing field have value 1, all
            other cells have value 0
    fields_map : np.ndarray
        labelled integer image (i.e. background = 0, field1 = 1, field2 = 2, etc.)

    Raises
    ------
    ValueError
        Invalid input arguments
    NotImplementedError
        non-defined peack searching methods

    Notes
    --------
    BNT.+analyses.placefieldAdaptive

    https://se.mathworks.com/help/images/understanding-morphological-reconstruction.html

    Copyright (C) 2018 by Vadim Frolov, (C) 2019 by Simon Ball, Horst Obenhaus
    '''
    ##########################################################################
    #####                   Part 1: Handle inputs
    # Get keyword arguments
    min_bins = kwargs.get("min_bins", default.firing_field_min_bins)
    min_peak = kwargs.get("min_peak", default.firing_field_min_peak)
    min_mean = kwargs.get("min_mean", default.firing_field_min_mean)
    init_thresh = kwargs.get("init_thresh", default.initial_search_threshold)
    search_method = kwargs.get("search_method", default.search_method)
    peak_coords = kwargs.get("peak_coords", None)
    debug = kwargs.get("debug", False)

    if not 0 < init_thresh <= 1:
        raise err.ArgumentError("Keyword 'init_thresh' must be in the range [0, 1]."\
                         f" You provided {init_thresh}")
    try:
        search_method = search_method.lower()
    except AttributeError:
        raise err.ArgumentError("Keyword 'search_method' is expected to be a string"\
                         f" You provided a {type(search_method)} ({search_method})")
    if search_method not in default.all_methods:
        raise err.ArgumentError("Keyword 'search_method' must be left blank or given a"\
                         f" value from the following list: {default.all_methods}."\
                         f" You provided '{search_method}'.")

    global_peak = np.nanmax(firing_map)
    if np.isnan(global_peak) or global_peak == 0:
        if debug:
            print(f"Terminating due to invalid global peak: {global_peak}")
        return [], np.zeros_like(firing_map)

    # Construct a mask of bins that the animal never visited (never visited -> true)
    # This needs to account for multiple input formats.
    # The standard that I want to push is that firing_map is type MaskedArray
        # In this case, the cells that an animal never visited have firing_map.mask[cell]=True
        # while firing_map.data[cell] PROBABLY = 0
    # An alternative is the BNT standard, where firing_map is an ndarray
        # In this case, the cells never visited are firing_map[cell] = np.nan
    # In either case, we need to get out the following:
        # finite_firing_map is an ndarray (float) where unvisted cells have a
        # meaningfully finite value (e.g. zero, or min())
        # mask is an ndarray (bool) where unvisited cells are True, all other cells are False

    if isinstance(firing_map, np.ma.MaskedArray):
        occupancy_mask = firing_map.mask
        finite_firing_map = firing_map.data.copy()
        finite_firing_map[np.isnan(firing_map.data)] = 0

    else:
        occupancy_mask = np.zeros_like(firing_map).astype('bool')
        occupancy_mask[np.isnan(firing_map)] = True
        finite_firing_map = firing_map.copy()
        finite_firing_map[np.isnan(firing_map)] = 0

    structured_element = morphology.disk(1)
    image_eroded = morphology.erosion(finite_firing_map, structured_element)
    fmap = morphology.reconstruction(image_eroded, finite_firing_map)
    
    ##########################################################################
    #####                   Part 2: find local maxima
    # Based on the user-requested search method, find the co-ordinates of local maxima
    if peak_coords is None:
        if search_method == default.search_method:
            peak_coords = opexebo.general.peak_search(fmap, **kwargs)
        elif search_method == "sep":
            #fmap = finite_firing_map
            peak_coords = opexebo.general.peak_search(fmap, **kwargs)
        else:
            raise NotImplementedError("The search method you have requested (%s) is"\
                                      " not yet implemented" % search_method)

    # obtain value of found peaks
    found_peaks = finite_firing_map[peak_coords[:, 0], peak_coords[:, 1]]

    # leave only peaks that satisfy the threshold
    good_peaks = (found_peaks >= min_peak)
    peak_coords = peak_coords[good_peaks, :]


    ##########################################################################
    #####    Part 3: from local maxima get fields by expanding around maxima
    max_value = np.max(fmap)
    # prevent peaks with small values from being detected
    # SWB - This causes problems where a local peak is next to a cell that the animal never went
    # As that risks the field becoming the entire null region
    # Therefore, adding 2nd criterion to avoid adding information where none was actually known.
    fmap[np.logical_and(fmap < min_peak, fmap > 0.01)] = max_value * 1.5

    # this can be confusing, but this variable is just an index for the vector
    # peak_linear_ind
    peaks_index = np.arange(len(peak_coords))
    fields_map = np.zeros(fmap.shape, dtype=np.integer)
    field_id = 1
    for i, peak_rc in enumerate(peak_coords):
        # peak_rc == [row, col]

        # select all peaks except the current one
        other_fields = peak_coords[peaks_index != i]
        if other_fields.size > 0:
            other_fields_linear = np.ravel_multi_index(
                        multi_index=(other_fields[:, 0], other_fields[:, 1]),
                        dims=fmap.shape, order='F')
        else:
            other_fields_linear = []

        used_th = init_thresh
        res = _area_change(fmap, occupancy_mask, peak_rc, used_th,
                           used_th-0.02, other_fields_linear)
        initial_change = res['acceleration']
        area2 = res['area2']
        first_pixels = np.nan
        if np.isnan(initial_change):
            for j in np.linspace(used_th+0.01, 1., 4):
                # Thresholds get higher, area should tend downwards to 1
                # (i.e. only including the actual peak)
                res = _area_change(fmap, occupancy_mask, peak_rc, j, j-0.01, other_fields_linear)
                initial_change = res['acceleration']
                area1 = res['area1']
                area2 = res['area2']
                # initial_change is the change from area1 to area 2
                # area2>area1 -> initial_change > 1
                # area2<area1 -> initial_change < 1
                # area 2 is calculated with lower threshold - should usually be larger
                first_pixels = res['first_pixels']
                if not np.isnan(initial_change) and initial_change > 0:
                    # True is both area1 and area2 are valid
                    # Weird conditonal from Vadim - initial change will EITHER:
                    # be greater than zero (can't get a negative area to give negative % change)
                    # OR be NaN (which will always yield false when compared to a number)
                    used_th = j - 0.01
                    break

            if np.isnan(initial_change) and not np.isnan(area1):
                # For the final change
                pixels = np.unravel_index(first_pixels, fmap.shape, 'F')
                fmap[pixels] = max_value * 1.5
                fields_map[pixels] = field_id
                field_id = field_id + 1

            if np.isnan(initial_change):
                # failed to extract the field
                # Do nothing and continue for-loop
                pass

        pixel_list = _expand_field(fmap, occupancy_mask, peak_rc, initial_change, area2,
                                   other_fields_linear, used_th)
        if np.any(np.isnan(pixel_list)):
            _, pixel_list, _ = _area_for_threshold(fmap, occupancy_mask,
                                                   peak_rc, used_th+0.01,
                                                   other_fields_linear)
        if len(pixel_list) > 0:
            pixels = np.unravel_index(pixel_list, fmap.shape, 'F')
        else:
            pixels = []

        fmap[pixels] = max_value * 1.5
        fields_map[pixels] = field_id
        field_id = field_id + 1


    ##########################################################################
    #####     Part 4: Determine which, if any, fields meet filtering criteria
    regions = measure.regionprops(fields_map)

    fields = []
    fields_map = np.zeros(finite_firing_map.shape)  # void it as we can eliminate some fields

    for region in regions:
        field_map = finite_firing_map[region.coords[:, 0], region.coords[:, 1]]
        mean_rate = np.nanmean(field_map)
        num_bins = len(region.coords)

        peak_rate = np.nanmax(field_map)
        peak_relative_index = np.argmax(field_map)
        peak_coords = region.coords[peak_relative_index, :]

        if num_bins >= min_bins and mean_rate >= min_mean:
            field = {}
            field['coords'] = region.coords
            field['peak_coords'] = peak_coords
            field['area'] = region.area
            field['bbox'] = region.bbox
            field['centroid_coords'] = region.centroid
            field['mean_rate'] = mean_rate
            field['peak_rate'] = peak_rate
            mask = np.zeros(finite_firing_map.shape)
            mask[region.coords[:, 0], region.coords[:, 1]] = 1
            field['map'] = mask

            fields.append(field)

            fields_map[region.coords[:, 0], region.coords[:, 1]] = len(fields)
        elif debug:
            # Print out some information about *why* the field failed
            if num_bins < min_bins:
                print("Field size too small (%d)" % num_bins)
            if mean_rate < min_mean:
                print("Field mean rate too low (%.2f Hz)" % mean_rate)
        else:
            # Field too small and debugging information not needed
            # Do nothing
            pass
    #fields_map = np.ma.masked_where(occupancy_mask, fields_map)
    return (fields, fields_map)



#########################################################
################        Helper Functions
#########################################################


def _expand_field(image, occupancy_mask, peak_rc, initial_change,
                  initial_area, other_fields_linear, initial_th):
    '''
    Adaptive placefield detection:
        Start with a threshold around 80%, step down in ~0.02
        Measure field at threshold
        If the field is invalid, take the previous iteration
        If field size has decreased, keep count
        If field size has increased by less than 300% of initial_change AND the
          field has decreased in size fewer than 3 times in a row
            If the field size hasn't changed in 10 steps, return the current field size
        Else
    '''
    pixel_list = np.nan
    last_area = initial_area
    last_pixels = []
    num_not_changing = 0
    num_decrease = 0

    num_steps = int((initial_th - 0.2) / 0.02) + 1
    for threshold in np.linspace(initial_th, 0.2, num_steps):
        threshold = np.round(threshold, 2)

        area, pixels, is_bad = _area_for_threshold(image, occupancy_mask, peak_rc,
                                                   threshold, other_fields_linear)
        if np.isnan(area) or is_bad:
            pixel_list = last_pixels
            break
        current_change = area / last_area
        if current_change < 1:
            num_decrease = num_decrease + 1
        else:
            num_decrease = 0

        if np.floor(current_change / initial_change) <= 2 and num_decrease < 3:
            if current_change == 1:
                num_not_changing = num_not_changing + 1
            else:
                num_not_changing = 0

            if num_not_changing < 10:
                last_area = area
                last_pixels = pixels
                continue

        pixel_list = last_pixels
        break

    peak_linear = np.ravel_multi_index(multi_index=(peak_rc[0], peak_rc[1]),
                                       dims=image.shape, order='F')
    # last_pixels is a vector, peak_linear is a single value
    if np.any(last_pixels == peak_linear):
        # good field
        pixel_list = last_pixels

    return pixel_list


def _area_change(image, occupancy_mask, peak_rc, first, second, other_fields_linear):
    ''' Compare the change in field area based on two threshold values

    If either threshold results in an invalid field (based on criteria in
    _area_for_threshold() ), then return NaNs to signal this fact

    Otherwise return information about the results from the two thresholds, and
    the % change in area

    Params
    ------
    image : np.ndarray
        Image, i.e. rate map being considered
    occupancy_mask: np.ndarray
        binary mask. True where the animal spent zero time
    peak_rc : tuple
        [row, col] of local maxima
    first : float
        first threshold
    second : float
        second threshold
    other_fields_linear : np.ndarray
        All other local maxima except the ony under consideration
    '''
    results = {'acceleration': np.nan, 'area1': np.nan, 'area2': np.nan,
               'first_pixels': np.nan,
               'second_pixels': np.nan}

    area1, first_pixels, is_bad1 = _area_for_threshold(image, occupancy_mask, peak_rc, first,
                                                       other_fields_linear)
    if np.isnan(area1) or is_bad1:
        return results

    area2, second_pixels, is_bad2 = _area_for_threshold(image, occupancy_mask, peak_rc, second,
                                                        other_fields_linear)
    if np.isnan(area2) or is_bad2:
        return results

    acceleration = area2 / area1 #* 100
    results['acceleration'] = acceleration
    results['area1'] = area1
    results['area2'] = area2
    results['first_pixels'] = first_pixels
    results['second_pixels'] = second_pixels

    return results


def _area_for_threshold(image, occupancy_mask, peak_rc, threshold, other_fields_linear):
    '''Calculate the area of the field defined by the local maxima 'peak_rc' and
    the relative thresholding value 'threshold'

    In addition, determine:
        * is the field contiguous (good), or does it contain voids? (bad)
        * Does the field extend to include other local maxima?

    1 - Based on the threshold, find the binary map of the field
    2 - Determine if there are any voids
        If there are any voids, then return an area of np.nan
    3 - Determine the co-ordinates of all cells inside the field. The area is
        given by the number of cells
        If any included cell ALSO appears in the list of 'other_fields_linear'
        then another local maxima has been included. In that case, return is_bad=True

    returns
    -------
    area : int
        Number of cells in field OR np.nan if field contains holes
    area_linear_indicies : np.ndarray
        indicies of all cells within field if field is valid
    is_bad : bool
        True IF field includes a second local maxima or IF field contains holes
    '''
    area = np.nan
    # Field is bad if it contains any other peak
    is_bad = False

    peak_value = image[peak_rc[0], peak_rc[1]]
    threshold_value = peak_value * threshold
    mask = (image >= threshold_value)

    # Mask includes all pixels above the desired threshold, including other disconnected fields
    # use morphology.label to exclude disconnected fields
    # connectivity=1 means only consider vertical/horizontal connections
    labeled_img = morphology.label(mask, connectivity=1)

    # we need to leave only one label that corresponds to the peak
    target_label = labeled_img[peak_rc[0], peak_rc[1]]
    labeled_img[labeled_img != target_label] = 0
    labeled_img[labeled_img == target_label] = 1

    #labelled_img only includes cells that are:
    #   Above the current threshold
    #   Connected (V/H) to the currently considered local maxima

    # calclate euler_number by hand rather than by regionprops
    # This yields results that are more similar to Matlab's regionprops
    # NOTE - this uses scipy.ndimage.morphology, while most else uses skimage.morphology
    filled_image = ndimage.morphology.binary_fill_holes(labeled_img)
    euler_array = (filled_image != labeled_img)  # True where holes were filled in

    euler_array = np.maximum((euler_array*1) - (occupancy_mask*1), 0)
    # Ignore filled-in holes if it is due to the animal never visiting that location
    # Convert both arrays to integer, subtract one from the other, and replace resulting -1 values with 0
    # NOTE! np.maximum is element-wise, i.e. it returns an array. This is DIFFERENT to np.max, which returns a float.

    euler_objects = morphology.label(euler_array, connectivity=2) # connectivity=2 : vertical, horizontal, and diagonal
    num = np.max(euler_objects) # How many holes were filled in
    euler_number = -num + 1

    if euler_number <= 0:
        # If any holes existed, then return this
        is_bad = True
        return (area, [], is_bad)

    regions = measure.regionprops(labeled_img)
    area = np.sum(labeled_img == 1)
    area_linear_indices = np.ravel_multi_index(multi_index=(regions[0].coords[:, 0],
            regions[0].coords[:, 1]), dims=image.shape, order='F') # co-ordinates of members of field
    if len(other_fields_linear) > 0:
        is_bad = len(np.intersect1d(area_linear_indices, other_fields_linear)) > 0 # True if any other local maxima occur within this field

    return (area, area_linear_indices, is_bad)
