"""
Provide function for 2D placefield detection.
"""

import numpy as np

from scipy import ndimage
from skimage import measure, morphology
import opexebo.defaults as default


def placefield(firing_map, **kwargs):
    """Locate place fields on a firing map.

    Identifies place fields in 2D firing map. Placefields are identified by 
    using an adaptive threshold. The idea is that we start with a peak value as 
    the threshold. Then we gradually decrease the threshold until the field 
    area doesn't change any more or the area explodes (this means the threshold 
    is too low).
    
    Parameters
    ----------
    firing_map : np.ndarray or np.ma.MaskedArray
        smoothed rate map
    **kwargs
        min_bins : int
            Fields containing fewer than this many bins will be discarded.
            Default 9
        min_peak : float
            Fields with a peak firing rate lower than this absolute value will 
            be discarded. Default 1 Hz
        min_mean : float
            Fields with a mean firing rate lower than this absolute value will 
            be discarded. Default 0 Hz
        peak_coords : array-like
            List of peak co-ordinates to consider instead of auto detection
            Default None
            
    Returns
    -------
    fields      : list (of dict)
        'coords'        : np.ndarray    : Co-ordinates of all bins in the firing field
        'peak_coords'   : np.ndarray    : Co-ordinates of the cell with the peak firing rate
        'area'          : int           : Number of bins in firing field. [bins]
        'bbox'          : tuple         : Co-ordinates of bounding box including the firing field (y_min, x_min, y_max, y_max)
        'x'             : float         : x co-ordinate of centroid. (decimal) [bins]
        'y'             : float         : y co-ordinate of centroid. (decimal) [bins]
        'mean_rate'     : float         : mean firing rate [Hz]
        'peak_rate'     : float         : peak firing rate [Hz]
        'map'           : np.ndarray    : Binary map of arena. Cells inside firing field have value 1, all other cells have value 0
    fields_map  : np.ndarray
        labelled integer image (i.e. background = 0, field 1 = 1, field2 = 2, etc)
        
    See also
    --------
    BNT.+analyses.placefieldAdaptive
    https://se.mathworks.com/help/images/understanding-morphological-reconstruction.html
    
    Copyright (C) 2018 by Vadim Frolov, (C) 2019 by Simon Ball
    
    This program is free software; you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation; either version 3 of the License, or
    (at your option) any later version.
    
    """
    # Get keyword arguments
    min_bins = kwargs.get("min_bins", default.firing_field_min_bins)
    min_peak = kwargs.get("min_peak", default.firing_field_min_peak)
    min_mean = kwargs.get("min_mean", default.firing_field_min_mean)
    peak_coords = kwargs.get("peak_coords", None)
    debug = kwargs.get("debug", False)
    
    global_peak = np.nanmax(firing_map)
    if np.isnan(global_peak) or global_peak == 0:
        return None

    nan_indices = np.isnan(firing_map)
    firing_map = np.nan_to_num(firing_map)
    
        
    # disc structural element of size 1
    se = morphology.disk(1)

    Ie = morphology.erosion(firing_map, se)
    Iobr = morphology.reconstruction(Ie, firing_map)
    # The effect of this erosion/reconstruction is to reduce value of cells 
    # around local maxima of the firing map. This can be shown, for instance, by
    # plt.imshow(np.ma.masked_where(firing_map-Iobr!=0, firing_map))
    # Also well explained here 
    # https://se.mathworks.com/help/images/understanding-morphological-reconstruction.html
    # This acts to discard some minor local maxima, especially in poorly connected
    # parts of the ratemap


    # this is regionmax equivalent
    regionalMaxMap = morphology.local_maxima(Iobr)

    # regionprops works on labeled iamge, so we have to convert binary to labeled
    labeled_max = measure.label(regionalMaxMap)
    regions = measure.regionprops(labeled_max)

    if peak_coords is None:
        peak_coords = np.zeros(shape=(len(regions), 2), dtype=np.int)

        for i, props in enumerate(regions):
            y0, x0 = props.centroid
            peak = np.array([y0, x0])
            peak = np.round(peak)
            # ensure that there are no peaks off the map (due to rounding)
            peak[peak < 0] = 0
            for j in range(firing_map.ndim):
                if peak[j] > firing_map.shape[j]:
                    peak[j] = firing_map.shape[j] - 1

            peak_coords[i, :] = peak

    # obtain value of found peaks
    found_peaks = firing_map[peak_coords[:, 0], peak_coords[:, 1]]

    # leave only peaks that satisfy the threshold
    good_peaks = (found_peaks >= min_peak)
    peak_coords = peak_coords[good_peaks, :]

    I = Iobr
    max_value = np.max(I)
    # prevent peaks with small values from being detected
    # SWB - This causes problems where a local peak is next to a cell that the animal never went
    # As that risks the field becoming the entire null region
    # Therefore, adding 2nd criterion to avoid adding information where none was actually known. 
    I[np.logical_and(I < min_peak, I > 0.01)] = max_value * 1.5

    # this can be confusing, but this variable is just an index for the vector
    # peak_linear_ind
    peaks_index = np.arange(len(peak_coords))

    fields_map = np.zeros(I.shape, dtype=np.integer)
    field_id = 1

    for i, peak_rc in enumerate(peak_coords):
        # peak_rc == [row, col]

        # select all peaks except the current one
        other_fields = peak_coords[peaks_index != i]
        if other_fields.size > 0:
            other_fields_linear = np.ravel_multi_index(
                    multi_index=(other_fields[:, 0], other_fields[:, 1]),
                    dims=I.shape, order='F')
        else:
            other_fields_linear = []

        used_th = 0.96
        res = _area_change(I, peak_rc, 0.96, 0.94, other_fields_linear)
        initial_change = res['acceleration']
        area2 = res['area2']
        first_pixels = np.nan
        if np.isnan(initial_change):
            for j in np.linspace(0.97, 1., 4):
                # Thresholds get higher, area should tend downwards to 1 (i.e. only including the actual peak)
                res = _area_change(I, peak_rc, j, j-0.01, other_fields_linear)
                initial_change = res['acceleration']
                area1 = res['area1']
                area2 = res['area2']
                first_pixels = res['first_pixels']
                if not np.isnan(initial_change) and initial_change > 0:
                    used_th = j - 0.01
                    break

            if np.isnan(initial_change) and not np.isnan(area1):
                # For the final change
                pixels = np.unravel_index(first_pixels, I.shape, 'F')
                I[pixels] = max_value * 1.5
                fields_map[pixels] = field_id
                field_id = field_id + 1
                continue

            if np.isnan(initial_change):
                # failed to extract the field
                continue

        pixel_list = _expand_field(I, peak_rc, initial_change, area2,
                                   other_fields_linear, used_th)
        if np.any(np.isnan(pixel_list)):
            # nu - not used
            nu, pixel_list, nu = _area_for_threshold(I, peak_rc, used_th+0.01, other_fields_linear)

        pixels = np.unravel_index(pixel_list, I.shape, 'F')
        I[pixels] = max_value * 1.5
        fields_map[pixels] = field_id
        field_id = field_id + 1

    regions = measure.regionprops(fields_map)

    fields = []
    fields_map = np.zeros(firing_map.shape)  # void it as we can eliminate some fields

    for region in regions:
        field_map = firing_map[region.coords[:, 0], region.coords[:, 1]]
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
    
            field['x'] = region.centroid[0]
            field['y'] = region.centroid[1]
    
            field['mean_rate'] = mean_rate
            field['peak_rate'] = peak_rate
            mask = np.zeros(firing_map.shape)
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
        

    return (fields, fields_map)


def _expand_field(I, peak_rc, initial_change, initial_area, other_fields_linear, initial_th):
    pixel_list = np.nan
    last_area = initial_area
    last_pixels = []
    num_not_changing = 0
    num_decrease = 0

    num_steps = np.round((initial_th - 0.2) / 0.02) + 1
    for i in np.linspace(initial_th, 0.2, num_steps):
        i = round(i, 2)

        area, pixels, is_bad = _area_for_threshold(I, peak_rc, i, other_fields_linear)
        if np.isnan(area) or is_bad:
            pixel_list = last_pixels
            break
        current_change = area / last_area * 100
        if current_change < 100:
            num_decrease = num_decrease + 1
        else:
            num_decrease = 0

        if np.floor(current_change / initial_change) <= 2 and num_decrease  < 3:
            if current_change == 100:
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
                                       dims=I.shape, order='F')
    # last_pixels is a vector, peak_linear is a single value
    if np.any(last_pixels == peak_linear):
        # good field
        pixel_list = last_pixels

    return pixel_list


def _area_change(I, peak_rc, first, second, other_fields_linear):
    ''' Compare the change in field area based on two threshold values
    
    If either threshold results in an invalid field (based on criteria in 
    _area_for_threshold() ), then return NaNs to signal this fact
    
    Otherwise return information about the results from the two thresholds, and
    the % change in area
    
    Params
    ------
    I : np.ndarray                      : Image
    peak_rc : tuple                     : [row, col] of local maxima
    first : float                       : first threshold
    second : float                      : second threshold
    other_fields_linear : np.ndarray    : All other local maxima *except* the one under consideration
    '''
    results = {'acceleration': np.nan, 'area1': np.nan, 'area2': np.nan,
               'first_pixels': np.nan,
               'second_pixels': np.nan}

    area1, first_pixels, is_bad1 = _area_for_threshold(I, peak_rc, first,
                                                       other_fields_linear)
    if np.isnan(area1) or is_bad1:
        return results

    area2, second_pixels, is_bad2 = _area_for_threshold(I, peak_rc, second,
                                                        other_fields_linear)
    if np.isnan(area2) or is_bad2:
        return results

    acceleration = area2 / area1 * 100
    results['acceleration'] = acceleration
    results['area1'] = area1
    results['area2'] = area2
    results['first_pixels'] = first_pixels
    results['second_pixels'] = second_pixels

    return results


def _area_for_threshold(I, peak_rc, th, other_fields_linear):
    '''Calculate the area of the field defined by the local maxima 'peak_rc' and
    the relative thresholding value 'th'
    
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
    ar : int
        Number of cells in field OR np.nan if field contains holes
    area_linear_indicies : np.ndarray
        indicies of all cells within field if field is valid
    is_bad : bool
        True IF field includes a second local maxima
    '''
    ar = np.nan
    # Field is bad if it contains any other peak
    is_bad = False

    peak_value = I[peak_rc[0], peak_rc[1]]
    th_value = peak_value * th
    mask = (I >= th_value)
    
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
    euler_array = (filled_image != labeled_img)  # The cells that were filled in
    euler_objects = morphology.label(euler_array, connectivity=2) # connectivity=2 : vertical, horizontal, and diagonal
    num = np.max(euler_objects) # How many holes were filled in
    euler_number = -num + 1

    if euler_number <= 0:
        # If any holes existed, then return this
        return (ar, [], is_bad)

    regions = measure.regionprops(labeled_img)
    ar = np.sum(labeled_img == 1)
    area_linear_indices = np.ravel_multi_index(multi_index=(regions[0].coords[:, 0],
            regions[0].coords[:, 1]), dims=I.shape, order='F') # co-ordinates of members of field
    if len(other_fields_linear) > 0:
        is_bad = len(np.intersect1d(area_linear_indices, other_fields_linear)) > 0 # True if any other local maxima occur within this field

    return (ar, area_linear_indices, is_bad)
