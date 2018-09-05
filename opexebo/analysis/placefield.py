"""
Provide function for 2D placefield detection.
"""

import numpy as np

from scipy import ndimage
from scipy.ndimage import filters
from skimage import measure, morphology


def placefield(map, min_bins=9, min_peak=1, min_mean_rate=0, peak_coords=None):
    """Locate place fields on a firing map.

    Identifies place fields in 2D firing map.
    """

    global_peak = np.nanmax(map)
    if np.isnan(global_peak) or global_peak == 0:
        return None

    nan_indices = np.isnan(map)
    map = np.nan_to_num(map)

    # disc structural element of size 1
    se = morphology.disk(1)

    Ie = morphology.erosion(map, se)
    Iobr = morphology.reconstruction(Ie, map)

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
            for j in range(map.ndim):
                if peak[j] > map.shape[j]:
                    peak[j] = map.shape[j] - 1

            peak_coords[i, :] = peak

    # obtain value of found peaks
    found_peaks = map[peak_coords[:, 0], peak_coords[:, 1]]

    # leave only peaks that satisfy the threshold
    good_peaks = found_peaks >= min_peak
    peak_coords = peak_coords[good_peaks, :]

    I = Iobr
    max_value = np.max(I)
    # prevent peaks with small values from being detected
    I[I < min_peak] = max_value * 1.5

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
                res = _area_change(I, peak_rc, j, j-0.01, other_fields_linear)
                initial_change = res['acceleration']
                area1 = res['area1']
                area2 = res['area2']
                first_pixels = res['first_pixels']
                if not np.isnan(initial_change) > 0:
                    used_th = j - 0.01
                    break

            if np.isnan(initial_change) and not np.isnan(area1):
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
    fields_map = np.zeros(map.shape)  # void it as we can eliminate some fields

    for region in regions:
        field_map = map[region.coords[:, 0], region.coords[:, 1]]
        mean_rate = np.nanmean(field_map)
        num_bins = len(region.coords)

        peak_value = np.nanmax(field_map)
        peak_relative_index = np.argmax(field_map)
        peak_coords = region.coords[peak_relative_index, :]

        if num_bins <= min_bins:
            continue

        field = {}
        field['coords'] = region.coords
        field['peak_value'] = peak_value
        field['peak_coords'] = peak_coords
        field['area'] = region.area
        field['bbox'] = region.bbox

        field['x'] = region.centroid[0]
        field['y'] = region.centroid[1]

        field['mean_rate'] = mean_rate
        mask = np.zeros(map.shape)
        mask[region.coords[:, 0], region.coords[:, 1]] = 1
        field['map'] = mask

        fields.append(field)

        fields_map[region.coords[:, 0], region.coords[:, 1]] = len(fields)

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
    ar = np.nan
    # Field is bad if it contains any other peak
    is_bad = False

    peak_value = I[peak_rc[0], peak_rc[1]]
    th_value = peak_value * th
    mask = I >= th_value

    labeled_img = morphology.label(mask, connectivity=1)

    # we need to leave only one label that corresponds to the peak
    target_label = labeled_img[peak_rc[0], peak_rc[1]]
    labeled_img[labeled_img != target_label] = 0
    labeled_img[labeled_img == target_label] = 1

    # calclate euler_number by hand rather than by regionprops
    # This yields results that are more similar to Matlab's regionprops
    filled_image = ndimage.morphology.binary_fill_holes(labeled_img)
    euler_array = filled_image != labeled_img
    euler_objects = morphology.label(euler_array, connectivity=2)
    num = np.max(euler_objects)
    euler_number = -num + 1

    if euler_number <= 0:
        return (ar, [], is_bad)

    regions = measure.regionprops(labeled_img)
    ar = np.sum(labeled_img == 1)
    area_linear_indices = np.ravel_multi_index(multi_index=(regions[0].coords[:, 0],
            regions[0].coords[:, 1]), dims=I.shape, order='F')
    if len(other_fields_linear) > 0:
        is_bad = len(np.intersect1d(area_linear_indices, other_fields_linear)) > 0

    return (ar, area_linear_indices, is_bad)
