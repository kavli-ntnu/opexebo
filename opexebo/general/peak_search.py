import numpy as np
import opexebo.defaults as default
from skimage import measure, morphology
from scipy.ndimage import distance_transform_cdt


#import sep
# NOTE!! Due to how annoying it is to install sep, I have moved this dependency 
# into the method that uses it. This will allow users that struggle to install it
# to proceed with other methods that do not rely on it

# TODO! Decide what, if any, parameters need to move over to Defaults


def peak_search(image, **kwargs):
    """Given a 1D or 2D array, return a list of co-ordinates of the local 
    maxima or minima
    
    Multiple searching techniques are provided:
        
        * `default`: uses `skimage.morphology.get_maxima`
        * `sep`: uses the Python wrapper to the Source Extractor astronomy tool
          to identify peaks
    
    Parameters
    ----------
    image: np.ndarray
        1D or 2D array of data
    search_method : str, optional, {"default", "sep"}
    mask : np.ndarray, optional
        Array of masked locations in the image with the same dimensions.
        Locations where the mask value is True are ignored for the purpose of
        searching.
    maxima: bool, optional
        [`default` search method only] Define whether to search for maxima or
        minima in the provided array
    null_background: bool
        [`sep` search method only] Set the image background to zero for
        calculation purposes rather than attempt to calculate a background
        gradient. This should generally be True, as our images are not directly
        comparable to standard telescope output
    threshold : float, optional
        [`sep` search method only] Threshold for identifiying maxima area
    
    Returns
    -------
    peak_coords: tuple
        Co-ordinates of peaks, in the form ((x0, x1, x2...), (y0, y1, y2...))
    
    
    Notes
    --------
    Copyright (C) 2019 by Simon Ball
    """
    
    search_method = kwargs.get("search_method", default.search_method)
    get_maxima = kwargs.get("maxima", True)
    
    if search_method not in default.all_methods:
        raise ValueError("Keyword 'search_method' must be left blank or given a"\
                    " value from the following list: %s. You provided '%s'."\
                    % (default.all_methods, search_method) )
    if search_method != "default" and not get_maxima:
        raise NotImplementedError("Local minima detection is currently only"\
                            " implemented for the 'default' search method")
        
    if search_method == default.search_method:
        peak_coords = _peak_search_skimage(image, **kwargs)
    elif search_method == "sep":
        peak_coords = _peak_search_sep_wrapper(image, **kwargs)
    else:
        raise NotImplementedError("The search method you have requested (%s) is"\
                                  " not yet implemented" % search_method)
        
    return peak_coords
        
        
        
        
#########################################################
################        Helper Functions
#########################################################


def _peak_search_skimage(image, **kwargs):
    '''Default peak detection method:
        skimage.morphology.get_m**ima (either minima or maxima)
    Since skimage doesn't handle masked arrays, the masking is a bit of a bodge
    job here. The basic plan is as follows:
        * Set the area of the image covered by the mask to a value that cannot include a maxima (or minima, as appropriate)
        * Search for peak coordinates
        * Check if, after rounding, any of the peaks are outside the image dimensions
        * Check if, after rounding, any of the peaks are extremely close to the mask
    
    Since the mask is a ahrd-edged area, if there is even a slight rise just
    outside it, spurious peaks can be detected. Therefore, we automatically reject
    any peaks for a short distance outside the actual mask
    
    '''
    connectivity = 2
    get_maxima = kwargs.get("maxima", True)
    mask = kwargs.get("mask", np.zeros(image.shape, dtype=bool))
    image_copy = image.copy()
    if get_maxima:
        image_copy[mask] = np.nanmin(image_copy)
        regionalMaxMap = morphology.local_maxima(image_copy, connectivity=connectivity, allow_borders=True)
    else:
        image_copy[mask] = np.nanmax(image_copy)
        regionalMaxMap = morphology.local_minima(image_copy, connectivity=connectivity, allow_borders=True)
    labelled_max = measure.label(regionalMaxMap, connectivity=connectivity)
    regions = measure.regionprops(labelled_max)
    peak_coords = np.zeros(shape=(len(regions), 2), dtype=int)
    
    distance_from_mask = distance_transform_cdt(image_copy * (1-mask))

    for i, props in enumerate(regions):
        y0, x0 = props.centroid
        peak = np.array([y0, x0])

        # ensure that there are no peaks off the map (due to rounding)
        peak[peak < 0] = 0
        for j in range(image_copy.ndim):
            if peak[j] > image_copy.shape[j]:
                peak[j] = image_copy.shape[j] - 1
        
        peak_index = tuple(np.round(peak, 0).astype(int)) # indexing with a floating point array sucks, so convert to a more convenient form
        if distance_from_mask[peak_index] > 2*connectivity:
            peak_coords[i, :] = peak
    return peak_coords


def _peak_search_sep_wrapper(firing_map, **kwargs):
    ''' Wrapper around the 'sep' Peak Search method
    Because the 'sep' package is a nightmare to install, and not used for most
    analysis routines, this wrapper allows most users to ignore it
    
    If the user tries to invoke the 'sep' routines, this will try to do so
    If it fails due to ModuleNotFound, it will use the default algorithm instead
    with a warning to the user
    '''
    try:
        import sep
        return _peak_search_sep(firing_map, **kwargs)
    except ModuleNotFoundError:
        raise ModuleNotFoundError("The package 'sep' is missing from your system."\
                " You can invoke an alternative algorithm that does not depend on"\
                " 'sep' by assigning a different value to 'search_method'."\
                " Alternatively, install 'sep' on your system:"\
                " 'pip install sep'")
    
        
        

def _peak_search_sep(firing_map, **kwargs):
    '''Peak search using sep, a Python wrapper for a standard astronomy library.
    sep is typically used to identify astonomical objects in telescope images
    Sep requires copies of the arrays that are C-ordered (Python default is 
    Fortran-ordered, the difference is row vs column-major).
    
    TODO: The behaviour of sep with masks needs to be investigated - where the 
    edge of the image is masked, objects are still sometimes found. Example: 
        {'path': 'N:\\davidcr\\84932\\19032019',
         'basename': '19032019s1',
         'tetrode': 6,
         'cell': 23}
    '''
    import sep
    
    mask = kwargs.get("mask", np.zeros(firing_map.shape, dtype=bool))
    null_background = kwargs.get("null_background", True)
    threshold = kwargs.get("threshold", 0.2)
    
    tmp_firing_map = firing_map.copy('C')
    tmp_mask = mask.copy(order='C')
    
    if null_background:
        bkg = sep.Background(np.zeros_like(tmp_firing_map))
    else:
        bkg = sep.Background(tmp_firing_map, mask=tmp_mask, fw=2, fh=2, \
                     bw=int(tmp_firing_map.shape[0]), bh=int(tmp_firing_map.shape[1]))

    init_fields = sep.extract(tmp_firing_map-bkg, mask=tmp_mask, thresh=threshold, \
                          err=bkg.globalrms)

    peak_coords = np.zeros(shape=(len(init_fields), 2), dtype=np.int)

    for i, props in enumerate(init_fields):
        peak = np.array([props['y'], props['x']])
        peak = np.round(peak)
        # ensure that there are no peaks off the map (due to rounding)
        peak[peak < 0] = 0
        for j in range(firing_map.ndim):
            if peak[j] > firing_map.shape[j]:
                peak[j] = firing_map.shape[j] - 1

        peak_coords[i, :] = peak
    return peak_coords