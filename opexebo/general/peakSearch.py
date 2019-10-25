import numpy as np
import warnings
import opexebo.defaults as default
from skimage import measure, morphology


#import sep
# NOTE!! Due to how annoying it is to install sep, I have moved this dependency 
# into the method that uses it. This will allow users that struggle to install it
# to proceed with other methods that do not rely on it

# TODO! Decide what, if any, parameters need to move over to Defaults


def peak_search(image, **kwargs):
    """Given a 1D or 2D array, return a list of co-ordinates of the local 
    maxima or minima
    
    Multiple searching techniques are provided
    
    Parameters
    ----------
    image : np.ndarray
        1D or 2D array of data
    kwargs
        search_method : str
        maxima : bool
            If True, return the local maxima. Else, return the local minima.
            Default True
        mask : np.ndarray
            Same dimensions as image, True where values of image are to be ignored
            Only relevant to search method 'sep'
        null_background : bool
            Set the image background to zero for calculation purposes
            Only relevant to 'sep' : astronomical images typically have both a 
            background gradient and randomised noise in the image. SEP can
            generate a compensation for this - but the images we typically work
            with do not suffer the same problem. This should generally be True,
            unless you know exactly why it shouldn't be. 
    
    Returns
    -------
    peak_coords : tuple
        Co-ordinates of peaks, in the form ((x0, x1, x2...), (y0, y1, y2...))
    
    
    See also
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
    '''
    get_maxima = kwargs.get("maxima", True)
    mask = kwargs.get("mask", np.zeros(image.shape, dtype=bool))
    image_copy = image.copy()
    if get_maxima:
        image_copy[mask] = 0
        regionalMaxMap = morphology.local_maxima(image_copy, connectivity=2, allow_borders=True)
    else:
        image_copy[mask] = np.max(image_copy)
        regionalMaxMap = morphology.local_minima(image_copy, connectivity=2, allow_borders=True)
    labelled_max = measure.label(regionalMaxMap, connectivity=2)
    regions = measure.regionprops(labelled_max)
    peak_coords = np.zeros(shape=(len(regions), 2), dtype=np.int)

    for i, props in enumerate(regions):
        y0, x0 = props.centroid
        peak = np.array([y0, x0])

        # ensure that there are no peaks off the map (due to rounding)
        peak[peak < 0] = 0
        for j in range(image_copy.ndim):
            if peak[j] > image_copy.shape[j]:
                peak[j] = image_copy.shape[j] - 1

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
    null_background = kwargs.get("null_background", False)
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