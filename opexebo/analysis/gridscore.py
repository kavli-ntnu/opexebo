"""
Provide function for gridness score calculation.
"""

import os  # TODO TODO
os.environ['HOMESHARE'] = r'C:\temp\astropy'   #TODO TODO


import matplotlib.pyplot as plt
import numpy as np

from scipy.spatial.distance import cdist
from scipy.stats import pearsonr
from skimage.transform import rotate
from skimage import morphology, measure

import opexebo
import opexebo.defaults as default

def gridscore(aCorr, **kwargs):
    """Calculate gridness score for an autocorrelogram.

    Calculates a gridness score by expanding a circle around the centre field
    and calculating a correlation value of that circle with it's rotated versions.
    The expansion is done up until the smallest side of the autocorrelogram.
    The function may also calculate grid statistics.

    Gridness score value by itself is calculated as a maximum over a sliding mean
    of expanded circle. The width of the sliding window is given by a variable
    numGridnessRadii. This is done in order to keep gridness score the same as
    historical values (i.e. older versions of gridness score).

    Arguments:
    acorr: np.ndarray
        A 2D autocorrelogram.
    **kwargs
        'debug' : bool
            if true, output debugging information. Default False
        'search_method' : str
            Method passed to opexebo.analysis.placefield for detecting the central 
            peak of aCorr.
            Default and all possible values are stored in opexebo.defaults

    grid_stats = {'ellipse':gs_ellipse, 'ellipse_theta':gs_ellipse_theta,
                  'spacing':gs_spacing, 'orientation':gs_orientation}    

    Returns:
    -------
    grid_score : float
        Always returns a gridness score value. It ranges from -2 to 2. 2 is 
        more of a theoretical bound for a perfect grid. More practical value for 
        a good grid is around 1.3. If function can not calculate a gridness 
        score, NaN value is returned.
    grid_stats : dict
        'ellipse' : np.ndarray
            Definition of the ellipse fitting the 6 fields in the autocorrelogram
            closest to, but not at, the centre
            Output is determined by opexebo.general.fitellipse
            [centre_x, centre_y, radius_major, radius_minor, angle (rad)]
        'ellipse_theta' : float
            Angle of ellipse in degrees. 
        'spacing' : np.ndarray
            3x1 array giving distance (in bins) to closest fields. 
        'orientation' : np.ndarray
            3x1 array giving orientation (in degrees) to closest fields. 
        
    See Also
    """
    # Arrange keyword arguments
#    fieldThreshold = kwargs.get("field_threshold", default.field_threshold)
#    minOrientation = kwargs.get("min_orientation", default.min_orientation)
    debug = kwargs.get("debug", False)
    
    
    # normalize aCorr in order to find contours
    aCorr = aCorr / aCorr.max()
    cFieldRadius = int( np.floor(_findCentreRadius(aCorr, **kwargs)) )
    if debug:
        print("Center radius is {}".format(cFieldRadius))

    if cFieldRadius in [-1, 0, 1]:# or cFieldRadius[0] > .8*aCorr.shape[0]/2 or cFieldRadius[0] > .8*aCorr.shape[1]/2:
        return (np.nan, _grid_score_stats(np.zeros(aCorr.shape)))

    halfHeight = np.ceil(aCorr.shape[0]/2)
    halfWidth = np.ceil(aCorr.shape[1]/2)
    heightIndices = np.arange(aCorr.shape[0])
    widthIndices = np.arange(aCorr.shape[1])

    # Define radii that will be iterated over for the gridness score
    # outer bound is defined by the minimum of autocorrelogram's dimensions
    # this is need for rectangular autocorrelograms.
    outerBound = int(np.ceil(np.min(np.array(aCorr.shape)/2)))
    radii = np.linspace(cFieldRadius+1, outerBound, outerBound-cFieldRadius).astype(int)
    numSteps = len(radii)
    rotAngles_deg = np.arange(30, 151, 30)  # 30, 60, 90, 120, 150
    rotatedACorr = np.zeros(
            shape=(aCorr.shape[0], aCorr.shape[1], len(rotAngles_deg)),
            dtype=float)

    # we get rotated maps here as it is a heavy operation
    for n, angle in enumerate(rotAngles_deg):
        rotatedACorr[:, :, n] = rotate(aCorr, angle, preserve_range=True, clip=False)

    rr, cc = np.meshgrid(heightIndices, widthIndices, sparse=False)

    # This is needed for compatibility with Matlab's code
    rr = rr + 1
    cc = cc + 1

    mainCircle = np.sqrt(np.power((cc - halfWidth), 2) + np.power(rr-halfHeight, 2))
    innerCircle = mainCircle > cFieldRadius
    GNS = np.zeros(shape=(numSteps, 2), dtype=float)
    for i, radius in enumerate(radii):
        mask = innerCircle & (mainCircle < radius)
        aCorrValues = aCorr[mask]

        rotCorr = np.zeros(len(rotAngles_deg, ), dtype=float)
        for j, angle in enumerate(rotAngles_deg):
            rotatedValues = rotatedACorr[mask, j]
            r, p = pearsonr(aCorrValues, rotatedValues)
            rotCorr[j] = r
            if debug:
                print("Step {}, angle {}, corr value {}".format(i, angle, r))

        GNS[i, 0] = np.min(rotCorr[[1, 3]]) - np.max(rotCorr[[0, 2, 4]])
        GNS[i, 1] = radius

    # find the greatest gridness score value and radius
    gscoreInd = np.argmax(GNS[:,0])

    numGridnessRadii = 3
    numStep = max(numSteps - numGridnessRadii, 1) # minimum value 1

    if numStep == 1:
        gscore = np.mean(GNS, axis=0)[0]
    else:
        meanGridness = np.zeros(numStep)
        for ii in range(numStep):
            meanGridness[ii] = np.nanmean(GNS[ii:ii+numGridnessRadii, 0])
        gscore = np.max(meanGridness)

    '''Then calculate stats about the autocorrelogram'''
    bestCorr = (mainCircle < radii[gscoreInd]*1.25) * aCorr
    gstats = _grid_score_stats(bestCorr, **kwargs)    
    
    return (gscore, gstats)


def _grid_score_stats(bestCorr, **kwargs):
    # Get kwargs
    debug = kwargs.get('debug', False)
    min_orientation = kwargs.get('min_orientation', default.min_orientation)
    min_orientation = np.radians(min_orientation)
    
    # Initialise default output in case stats are uncalculable
    gs_ellipse_theta = np.nan
    gs_ellipse = np.nan
    gs_orientation = np.array([np.nan, np.nan, np.nan])
    gs_spacing = np.array([np.nan, np.nan, np.nan])
    
    
    # Identify local maxima    
    regionalMax = morphology.local_maxima(bestCorr, connectivity=4, indices=False)
    selem = morphology.square(3)
    dilated_img = morphology.dilation(regionalMax, selem)
    labelled_img = morphology.label(dilated_img)
    #labelled_img[labelled_img==int(np.max(labelled_img)/2)+1] = 0
    
    if np.max(labelled_img) >= 7:
        properties = measure.regionprops(labelled_img, cache=True)
        
        # Typical shape is 71x71 -> central bin is 35,35
        centre = -0.5 + np.array(bestCorr.shape)/2 # centre : also [y, x]
        all_coords = np.array([region.centroid for region in properties])
        # x-coords are all_coords[:,1], y are [:,0].
        
        if debug:
            plt.figure()
            plt.title("Labelled autocorrelogram and ellipse")
            plt.imshow(labelled_img)
            plt.scatter(all_coords[:,1], all_coords[:,0])
        
        
        # Bearing from field to centre
        orientation = np.arctan2(all_coords[:,0] - centre[0], all_coords[:,1] - centre[1]) # in radians
        distance = np.sqrt(np.square(all_coords[:,0]-centre[0]) + np.square(all_coords[:,1]-centre[1]))

        # The centre field really mucks up the later min_orientation filtering
        # Deleting it here is the best solution I found, but I'm not entirely happy
        # It's both inefficient and inelegant
        centre_index = np.where(distance==0)
        all_coords = np.delete(all_coords, centre_index, axis=0)
        orientation = np.delete(orientation, centre_index)
        distance = np.delete(distance, centre_index)
        

        
        # Where two fields have a very similar orientation, discard the more distant one
        orient_distsq = np.abs(_circ_dist2(orientation))
        close_fields = orient_distsq < min_orientation  
        close_fields = np.triu(close_fields, 1) # Upper triangle only - set lower triangle to zero
                                                # k=1: +1 offset from diagonal: set diagonal to zero too
        row, col = np.where(close_fields)
        to_del = np.zeros(row.size) # this is more efficient than appending to a list
        for i in range(row.size):
            r = row[i]
            c = col[i]
            if distance[r] > distance[c]:
                j = r
            else:
                j = c
            to_del[i] = int(j)
        to_del = np.unique(to_del).astype(int) # this gets rid of all the zeros
        distance = np.delete(distance, to_del)
        all_coords = np.delete(all_coords, to_del, axis=0)
        orientation = np.delete(orientation, to_del)
        # Actual deletion instead of substituting values - this handles cases 
        # where there are fewer than 6 fields remaining after filtering
        
        
        # Consider the 6 closest fields -> sort by distance
        # The central point has already been deleted.
        # Positions needs all 6 indicies, in order to calculate the ellipse
        # Discard the duplicate values -> because they're floating point, don't use np.unique()
        # Instead, use array slicing: sort by size, and then take every 2nd value
        # Numpy slicing: [start:stop:step]
        sorted_ids = np.argsort(distance)[:6]
        positions = all_coords[sorted_ids,:]
        gs_spacing = distance[sorted_ids[::2]]
        gs_orientation = orientation[sorted_ids[::2]]
        gs_orientation = np.degrees(gs_orientation)%180 # convert from rad to deg and wrap into [0-180]°
        
        # Sometimes there are not sufficient fields to yield 3 spacing, orientation
        # In these cases, pad the arrays out tot he expected length with NaN
        if gs_spacing.size < 3:
            tmp = np.full(3, fill_value = np.nan, dtype=float)       
            tmp[:gs_spacing.size] = gs_spacing
            gs_spacing = tmp
        if gs_orientation.size < 3:
            tmp = np.full(3, fill_value = np.nan, dtype=float) 
            tmp[:gs_orientation.size] = gs_orientation            
            gs_orientation = tmp
        
        # Fit an ellipse to those remaining fields:
        gs_ellipse =  opexebo.general.fitellipse(positions[:,1], positions[:,0])
        gs_ellipse_theta = np.degrees(gs_ellipse[4]+np.pi)%360
        # The +pi term was included in the original BNT, I have kept it to 
        # maintain consistency with past results. 
        
        
        if debug:
            a, b, c, d, e = gs_ellipse
            _draw_ellipse(a, b, c, d, e)
            plt.scatter(positions[:,1], positions[:,0])
    
    # Not sure whether a list or dictionary is preferred here.
    grid_stats = {'ellipse':gs_ellipse, 'ellipse_theta':gs_ellipse_theta,
                  'spacing':gs_spacing, 'orientation':gs_orientation}
    #grid_stats = [gs_ellipse, gs_ellipse_theta, gs_spacing, gs_orientation]
    return grid_stats
        
    
    
def _draw_ellipse(x, y, rl, rs, theta):
    from matplotlib.patches import Ellipse
    from matplotlib import transforms
    theta = (theta%(2*np.pi))# + np.pi
    ell = Ellipse((0,0), width=rs*2, height=rl*2, facecolor=(1,0,0,0.2),
                  edgecolor=(1,1,1,0.75))
    ax = plt.gca()
    transf = transforms.Affine2D().rotate(theta+np.pi/2).translate(x, y)
    ell.set_transform(transf + ax.transData)
    ax.add_patch(ell)
    

def _plotContours(img, contours):
    fig, ax = plt.subplots()
    ax.imshow(img, cmap='jet', origin='lower')

    centroids = np.zeros(shape=(len(contours), 2), dtype=float)
    radii = np.zeros(shape=(len(contours), 1), dtype=float)
    for n, contour in enumerate(contours):
        x = contour[:, 1]
        y = contour[:, 0]
        ax.plot(x, y, linewidth=2)
        # mean x and y
        centroid = (sum(x) / len(contour), sum(y) / len(contour))
        radius = [np.mean([np.sqrt(np.square(x-centroid[0]) + np.square(y-centroid[1]))])]

        centroids[n] = centroid
        radii[n] = radius

        ax.text(centroid[0], centroid[1], str(n))
        ax.plot((centroid[0], centroid[0]+radius), (centroid[1], centroid[1]), linewidth=1)

    ax.axis('image')
    ax.set_xticks([])
    ax.set_yticks([])
    plt.show()
    return radii, centroids



def _circ_dist2(X):
    '''Given a 1D array of angles, find the 2D array of pairwise differences
    Based on https://github.com/circstat/circstat-matlab/blob/master/circ_dist2.m'''
    x = np.outer(np.exp(1j*X), np.ones(X.size)) # similar to meshgrid, but simpler to implement
    y = np.transpose(x)
    return np.angle(x/y)


# Polygon area, from https://stackoverflow.com/questions/24467972/calculate-area-of-polygon-given-x-y-coordinates
# Seems to be the same as Matlab's polyarea
def _polyArea(x, y):
    return 0.5*np.abs(np.dot(x, np.roll(y, 1)) - np.dot(y, np.roll(x, 1)))


def _contourArea(contours, i):
    contour = contours[i]
    x = contour[:, 1]
    y = contour[:, 0]
    area = _polyArea(x, y)
    return area


def _findCentreRadius(aCorr, **kwargs):
    centroids = []
    radii = []
    

    halfHeight = np.ceil(aCorr.shape[0]/2)
    halfWidth = np.ceil(aCorr.shape[1]/2)
    peak_coords = np.ones(shape=(1, 2), dtype=np.int)
    peak_coords[0, 0] = halfHeight-1
    peak_coords[0, 1] = halfWidth-1
    fields = opexebo.analysis.placefield(aCorr, min_bins=2, min_peak=0, peak_coords=peak_coords, **kwargs)[0]
    if fields is None or len(fields) == 0:
        return 0

    peak_coords = np.ndarray(shape=(len(fields), 2), dtype=np.integer)
    areas = np.ndarray(shape=(len(fields), 1), dtype=np.integer)
    for i, field in enumerate(fields):
        peak_rc = field['peak_coords']
        peak_coords[i, 0] = peak_rc[0]
        peak_coords[i, 1] = peak_rc[1]
        areas[i] = field['area']

    aCorrCentre = np.zeros(shape=(1, 2), dtype=float)
    aCorrCentre[0] = aCorr.shape[0]
    aCorrCentre[0, 1] = aCorr.shape[1]
    aCorrCentre = np.ceil(aCorrCentre/2)

    # index of the closest field to the centre
    closestFieldInd = 0
    if len(peak_coords) >= 2:
        # get all distances and check two minimum of them
        distancesToCentre = cdist(peak_coords, aCorrCentre)
        sortInd = np.squeeze(np.argsort(distancesToCentre, axis=0))

        closestFieldInd = sortInd[0]
        twoMinDistances = distancesToCentre[sortInd[:2]]

        areFieldsClose = np.abs(twoMinDistances[0] - twoMinDistances[1])[0] < 2
        if areFieldsClose:
            # two fields with close middle point. Let's select one with minimum area
            indices_to_test = sortInd[:2]
            min_ind = np.argmin(areas[indices_to_test])
            closestFieldInd = indices_to_test[min_ind]

    radius = np.floor(np.sqrt(areas[closestFieldInd] / np.pi))
    # print("radius is {}".format(radius))
    return radius



if __name__ == '__main__':
    plt.close("all")
    print("Loading modules")
    import os
    os.environ['HOMESHARE'] = r'C:\temp\astropy'
    import scipy.io as spio
    import matplotlib.pyplot as plt
    
    bnt_output = r'C:\Users\simoba\Documents\_work\Kavli\bntComp\Output\auto_input_file_vars.mat'
    print("Loading data")
    #bnt = spio.loadmat(bnt_output)
    print("Data loaded")
    
    i = 199
    acorr = bnt['cellsData'][i,0]['epochs'][0,0][0,0]['aCorr'][0,0]
    gscore = bnt['cellsData'][i,0]['epochs'][0,0][0,0]['gridScore'][0,0][0,0]
    
    gsop= gridscore(acorr, debug=True, search_method='default')
    print(gsop)