"""
Provide function for gridness score calculation.
"""

import matplotlib.pyplot as plt
import numpy as np

from scipy.spatial.distance import cdist
from scipy.stats import pearsonr
from skimage.transform import rotate

from .. import defaults as default
from .placefield import * 

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
        field_threshold: float
            Normalized threshold value used to search for peaks on the
            autocorrelogram. Ranges from 0 to 1, default value is 0.2.
        min_orientation: float
            Value of minimal difference of inner fields orientation (in
            degrees). If there are fields that differ in orientation for less than
            min_orientation, then only the closest to the centre field is left.
            Default value is 15.

    

    Returns:
    Always returns a gridness score value. It ranges from -2 to 2. 2 is more of
    a theoretical bound for a perfect grid. More practical value for a good
    grid is around 1.3. If function can not calculate a gridness score, NaN value
    is returned.
    """
    # Arrange keyword arguments
    fieldThreshold = kwargs.get("field_threshold", default.field_threshold)
    minOrientation = kwargs.get("min_orientation", default.min_orientation)
    debug = kwargs.get("debug", False)
    
    
    # normalize aCorr in order to find contours
    aCorr = aCorr / aCorr.max()
    cFieldRadius = np.floor(_findCentreRadius(aCorr, fieldThreshold, debug))
    if debug:
        print("Center radius is {}".format(cFieldRadius))

    if cFieldRadius in [-1, 0, 1]:
        return np.NaN

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
    rotAngles_deg = 30 * np.arange(1, 6)  # 30, 60, 90, 120, 150
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
    gscoreInd = np.argmax(GNS, axis=0)

    numGridnessRadii = 3
    numStep = numSteps - numGridnessRadii
    if numStep < 1:
        numStep = 1

    if numStep == 1:
        gscore = np.mean(GNS, axis=0)[0]
    else:
        meanGridness = np.zeros((numStep, ))
        for ii in range(numStep):
            meanGridness[ii] = np.nanmean(GNS[ii:ii+numGridnessRadii, 0])
        gscore = np.max(meanGridness)

    return gscore


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


def _findCentreRadius(aCorr, fieldThreshold, debug):
    centroids = []
    radii = []

    halfHeight = np.ceil(aCorr.shape[0]/2)
    halfWidth = np.ceil(aCorr.shape[1]/2)
    peak_coords = np.ones(shape=(1, 2), dtype=np.int)
    peak_coords[0, 0] = halfHeight-1
    peak_coords[0, 1] = halfWidth-1
    fields = placefield(aCorr, min_bins=2, min_peak=0, peak_coords=peak_coords)[0]
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
    if debug: print('Center field coord: {}'.format(peak_coords[closestFieldInd]))
    radius = np.floor(np.sqrt(areas[closestFieldInd] / np.pi))
    if debug: print("Radius is {}".format(radius))
    return radius
