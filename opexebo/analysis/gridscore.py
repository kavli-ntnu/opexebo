"""
Created on Wed Aug 29 10:49:42 2018

@author: Vadim Frolov
"""

from skimage import measure
from skimage.transform import rotate
import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial.distance import cdist
from scipy.stats import pearsonr

def gridness_score(aCorr, fieldThreshold=0.2):

    # normalize aCorr in order to find contours
    aCorr = aCorr / aCorr.max()
    cFieldRadius = np.floor(_findCentreRadius(aCorr, fieldThreshold))
    print("Center radius is {}".format(cFieldRadius))

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
    rotAngles_deg = 30 * np.arange(1,6); # 30, 60, 90, 120, 150
    rotatedACorr = np.zeros(shape=(aCorr.shape[0], aCorr.shape[1], len(rotAngles_deg)), dtype=float)

    # we get roated maps here as it is a heavy operation
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
        mask = innerCircle & (mainCircle < radius);
        aCorrValues = aCorr[mask]

        rotCorr = np.zeros(len(rotAngles_deg, ), dtype=float)
        for j, angle in enumerate(rotAngles_deg):
            rotatedValues = rotatedACorr[mask, j]
            r, p = pearsonr(aCorrValues, rotatedValues)
            rotCorr[j] = r
            print("Step {}, angle {}, corr value {}".format(i, angle, r))

        GNS[i, 0] = np.min(rotCorr[[1, 3]]) - np.max(rotCorr[[0, 2, 4]]);
        GNS[i, 1] = radius

    # find the greatest gridness score value and radius
    gscoreInd = np.argmax(GNS, axis=0)

    numGridnessRadii = 3;
    numStep = numSteps - numGridnessRadii
    if numStep < 1:
        numStep = 1

    if numSteps == 1:
        gscore = np.mean(GNS, axis=0)
    else:
        meanGridness = np.zeros((numStep, ))
        for ii in range(numStep):
            meanGridness[ii] = np.nanmean(GNS[ii:ii+numGridnessRadii, 0])
        gscore = np.max(meanGridness)

    return gscore


def plotContours(img, contours):
    fig, ax = plt.subplots()
    ax.imshow(img, cmap='jet', origin='lower')

    centroids = np.zeros(shape=(len(contours), 2), dtype=float)
    radii = np.zeros(shape=(len(contours),1), dtype=float)
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

def _findCentreRadius(aCorr, fieldThreshold):
    centroids = []
    radii = []

    contours = measure.find_contours(aCorr, fieldThreshold)
    for n, contour in enumerate(contours):
        x = contour[:, 1]
        y = contour[:, 0]

        centroid = (sum(x) / len(contour), sum(y) / len(contour))
        radius = [np.mean([np.sqrt(np.square(x-centroid[0]) + np.square(y-centroid[1]))])]

        centroids.append(centroid)
        radii.append(radius)

    aCorrCentre = np.zeros(shape=(1,2), dtype=float)
    aCorrCentre[0] = aCorr.shape[0]
    aCorrCentre[0,1] = aCorr.shape[1]
    aCorrCentre = np.ceil(aCorrCentre/2)

    # get all distances and check two minimum of them
    distancesToCentre = cdist(centroids, aCorrCentre)
    sortInd = np.squeeze(np.argsort(distancesToCentre, axis=0))

    # index of the closest field to the centre
    closestFieldInd = sortInd[0]
    twoMinDistances = distancesToCentre[sortInd[:2]]

    areFieldsClose = np.abs(twoMinDistances[0] - twoMinDistances[1])[0] < 2
    if areFieldsClose:
        # two fields with close middle point. Let's select one with minimum square
        areas = np.zeros((1, 2), dtype=float)
        areas[0, 0] = _contourArea(contours, sortInd[0])
        areas[0, 1] = _contourArea(contours, sortInd[1])
        minInd = np.squeeze(np.argsort(areas))
        closestFieldInd = minInd[0]

    radius = radii[closestFieldInd]
    return radius