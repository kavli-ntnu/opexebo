"""
Provide function for gridness score calculation.
"""
import matplotlib.pyplot as plt
import numpy as np

from scipy.spatial.distance import cdist
from scipy.stats import pearsonr
from skimage import transform
import opexebo
import opexebo.defaults as default


def grid_score(aCorr, **kwargs):
    """Calculate gridness score for an autocorrelogram.

    Calculates a gridness score by expanding a circle around the centre field
    and calculating a correlation value of that circle with it's rotated versions.
    The expansion is done up until the smallest side of the autocorrelogram.
    The function may also calculate grid statistics.

    Gridness score value by itself is calculated as a maximum over a sliding mean
    of expanded circle. The width of the sliding window is given by a variable
    numGridnessRadii. This is done in order to keep gridness score the same as
    historical values (i.e. older versions of gridness score).

    Parameters
    ----------
    acorr: np.ndarray
        A 2D autocorrelogram.
    **kwargs
        min_orientation : int
            See function "grid_score_stats"

    Returns
    -------
    grid_score : float
        Always returns a gridness score value. It ranges from -2 to 2. 2 is
        more of a theoretical bound for a perfect grid. More practical value for
        a good grid is around 1.3. If function can not calculate a gridness
        score, NaN value is returned.
    grid_stats : dictionary
        grid_spacings              : np.array
            Spacing of three adjacent fields closest to center in autocorr
            (in [bins])
        grid_spacing                : float
            Nanmean of 'spacings' in [bins]
        grid_orientations           : np.array
            Orientation of three adjacent fields closest to center in autocorr
            (in [degrees])
        grid_orientations_std       : float
            Standard deviation of orientations % 60
        grid_orientation            : float
            Orientation of grid in [degrees] (mean of fields of 3 main axes)
        grid_positions              : np.array
            [y,x] coordinates of six fields closest to center
        grid_ellipse                : np.array
            Ellipse fit returning 
            [x coordinate, y coordinate, major radius, minor radius, theta]
        grid_ellipse_aspect_ratio   : float
            Ellipse aspect ratio (major radius / minor radius)
        grid_ellipse_theta          : float
            Ellipse theta (corrected according to previous BNT standard) in [degrees]

    See Also
    --------
    BNT.+analyses.gridnessScore
    
    opexebo.analysis.placefield

    Copyright (C) 2018 by Vadim Frolov, (C) 2019 by Simon Ball, Horst Obenhaus

    This program is free software; you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation; either version 3 of the License, or
    (at your option) any later version.
    """
    # Arrange keyword arguments
#    fieldThreshold = kwargs.get("field_threshold", default.field_threshold)
#    minOrientation = kwargs.get("min_orientation", default.min_orientation)
    debug = kwargs.get("debug", False)
    # normalize aCorr in order to find contours
    aCorr = aCorr / aCorr.max()
    centre = -0.5 + np.array(aCorr.shape)/2 # centre : also [y, x]
    cFieldRadius = int(np.floor(_findCentreRadius(aCorr, **kwargs)))

    if cFieldRadius in [-1, 0, 1]:
        if debug:
            print("Terminating due to invalid cFieldRadius")
        return (np.nan, grid_score_stats(np.zeros_like(aCorr), 
                                    np.zeros_like(aCorr), centre))

    halfHeight = np.ceil(aCorr.shape[0]/2)
    halfWidth  = np.ceil(aCorr.shape[1]/2)
    heightIndices = np.arange(aCorr.shape[0])
    widthIndices  = np.arange(aCorr.shape[1])

    # Define radii that will be iterated over for the gridness score
    # outer bound is defined by the minimum of autocorrelogram's dimensions
    # this is need for rectangular autocorrelograms.
    outerBound = int(np.floor(np.min(np.array(aCorr.shape)/2)))
    if outerBound < cFieldRadius:
        if debug:
            print("Terminating due to invalid outerBound"\
                  f" ({outerBound} < {cFieldRadius})")
        return (np.nan, grid_score_stats(np.zeros_like(aCorr), 
                                    np.zeros_like(aCorr), centre))
        
    radii = np.linspace(cFieldRadius+1, outerBound, outerBound-cFieldRadius)
    radii = radii.astype(int)
    numSteps = len(radii)
    if numSteps < 1:
        if debug:
            print("Terminating due to invalud numSteps")
        return (np.nan, grid_score_stats(np.zeros_like(aCorr), 
                                    np.zeros_like(aCorr), centre))

    rotAngles_deg = np.arange(30, 151, 30)  # 30, 60, 90, 120, 150
    rotatedACorr = np.zeros(
            shape=(aCorr.shape[0], aCorr.shape[1], len(rotAngles_deg)),
            dtype=float)
    # we get rotated maps here as it is a heavy operation
    for n, angle in enumerate(rotAngles_deg):
        rotatedACorr[:, :, n] = transform.rotate(aCorr, angle,
                                            preserve_range=True, clip=False)
    rr, cc = np.meshgrid(widthIndices, heightIndices, sparse=False)
    # This is needed for compatibility with Matlab's code
    rr += 1
    cc += 1
    mainCircle = np.sqrt(np.power((rr - halfWidth), 2) + np.power(cc-halfHeight, 2))
    innerCircle = mainCircle > cFieldRadius


    GNS = np.zeros(shape=(numSteps, 2), dtype=float)
    for i, radius in enumerate(radii):
        mask = innerCircle & (mainCircle < radius)
        aCorrValues = aCorr[mask]

        rotCorr = np.zeros_like(rotAngles_deg).astype(float)
        for j, angle in enumerate(rotAngles_deg):
            rotatedValues = rotatedACorr[mask, j]
            r, p = pearsonr(aCorrValues, rotatedValues)
            rotCorr[j] = r
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
    # Mask center field and fringes of autocorrelogram > best grid score radius
    mask_outwards = _circular_mask(aCorr, radii[gscoreInd]*1.25, 'outwards', centre)
    mask_center   = _circular_mask(aCorr, cFieldRadius*1.5, 'inwards', centre)
    mask = mask_outwards + mask_center

    grid_stats = grid_score_stats(aCorr, mask, centre, **kwargs)

    return gscore, grid_stats

#########################################################
################        Helper Functions
#########################################################


def grid_score_stats(aCorr, mask, centre, **kwargs):
    '''
    Calculate spatial characteristics of grid based on 2D autocorr

    Parameters
    ----------
    aCorr : np.array
        2D Autocorrelation
    mask : np.array
        Mask (masked=True) of shape aCorr for masking center field and
        fringes of aCorr above best grid score radius
    centre : np.array
        Centre coordinate [y,x]
    **kwargs :
        min_orientation : int
            Minimum difference in degrees that two neighbouring fields
            detected in 2D autocorrelation must have. If difference is
            below this threshold, discard the field that has larger
            distance from center

    Returns
    -------
    grid_stats : dictionary
        grid_spacings              : np.array
            Spacing of three adjacent fields closest to center in autocorr
            (in [bins])
        grid_spacing                : float
            Nanmean of 'spacings' in [bins]
        grid_orientations           : np.array
            Orientation of three adjacent fields closest to center in autocorr
            (in [degrees])
        grid_orientations_std       : float
            Standard deviation of orientations % 60
        grid_orientation            : float
            Orientation of grid in [degrees] (mean of fields of 3 main axes)
        grid_positions              : np.array
            [y,x] coordinates of six fields closest to center
        grid_ellipse                : np.array
            Ellipse fit returning 
            [x coordinate, y coordinate, major radius, minor radius, theta]
        grid_ellipse_aspect_ratio   : float
            Ellipse aspect ratio (major radius / minor radius)
        grid_ellipse_theta          : float
            Ellipse theta (corrected according to previous BNT standard) in [degrees]
    '''

    # Get kwargs
    debug = kwargs.get('debug', False)
    min_orientation = kwargs.get('min_orientation', default.min_orientation)
    min_orientation = np.radians(min_orientation)
    if debug:
        print('Min orientation: {} degrees'.format(np.degrees(min_orientation)))
    

    # Initialise default output in case stats are uncalculable
    gs_ellipse_theta    = np.nan
    gs_ellipse          = np.nan
    gs_aspect_ratio     = np.nan
    gs_orientations_std = np.nan
    gs_orientation      = np.nan
    gs_positions        = np.full(6, fill_value = np.nan, dtype=float)
    gs_orientations     = np.full(3, fill_value = np.nan, dtype=float)
    gs_spacings         = np.full(3, fill_value = np.nan, dtype=float)

    # Find fields in autocorrelogram
    all_coords = opexebo.general.peak_search(aCorr, mask=mask, search_method="sep",
                                             null_background=True, threshold=0.1)

    if all_coords.shape[0] >= 6:
        # Calculate orientation and distance of all local maxima to center
        orientation = np.arctan2(all_coords[:,0] - centre[0], all_coords[:,1] - centre[1]) # in radians
        distance = np.sqrt(np.square(all_coords[:,0]-centre[0]) + np.square(all_coords[:,1]-centre[1]))

        # Where two fields have a very similar orientation, discard the more distant one
        orient_distsq = np.abs(_circ_dist2(orientation))
        close_fields = orient_distsq < min_orientation
        close_fields = np.triu(close_fields, 1) # Upper triangle only - set lower triangle to zero
                                                # k=1: +1 offset from diagonal: set diagonal to zero too
        to_del = []
        for row,col in np.argwhere(close_fields):
            if distance[row] > distance[col]:
                to_del.append(row)
            else:
                to_del.append(col)

        distance    = np.delete(distance, to_del)
        orientation = np.delete(orientation, to_del)
        all_coords  = np.delete(all_coords, to_del, axis=0)

        # First sort by distance and take first 6 fields
        sorted_ids_dist = np.argsort(distance)[:6] # 6 closest fields
        positions = all_coords[sorted_ids_dist]
        distance = distance[sorted_ids_dist]
        orientation = orientation[sorted_ids_dist]
        
        # ... then re-sort remaining fields by angle
        sorted_ids_ang = np.argsort(orientation)

        ################# GATHER OUTPUT #################
        gs_positions = positions[sorted_ids_ang]
        # For grid orientation and spacing take only 3 out of 6 neighbouring fields
        gs_spacings = distance[sorted_ids_ang][:3]
        gs_orientations = np.degrees(orientation[sorted_ids_ang][:3]) % 180

        
        if debug:
            aCorr_masked = np.ma.masked_where(mask, aCorr.copy())
            plt.imshow(aCorr_masked)
            plt.scatter(centre[1],centre[0], s=600, marker='x', color='black')
            for field_no, coord in enumerate(gs_positions):
                plt.scatter(coord[1], coord[0], s=300, marker='x', color='red')
                plt.text(coord[1]+3, coord[0], field_no, label='Center')
            plt.title('Masked autocorr + 6 remaining fields')

        # Fit an ellipse to those remaining fields:
        if len(gs_positions) > 2:
            gs_ellipse =  opexebo.general.fit_ellipse(gs_positions[:,1], gs_positions[:,0])
            gs_ellipse_theta = np.degrees(gs_ellipse[4]+np.pi)%360
            # The +pi term was included in the original BNT, I have kept it to
            # maintain consistency with past results.
            gs_aspect_ratio = gs_ellipse[2]/gs_ellipse[3] # Major radius / Minor radius

        # Work out mean orientation of grid. Take standard deviation as quality marker
        gs_orientation       = np.nanmean(gs_orientations % 60)
        gs_orientations_std  = np.nanstd(gs_orientations % 60)
        # Find out polarity of rotation
        if np.argmin([np.abs(gs_orientation-60), np.abs(gs_orientation)]) == 0:
            gs_orientation -= 60

    else:
        if debug: 
            print('Not enough fields detected ({})'.format(len(all_coords)))
            

    grid_stats = {'grid_spacings': gs_spacings,
                  'grid_spacing': np.nanmean(gs_spacings),
                  'grid_orientations': gs_orientations,
                  'grid_orientations_std': gs_orientations_std,
                  'grid_orientation': gs_orientation,
                  'grid_positions': gs_positions,
                  'grid_ellipse': gs_ellipse,
                  'grid_ellipse_aspect_ratio': gs_aspect_ratio,
                  'grid_ellipse_theta': gs_ellipse_theta}
    return grid_stats


def _circular_mask(image, radius, polarity='outwards', center=None):
        '''
        Given height and width, create circular mask around point with defined radius
        Polarity:
            'inwards' : True inside,  False outside
            'outwards': True outside, False inside
        '''
        h = image.shape[0]
        w = image.shape[1]

        if center is None:
            center = [int(h/2), int(w/2)]

        Y, X = np.ogrid[:h, :w]
        dist_from_center = np.sqrt(np.power(X - center[1],2) + np.power(Y-center[0],2))
        if polarity.lower() == 'inwards':
            mask = dist_from_center <= radius
        elif polarity.lower() == 'outwards':
            mask = dist_from_center >= radius
        else:
            raise ValueError('Polarity "{}" not defined'.format(polarity))
        return mask

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



def _polyArea(x, y):
    '''Polygon area, from 
    https://stackoverflow.com/questions/24467972/calculate-area-of-polygon-given-x-y-coordinates
    Seems to be the same as Matlab's polyarea'''
    return 0.5*np.abs(np.dot(x, np.roll(y, 1)) - np.dot(y, np.roll(x, 1)))


def _contourArea(contours, i):
    contour = contours[i]
    x = contour[:, 1]
    y = contour[:, 0]
    area = _polyArea(x, y)
    return area


def _findCentreRadius(aCorr, **kwargs):
    debug = kwargs.get("debug", False)
    halfHeight = np.ceil(aCorr.shape[0]/2)
    halfWidth = np.ceil(aCorr.shape[1]/2)
    peak_coords = np.ones(shape=(1, 2), dtype=np.int)
    peak_coords[0, 0] = halfHeight-1
    peak_coords[0, 1] = halfWidth-1
    fields = opexebo.analysis.place_field(aCorr, min_bins=5, min_peak=0, min_mean=0, init_thresh=.95, \
                                         peak_coords=peak_coords)[0] # Fix all input args for now
    if fields is None or len(fields) == 0:
        if debug:
            print("Terminating _findCentreRadius due to no fields")
        return 0
    elif debug:
        print(f"Fields found: {len(fields)}")
    else:
        pass

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
    
    if debug:
        print("radius is {}".format(radius))
        
    return radius


