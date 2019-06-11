# -*- coding: utf-8 -*-
"""
Created on Tue Jun  4 11:37:57 2019

@author: simoba
"""

import numpy as np

def fitellipse(X, Y, **kwargs):
    '''
    Fit an ellipse to the provided set of X, Y co-ordinates
    
    Based on the approach taken in 
        Authors: Andrew Fitzgibbon, Maurizio Pilu, Bob Fisher
        Reference: "Direct Least Squares Fitting of Ellipses", IEEE T-PAMI, 1999
        
         @Article{Fitzgibbon99,
          author = "Fitzgibbon, A.~W.and Pilu, M. and Fisher, R.~B.",
          title = "Direct least-squares fitting of ellipses",
          journal = pami,
          year = 1999,
          volume = 21,
          number = 5,
          month = may,
          pages = "476--480"
         }
    and implemented in MatLab by Vadim Frolov
    
    Parameters
    ----------
    X - np.ndarray
        x co-ordinates of points
    Y - np.ndarray
        y co-ordinates of points
    
    Returns
    -------
    x_centre : float
        Centre of ellipse
    y_centre : float
        Centre of ellipse
    Ru : float
        Major radius
    Rv : float
        Minor radius
    theta_rad : float
        Ellipse orientation (in radians)
        
    
    See also
    --------
    BNT.+general.fitEllipse
    opexebo.analysis.gridscore
    
    Copyright (C) 2019 by Simon Ball
    
    This program is free software; you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation; either version 3 of the License, or
    (at your option) any later version.
    '''
    debug = kwargs.get('debug', False)
    if X.size != Y.size:
        raise ValueError("X and Y must be the same length. You provided (%d, %d)\
        values respectively" % (X.size, Y.size))
    # Normalise the data and move it to the origin
    mx = np.mean(X)
    my = np.mean(Y)
    
    sx = 0.5 * (np.max(X) - np.min(X))
    sy = 0.5 * (np.max(Y) - np.min(Y))
    
    x = (X - mx) / sx
    y = (Y - my) / sy
    
    if debug:
        x = X
        y = Y
    
    # Construct design matrix and scatter matrix
    D = np.zeros((X.size,6))
    D[:,0] = x*x
    D[:,1] = x*y
    D[:,2] = y*y
    D[:,3] = x
    D[:,4] = y
    D[:, 5] = np.ones(X.size)
    
    S = D.T @ D
    
    # Construct contraint matrix
    C = np.zeros((6,6))
    C[1,1] = 1
    C[2,0] = -2
    C[0,2] = -2
    # Solve eigensystem
    # Break into blocks
    tmpA = S[:3, :3]
    tmpB = S[:3, 3:]
    tmpC = S[3:, 3:]
    tmpD = C[:3, :3]
    tmpE = np.linalg.inv(tmpC) @ tmpB.T
    
    eval_x, evec_x = np.linalg.eig(np.linalg.inv(tmpD) @ (tmpA - (tmpB@tmpE)))

    # Find the positive eigenvalue (as det(tmpD) < 0)
    idx = np.argmax(np.logical_and(np.real(eval_x) < 1e-8, np.isfinite(eval_x)))
    vec_x = np.real(evec_x[:,idx]) # vector associated with idx
    vec_y = -tmpE @ vec_x
    A = np.concatenate((vec_x, vec_y))



    # Un-normalise
    par = np.zeros(6)
    par[0] = A[0] * sy * sy
    par[1] = A[1] * sx * sy
    par[2] = A[2] * sx * sx
    par[3] = (-2*A[0]*sy*sy*mx) - (A[1]*sx*sy*my) + (A[3]*sx*sy*sy)
    par[4] = (-A[1]*sx*sy*mx) - (2*A[2]*sx*sx*my) + (A[4]*sx*sx*sy)
    par[5] = (A[0]*sy*sy*mx*mx) + (A[1]*sx*sy*mx*my) + (A[2]*sx*sx*my*my) \
            - (A[3]*sx*sy*sy*mx) - (A[4]*sx*sx*sy*my) + (A[5]*sx*sx*sy*sy)

    # Geometric radii and centres 
    theta_rad = 0.5*np.arctan2(par[1], par[0]-par[2])
    cos_t = np.cos(theta_rad)
    sin_t = np.sin(theta_rad)
    sin2 = sin_t * sin_t
    cos2 = cos_t * cos_t
    Ao = par[5]
    Au = (par[3] * cos_t) + (par[4] * sin_t)
    Av = (-par[3] * sin_t) + (par[4] * cos_t)
    Auu = (par[0] * cos2) + (par[2] * sin2) + (par[1] * cos_t * sin_t)
    Avv = (par[0] * sin2) + (par[2] * cos2) - (par[1] * cos_t * sin_t)
    
    
    tu_centre = -Au / (2*Auu)
    tv_centre = -Av / (2*Avv)
    w_centre = Ao - (Auu*tu_centre*tu_centre) - (Avv*tv_centre*tv_centre)
    
    x_centre = (tu_centre * cos_t) - (tv_centre * sin_t)
    y_centre = (tu_centre * sin_t) + (tv_centre * cos_t)
    
    Ru = -w_centre / Auu
    Rv = -w_centre / Avv
    
    Ru = np.sqrt(np.abs(Ru)) * np.sign(Ru)
    Rv = np.sqrt(np.abs(Rv)) * np.sign(Rv)
    
    return x_centre, y_centre, Ru, Rv, theta_rad
    
    
