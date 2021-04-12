"""
Python version of Matlab's normxcorr2_general

This is a Python adaption of code found at
https://se.mathworks.com/matlabcentral/fileexchange/29005-generalized-normalized-cross-correlation
Since we use it for autocorrelograms exclusively some input arguments of the
original function have been dropped.
"""

from __future__ import division

import numpy as np
import numpy.matlib  # Not included in the default numpy namespace
from scipy.signal import convolve2d

REQUIRED_OVERLAP_PIXELS = 0


def normxcorr2_general(array):
    """Calculate spatial autocorrelation.

    Python implementation of the Matlab `generalized-normalized cross correlation`
    function, adapted by Vadim Frolov. Some generality was abandoned in the adaption
    as unnecessary for autocorrelogram calculation.
    
    For the original function, see https://se.mathworks.com/matlabcentral/fileexchange/29005-generalized-normalized-cross-correlation

    Parameters
    ----------
    array: NxM matrix
        firing array. array is not necessary a numpy array. Must not contain NaNs!

    Returns
    -------
    np.ndarray
        Resulting correlation matrix
    """

    if not isinstance(array, np.ndarray):
        array = np.array(array)
    if not np.sum(np.isfinite(array)) == array.size:
        raise ValueError("Input array contains NaN values.")

    
    A = _shift_data(array)
    T = _shift_data(array)

    number_of_overlap_pixels = _local_sum(np.ones(A.shape), T.shape[0], T.shape[1])

    local_sum_A = _local_sum(A, T.shape[0], T.shape[1])
    local_sum_A2 = _local_sum(A*A, T.shape[0], T.shape[1])

    # Note: diff_local_sums should be nonnegative, but it may have negative
    # values due to round off errors. Below, we use max to ensure the radicand
    # is nonnegative.
    diff_local_sums_A = (local_sum_A2 - np.power(local_sum_A, 2) / number_of_overlap_pixels)
    del local_sum_A2

    denom_A = np.maximum(diff_local_sums_A, 0)
    del diff_local_sums_A

    # Flip T in both dimensions so that its correlation can be more easily
    # handled.
    rotatedT = np.rot90(T, 2)
    local_sum_T = _local_sum(rotatedT, A.shape[0], A.shape[1])
    local_sum_T2 = _local_sum(rotatedT*rotatedT, A.shape[0], A.shape[1])
    del rotatedT

    diff_local_sums_T = (local_sum_T2 - np.power(local_sum_T, 2) / number_of_overlap_pixels)
    del local_sum_T2
    denom_T = np.maximum(diff_local_sums_T, 0)
    del diff_local_sums_T

    denom = np.sqrt(denom_T * denom_A)
    del denom_T, denom_A

    xcorr_TA = _xcorr2_fast(T, A)
    del A, T
    numerator = xcorr_TA - local_sum_A * local_sum_T / number_of_overlap_pixels
    del xcorr_TA, local_sum_A, local_sum_T

    # denom is the sqrt of the product of positive numbers so it must be
    # positive or zero.  Therefore, the only danger in dividing the numerator
    # by the denominator is when dividing by zero. We know denom_T~=0 from
    # input parsing; so denom is only zero where denom_A is zero, and in these
    # locations, C is also zero.
    C = np.zeros(numerator.shape)
    tol = 1000 * np.spacing(np.max(np.abs(denom)))
    i_nonzero = (denom > tol)
    C[i_nonzero] = numerator[i_nonzero] / denom[i_nonzero]
    del numerator, denom

    # Remove the border values since they result from calculations using very
    # few pixels and are thus statistically unstable.
    # By default, REQUIRED_OVERLAP_PIXELS = 0, so C is not modified.
    if REQUIRED_OVERLAP_PIXELS > np.max(number_of_overlap_pixels):
        raise ValueError("ERROR: REQUIRED_OVERLAP_PIXELS")

    C[number_of_overlap_pixels < REQUIRED_OVERLAP_PIXELS] = 0
    return C


def _xcorr2_fast(T, A):
    T_size = T.shape
    A_size = A.shape
    outsize = np.array(A.shape) + np.array(T.shape) - 1

    # Figure out when to use spatial domain vs. freq domain
    conv_time = _time_conv2(T_size, A_size)  # 1 conv2
    fft_time = 3*_time_fft2(outsize)  # 2 fft2 + 1 ifft2

    if conv_time < fft_time:
        cross_corr = convolve2d(np.rot90(T, 2), A)
    else:
        cross_corr = _freqxcorr(T, A, outsize)

    return cross_corr


def _freqxcorr(a, b, outsize):
    # Find the next largest size that is a multiple of a combination of 2, 3,
    # and/or 5.  This makes the FFT calculation much faster.
    optimalSize = np.zeros((2, 1))
    optimalSize[0] = _find_closest_valid_dimension(outsize[0])
    optimalSize[1] = _find_closest_valid_dimension(outsize[1])
    optimalSize = optimalSize.squeeze()
    optimalSize = optimalSize.astype(np.integer)

    # Calculate correlation in frequency domain
    rot_version = np.rot90(a, 2)
    Fa = np.fft.fft2(rot_version, s=(optimalSize[0], optimalSize[1]))
    Fb = np.fft.fft2(b, s=(optimalSize[0], optimalSize[1]))
    xcorr_ab = np.real(np.fft.ifft2(Fa * Fb))

    xcorr_ab = xcorr_ab[0:outsize[0], 0:outsize[1]]
    return xcorr_ab


def _time_conv2(obssize, refsize):
    # K was empirically calculated by the commented-out code above.
    K = 2.7e-8

    # convolution time = K*prod(obssize)*prod(refsize)
    time = K * np.prod(obssize) * np.prod(refsize)
    return time


def _time_fft2(outsize):
    # time a frequency domain convolution by timing two one-dimensional ffts

    R = outsize[0]
    S = outsize[1]

    # Tr = time_fft(R)
    # K_fft = Tr/(R*log(R))

    # K_fft was empirically calculated by the 2 commented-out lines above.
    K_fft = 3.3e-7
    Tr = K_fft * R * np.log(R)

    if S == R:
        Ts = Tr
    else:
        # Ts = time_fft(S)  % uncomment to estimate explicitly
        Ts = K_fft * S * np.log(S)

    time = S*Tr + R*Ts
    return time


def _local_sum(A, m, n):
    """
    This algorithm depends on precomputing running sums.

    If m, n are equal to the size of A, a faster method can be used for
    calculating the local sum.  Otherwise, the slower but more general method
    can be used.  The faster method is more than twice as fast and is also
    less memory intensive.
    
    As it is currently called (2021-04-12), the `else` case appears to never
    be invoked
    """
    if m == A.shape[0] and n == A.shape[1]:
        s = np.cumsum(A, axis=0)
        secondPart = np.matlib.repmat(s[-1, :], m-1, 1) - s[0:-1, :]
        c = np.concatenate((s, secondPart), axis=0)
        s = np.cumsum(c, axis=1)
        del c
        lastColumn = s[:, -1].reshape((s.shape[0], 1))
        secondPart = np.matlib.repmat(lastColumn, 1, n-1) - s[:, 0:-1]
        local_sum_A = np.concatenate((s, secondPart), axis=1)
    else:
        # breal the padding into parts to save on memory
        B = np.zeros((A.shape[0] + 2*m, A.shape[1]))

#        B(m+1:m+size(A,1),:) = A
#        s = cumsum(B,1)
#        c = s(1+m:end-1,:)-s(1:end-m-1,:)
#        d = zeros(size(c,1),size(c,2)+2*n)
#        d(:,n+1:n+size(c,2)) = c
#        s = cumsum(d,2)
#        local_sum_A = s(:,1+n:end-1)-s(:,1:end-n-1)
        local_sum_A = 0

    return local_sum_A


# we assume that we only deal with float number
def _shift_data(A):
    """
    Convert array to type Float, and shift the data range to be greater than zero
    """
    B = A.astype(np.float)

    if not np.issubdtype(A.dtype, np.unsignedinteger):
        min_B = np.min(B)
        if min_B < 0:
            B -= min_B
    return B


def _find_closest_valid_dimension(n):

    # Find the closest valid dimension above the desired dimension.  This
    # will be a combination of 2s, 3s, and 5s.

    # Incrementally add 1 to the size until
    # we reach a size that can be properly factored.
    new_number = n
    result = 0
    new_number -= 1
    while not result == 1:
        new_number += 1
        result = _factorize_number(new_number)

    return new_number


def _factorize_number(n):
    for ifac in np.array([2, 3, 5]):
        while np.fmod(n, ifac) == 0:
            n = n / ifac
    return n
