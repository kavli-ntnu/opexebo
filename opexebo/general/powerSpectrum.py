import numpy as np
from scipy import signal

import opexebo.defaults as default


def power_spectrum(values, time_stamps, **kwargs):
    '''
    Calculate the power spectrum of a time-series of data
    
    Parameters
    ----------
    values : np.ndarray
        Amplitude of time series at times `time_stamps`. Must be sampled at a
        constant frequency
    time_stamps : np.ndarray
        Time stamps of time-series. Must be the same length as `values`. 
        Assumed to be in [seconds]
    kwargs
        method : str
            Method for calculating a power spectrum. Accepted values are:
            ["welch", "fft"]. Defaults to "welch"
        fs : float
            Sampling rate. Assumed to be in [Hz]
        scale : str
            Should the power spectrum be returned in a linear form 
            propto v**2/sqrt(Hz)) ("linear"); 
            or decibel scale (dB/sqrt(Hz)) ("log")
            Defaults to "linear"
        resolution : float
            Desired output frequency resolution. Applicable only to Welch's 
            method. Due to the averaging effect of Welch's method, this sets
            the "effective" resolution, which is NOT the same as the minimum
            difference in `frequencies`
            Default 1 [Hz]. Values lower than 1/(time_series_duration) are 
            meaningless
    
    Returns
    -------
    frequencies : np.ndarray
        Discrete Fourier Transform sample frequencies
    power_spectrum : np.ndarray
        Power spectral density at that sample frequency
    '''
    method = kwargs.get("method", default.power_spectrum_method).lower()
    scale = kwargs.get("scale", default.psd_return_scale).lower()

    if scale not in ("linear", "log"):
        raise ValueError(f"Keyword `scale` must be one of ('linear', 'log').")

    if method == "welch":
        func = _power_spectrum_welch
    elif method == "fft":
        func = _power_spectrum_fft
    else:
        raise NotImplementedError(f"Method '{method}' not implemented")

    frequencies, power_spectrum = func(values, time_stamps, **kwargs)

    if scale == "log":
        power_spectrum = 20 * np.log10(power_spectrum)
    
    return frequencies, power_spectrum




def _power_spectrum_welch(values, time_stamps, **kwargs):
    '''Use Welch's method to calculate a power spectrum for a time-series of
    data
    '''
    resolution_multiplier = 32 
    output_resolution = kwargs.get("resolution", default.psd_resolution_welch)
    sampling_frequency = kwargs.get("fs")
    
    n = sampling_frequency * resolution_multiplier / output_resolution
    
    freqs, power_spectral_density = signal.welch(x=values, fs=sampling_frequency, nperseg=n, scaling="density")
    power_spectral_density = np.sqrt(power_spectral_density)
    
    return freqs, power_spectral_density




def _power_spectrum_fft(values, time_stamps, **kwargs):
    '''Calculate the simplest, noisiest, power spectrum by taking the real
    component of the fast Fourier transform of the time-series  
    '''
    sampling_frequency = kwargs.get("fs")
    sampling_time_period = 1/sampling_frequency
    
    amplitude_spectral_density = np.abs(np.fft.rfft(values))
    power_spectral_density = np.square(amplitude_spectral_density)
    freqs = np.fft.rfftfreqs(values.size, sampling_time_period)
    
    return freqs, power_spectral_density
