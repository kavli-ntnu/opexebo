'''Opexebo default values

This module provides default values for all keyword arguments in the Opexebo 
package. 

'''


'''spatial/angular occupancy'''
#: Standard spatial resolution, [cm]
bin_width = 2.5

#: Standard angular bin width [degree]
bin_angle = 15

#: Standard speed bin, [cm/s]
bin_speed = 5

#: Standard speed cutoff below which to ignore positions [cm/s]
speed_cutoff = 0

#: Shape of area: assume a rectangular arena
shape = "rectangular"
shapes_square = ("square", "rectangle", "rectangular", "rect", "s", "r")
shapes_circle = ("circ", "circular", "circle", "c")
shapes_linear = ("linear", "line", "l")



'''Smoothing'''
#: Standard Gaussian Standard Deviation for smoothing, [bins]
sigma = 2.0

#: Standard Gaussian stdev for smoothing firing rat, [s]. Default at 0.4s from doi 10.1038/nature14622 .     
sigma_speed = 0.4

#: Replacement value for masked values when smoothing. Use np.nan to interpolate through masked values instead of use fixed value
mask_fill = 0



'''LFP related'''
#: Default method for calculating a power spectrum
power_spectrum_method = "welch"

#: Default effective resolution for Welch's method (Hz)
psd_resolution_welch = 1

#: Default return scale, i.e. linear or decibel scale
psd_return_scale = "linear"



''' Firing field related'''
#: Initial relative threshold to search for local maxima, in range [0,1]
initial_search_threshold = 0.96

#: Minimum number of bins for a firing field to be valid [bins]
firing_field_min_bins = 9

#: Minimum peak firing rate for a firing field to be valid [Hz]
firing_field_min_peak = 1

#: Minimum mean firing rate for a firing field to be valid [Hz]
firing_field_min_mean = 0.1

#: The method used to find local maxima
search_method = 'default'

#: All implemented means of finding local maxima. Use lower case. 
all_methods = (search_method, "sep", "not implemented")    




''' Score related'''
#: Percentile for head drection arc, in range [0,1]
hd_percentile = 0.95

#: minimum angular separation between fields in acorr considered for grid ellipse [degrees]
min_orientation = 15


#: The default lower edge of the bandpass filter used for speedscore [cm/s]
lower_bound_speed = 2
#: The default time used to calculate the upper edge of the bandpass filter [s]
upper_bound_time = 10


#: Search width for border coverage, see analyses.bordercoverage, [bins]
search_width = 8
#: standard walls for border search, see analyses.bordercoverage, string    
walls = 'trbl'