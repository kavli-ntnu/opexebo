""" Provide common default values for analysis parameters """



# spatial/angular occupancy
bin_width = 2.5         # Standard spatial resolution, [cm]
bin_angle = 15        # Standard angular bin **quantity**, integer
bin_speed = 5 			# Standard speed bin, [cm/s]
speed_cutoff = 0   # Standard speed cutoff below which to ignore positions [cm/s]

# Smoothing
sigma = 2               # Standard Gaussian Standard Deviation for smoothing, [bins]
sigma_speed = 0.4       # Standard Gaussian stdev for smoothing firing rate. [s]
						# Default at 0.4s from doi:10.1038/nature14622 . 
mask_fill = 0         # Used to define the behaviour of smoothing around masked values
                        # A masked location will have this value inserted
                        # A value of NaN in the astropy.convolution.convolve function will
                        # be replaced by an interpolated value based on nearby locations


# Firing field related
firing_field_min_bins = 9   # Minimum number of bins for a firing field to be valid [bins]
firing_field_min_peak = 1   # Minimum peak firing rate for a firing field to be valid [Hz]
firing_field_min_mean = 0   # Minimum mean firing rate for a firing field to be valid [Hz]
init_thresh = .9 # Starting threshold for place field detection


# Grid Score
field_threshold = 0.2   # The default threshold to search for firing fields
min_orientation = 15    # minimum orientation before a field is considered for Gridness Score  [degrees]


# Speed Score
lower_bound_speed = 2	# The default lower edge of the bandpass filter used for speedscore [cm/s]
upper_bound_time = 10 	# The default time used to calculate the upper edge of the bandpass filter [s]


# Border score related
search_width = 8        # Search width for border coverage, see analyses.bordercoverage, [bins]
walls = 'trbl'          # standard walls for border search, see analyses.bordercoverage, string