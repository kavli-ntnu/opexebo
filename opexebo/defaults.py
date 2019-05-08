""" Provide common default values for analysis parameters """

from numpy import inf as np_inf

# spatial/angular occupancy
bin_width = 2.5         # Standard spatial resolution, [cm]
bins_angle = 180        # Standard angular bin **quantity**, integer
speed_cutoff = np_inf   # Standard speed cutoff above which to ignore positions [cm/s]

# Smoothing
sigma = 1               # Standard Gaussian Standard Deviation for smoothing, [bins]


# Firing field related
firing_field_min_bins = 9   # Minimum number of bins for a firing field to be valid [bins]
firing_field_min_peak = 1   # Minimum peak firing rate for a firing field to be valid [Hz]
firing_field_min_mean = 0   # Minimum mean firing rate for a firing field to be valid [Hz]

# Grid Score
field_threshold = 0.2   # The default threshold to search for firing fields
min_orientation = 15    # minimum orientation before a field is considered for Gridness Score  [degrees]


# Border score related
search_width = 8        # Search width for border coverage, see analyses.bordercoverage, [bins]
walls = 'trbl'          # standard walls for border search, see analyses.bordercoverage, string