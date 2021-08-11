# History

## 0.6.0

* Rewrite `spatial_occupancy` and `rate_map` to remove the walk filter (signature breaking)
  * The individual analysis functions should be single purpose, and the walk-filter breaks this rule.
  * Ideally, the walk-filter should be applied to spikes/tracking data _first_, and the resulting data then used for further analysis. 
  * A dedicated function has been provided to replace this: `general.walk_filter`
* Rewrite `shuffle` to better handle edge cases (including 1-spike-only) (signature breaking)
* Added `general.walk_filter` to provide replace the specific filtering implemented in each of `spatial_occupancy` and `rate_map`

* Fix a bug where `accumulate_spatial` can't handle cases where a single spike is present
* Fix an error in the LFP power spectrum calculation with fft




## 0.5.5 (2021-06-01)

* Publish to PyPi


## 0.5.2 (2021-04-19)

* Support 1D data in spatial occupancy and ratemap
* Add Opexebo-specific errors
* Add automated testing prior to release
* Add `calculate_speed`
* Add `egocentric_occupancy`
* Fixed an edge case error in `accumulate_spatial`


## 0.4.3 (2020-02-19)

* Substantial cleanup to `accumulate_spatial`
  - Function was mathematically correct but confusing, due to the difference between opexebo's standard of `(x, y)` and NumPy's standard of `(y, x)`
* Added `upsampling`
* Added `circular_mask`
* Mandatory keyword arguments have been made positional
* `peak_search` updated to handle issues with MaskedArrays containing negative values
* Dcumentation expanded, made Sphinx compatible, and published on ReadTheDocs
* Further work on BorderScore, still experimental though
  - Functional for Rectangular arenas
  - Circular arenas are still WIP, should not be relied upon yet.


## 0.4.2 (2020-02-05)

* Speedscore modified substantially
  - Adaptive filtering added, allowing the user to specify an upper speed as a bandwidth
  - Adaptive filtering adjusted to behave more sensibly in the case of small bin sizes
  - Speeds are smoothed before correlating
* Spatial cross correlation added
* General `bin_number` to `bin_size` code refactored to make future development more consistent
* Assorted unit tests added, including run-once code removed for speed
* Fix errors associated with `>` where applied to MaskedArrays (which, unlike ndarrays, do not correctly obey  the symbol)


## 0.4.0 (2019-12-10)

* Ratemaps updated to handle walk filter (signature now requires providing animal speeds)
* Angular occupancy coverage calculation corrected
* circular arenas now handled in Spatial Occupancy
* Population Vector Correlation added


## 0.3.5 (2019-11-18)

* Assorted bugfixes to HDtuning
* Gridscore to handle NaNs more gracefully


## 0.3.4 (2019-20-25)

* `sep` made an optional dependency
* Alternative, non `sep` based code uses `skimage`
* Attempting to use `sep` if not installed with raise an error


## 0.3.2 (2019-09-20)

* Consistent calculating of bin numbers when histogramming


## 0.3.0 (2019-08-27)

* Third development release
* Improve NaN handling throughout by moving to MaskedArrays
* Fix Angular handling to function in seconds instead of frames
* Remove matplotlib from requirements


## 0.2.0 

* Second development release
  - Implementation of time-map -> rate-map -> (acorr, gridness score)
  - Implementation of rate-map stats, grid stats
  - Implementation of angular-map -> tuning-curve -> head direction score, stats.
  - Non-production-ready implementation of border-score, speed-score. 


## 0.1.0 (2018-08-30)

* First development release
  - Implementation of gridness score, autocorrelogram, place-field detection
