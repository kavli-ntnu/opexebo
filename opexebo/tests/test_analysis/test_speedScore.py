"""Tests for SpeedScore"""
import numpy as np
import pytest

from opexebo.analysis import speed_score as func

print("=== tests_analysis_speedScore===")


def test_invalid_inputs():
    n = 1000
    # Wrong dimensions for inputs
    # Tracking times, speeds not the same size
    with pytest.raises(ValueError):
        spike_times = np.sort(np.random.rand(n)) * n
        tracking_times = np.arange(n)
        tracking_speeds = np.random.rand(n + 10) * 30
        func(spike_times, tracking_times, tracking_speeds)

    # tracking_speeds wrong dimensions
    with pytest.raises(ValueError):
        spike_times = np.sort(np.random.rand(n)) * n
        tracking_times = np.arange(n)
        tracking_speeds = np.random.rand(n, 2) * 30
        func(spike_times, tracking_times, tracking_speeds)

    # Invalid bandpass type
    with pytest.raises(NotImplementedError):
        spike_times = np.sort(np.random.rand(n)) * n
        tracking_times = np.arange(n)
        tracking_speeds = np.random.rand(n) * 30
        bandpass = "xyz"
        func(spike_times, tracking_times, tracking_speeds, bandpass=bandpass)

    # Impossibly high speed_bandwidth
    with pytest.raises(ValueError):
        spike_times = np.sort(np.random.rand(n)) * n
        tracking_times = np.arange(n)
        tracking_speeds = np.random.rand(n) * 30
        kwargs = {
            "bandpass": "adaptive",
            "lower_bound_speed": 2,
            "upper_bound_time": 10,
            "speed_bandwidth": 0.01,
        }
        func(spike_times, tracking_times, tracking_speeds, **kwargs)


def test_random_inputs():
    """Try randomised inputs and check we get the right _kind_ of outputs"""
    n = 10000
    spike_times = np.sort(np.random.rand(n)) * n
    tracking_times = np.arange(n)
    tracking_speeds = np.random.rand(n) * 30
    for kwargs in (
        {"bandpass": "none"},
        {"bandpass": "fixed", "lower_bound_speed": 2, "upper_bound_speed": 10},
        {
            "bandpass": "adaptive",
            "lower_bound_speed": 2,
            "upper_bound_time": 10,
            "speed_bandwidth": 2,
        },
    ):

        ss, thresholds = func(spike_times, tracking_times, tracking_speeds, **kwargs)
        for val in ss.values():
            assert np.isfinite(val)
            assert -1 <= val <= 1


# if __name__ == '__main__':
#    test_invalid_inputs()
#    test_random_inputs()
