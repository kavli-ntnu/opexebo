""" Tests for get_speed"""
import numpy as np
import math
import pytest
from opexebo.analysis import calc_speed as func


def test_invalid_inputs():
    n = 1000
    t = np.arange(n)
    invalid_kwargs = [
            {"t": t, "x": np.arange(n-1)},
            {"t": t, "x": t, "y": np.arange(n-1)},
            {"t": t, "x": 3},
            {"t": t, "x": t, "moving_average": 2.3},
            {"t": t, "x": t, "moving_average": 2},
            {"t": t, "x": t, "moving_average": 4},
            ]
    for kwargs in invalid_kwargs:
        with pytest.raises(ValueError):
            func(**kwargs)
    return

def test_valid_inputs():
    n = 1000
    t = np.arange(n, dtype=np.float64)
    nan_x = t.copy()
    nan_x[27] = np.nan
    inf_x = t.copy()
    inf_x[35] = np.inf
    mass_nan_x = t.copy()
    mass_nan_x[np.random.rand(n)>0.6] = np.nan
    valid_kwargs = [
            {"t": t, "x": t},
            {"t": t, "x": t, "y": t},
            {"t": t, "x": t, "moving_average": 5},
            {"t": t, "x": nan_x},
            {"t": t, "x": inf_x},
            {"t": t, "x": mass_nan_x}
            ]
    for kwargs in valid_kwargs:
        func(**kwargs)
    return

def test_known_cases():
    # constant speed:
    n = 1000
    t = np.arange(n)
    x = np.arange(n)
    speed = func(t, x)
    assert math.isclose(np.std(speed), 0, abs_tol=1e-6)
    assert speed[0] == 1
    assert speed[-1] == 1
    
    # constant acceleration
    # Should yield a speed that is equal to arange(n) EXCEPT for speed[0]
    t = np.arange(n)
    x = np.zeros(n)
    for i in np.arange(n-1):
        x[i+1] = x[i] + i+1
    speed = func(t, x)
    accel = np.diff(speed[1:])
    assert math.isclose(np.std(accel), 0, abs_tol=1e-6)
    assert accel[0] == 1
    assert speed[0] == 1
    assert speed[57] == 57
    
    # Orbiting a point
    t = np.arange(n)
    r = 28
    x = r * np.cos(2*np.pi*t/n)
    y = r * np.sin(2*np.pi*t/n)
    speed = func(t, x, y)
    assert math.isclose(np.std(speed), 0, abs_tol=1e-6)
    d = 2 * r * np.pi
    assert math.isclose(speed[193], d/n, rel_tol = 1e-4)
    
    
        
        