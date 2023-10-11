""" Tests for autocorrelogram"""
import inspect
import numpy as np

from opexebo.analysis import angular_occupancy as func

print("=== tests_analysis_angular_occupancy ===")


###############################################################################
################                MAIN TESTS
###############################################################################


def test_angular_random_data():
    """Doesn't test that meaningful results are produced, only that something is"""
    n = 1000
    time = np.arange(n)
    angles = np.random.rand(n)
    a, b, c = func(time, angles)
    print(f"{inspect.stack()[0][3]} passed")


def test_angular_zero_data():
    """Doesn't test that meaningful results are produced, only that something is"""
    n = 1000
    time = np.arange(n)
    angles = np.zeros(n)
    a, b, c = func(time, angles)
    print(f"{inspect.stack()[0][3]} passed")


# if __name__ == '__main__':
#    test_angular_random_data()
#    test_angular_zero_data()
