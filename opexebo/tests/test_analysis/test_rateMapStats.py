""" Tests for autocorrelogram"""
import numpy as np
import inspect

from opexebo.analysis import rate_map_stats as func

print("=== tests_rate_map_stats ===")


###############################################################################
################                MAIN TESTS
###############################################################################


def test_random_data():
    """Doesn't test that meaningful results are produced, only that something is"""
    n = 40
    tmap = np.random.rand(n, n) * 5
    rmap = np.random.rand(n, n)
    func(tmap, rmap)
    print(f"{inspect.stack()[0][3]} passed")


def test_zero_data():
    """Doesn't test that meaningful results are produced, only that something is"""
    n = 40
    tmap = np.ma.zeros((n, n))
    rmap = np.ma.zeros((n, n))
    func(tmap, rmap)
    print(f"{inspect.stack()[0][3]} passed")


# if __name__ == '__main__':
#    test_random_data()
#    test_zero_data()
