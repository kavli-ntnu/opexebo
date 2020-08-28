""" Tests for autocorrelogram"""
import numpy as np
import inspect

from opexebo.analysis import rate_map_coherence as func

print("=== tests_rate_map_coherence ===")


###############################################################################
################                MAIN TESTS
###############################################################################


def test_random_data():
    """Doesn't test that meaningful results are produced, only that something is"""
    n = 40
    rmap = np.random.rand(n, n)
    func(rmap)
    print(f"{inspect.stack()[0][3]} passed")
    return True


def test_zero_data():
    """Doesn't test that meaningful results are produced, only that something is"""
    n = 40
    rmap = np.zeros((n, n))
    func(rmap)
    print(f"{inspect.stack()[0][3]} passed")
    return True


def test_flat_data():
    """Doesn't test that meaningful results are produced, only that something is"""
    n = 40
    rmap = np.ones((n, n))
    func(rmap)
    print(f"{inspect.stack()[0][3]} passed")
    return True


# if __name__ == '__main__':
#    test_random_data()
#    test_zero_data()
#    test_flat_data()
