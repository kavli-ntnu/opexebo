""" Tests for turning curve stats"""
import numpy as np
import inspect

from opexebo.analysis import tuning_curve_stats as func

print("=== tests_analysis_turning_curve_stats ===")


###############################################################################
################                MAIN TESTS
###############################################################################


def test_random_data():
    n = 1000
    data = np.random.rand(n) * 2 * np.pi
    tuning_curve = np.histogram(data, bins=int(360 / 15))[0]
    func(tuning_curve)
    print(f"{inspect.stack()[0][3]} passed")
    return True


# if __name__ == '__main__':
#    test_random_data()
