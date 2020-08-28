""" Tests for turning curve"""
import numpy as np
import inspect

from opexebo.analysis import tuning_curve as func

print("=== tests_analysis_turning_curve ===")


###############################################################################
################                MAIN TESTS
###############################################################################


def test_random_data():
    n = 1000
    bw = 15
    spike_angles = np.random.rand(n) * 2 * np.pi
    angular_occupancy = np.random.rand(int(360 / bw))
    func(angular_occupancy, spike_angles, bin_width=bw)
    print(f"{inspect.stack()[0][3]} passed")
    return True


# if __name__ == '__main__':
#    test_random_data()
