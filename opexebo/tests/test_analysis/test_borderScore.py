""" Tests for spatial occupancy"""
import numpy as np
import matplotlib.pyplot as plt

from opexebo.analysis import border_score as func

print("=== tests_analysis_border_score ===")


###############################################################################
################                HELPER FUNCTIONS
###############################################################################


def field(grid, centre, width, peak):
    """
    grid: array
        the binned arena, e.g. np.zeros((40,40))
    centre: tuple
        x, y co-ordinates for the field to be centred
        (0,0) is assumed to be upper left
    width : tuple or float
        x, y sigma. If float, x=y
    peak : float
        peak rate
    """
    if type(width) in (float, int):
        width = (width, width)
    x = np.arange(grid.shape[1])
    y = np.arange(grid.shape[0])
    X, Y = np.meshgrid(x, y)
    xt = np.square(X - centre[0]) / (2 * width[0] ** 2)
    yt = np.square(Y - centre[1]) / (2 * width[1] ** 2)
    g = np.exp(-1 * (xt + yt))
    g[g < 0.025] = 0
    g *= peak
    return g


def show(f):
    plt.figure()
    plt.imshow(f)
    plt.show()


def rmap0():
    """f2 should have a border score of about 0.25 on the Right wall"""
    f = np.zeros((40, 40))
    f0 = field(f, (5, 14), (2.5, 4), 15)
    f1 = field(f, (23, 17), (1, 1.5), 10)
    f2 = field(f, (35, 3), (8, 8), 5)
    f3 = field(f, (12, 28), (6, 2), 6)
    f4 = field(f, (36, 37), (5, 3.3), 7)
    rmap0 = f0 + f1 + f2 + f3 + f4
    fmap0 = rmap0 > 3
    fields0 = [{"map": f0}, {"map": f1}, {"map": f2}, {"map": f3}, {"map": f4}]
    return rmap0, fmap0, fields0


def rmap1():
    """f0 should have a border score of 1 on the bottom wall"""
    f = np.zeros((40, 40))
    f0 = field(f, (20, 0), (40, 0.5), 15)
    f1 = field(f, (6, 32), (3, 3), 10)
    rmap1 = f0 + f1
    fmap1 = rmap1 > 3
    fields = [{"field_map": f0}, {"field_map": f1}]
    return rmap1, fmap1, fields


###############################################################################
################                MAIN TESTS
###############################################################################


def test_zero_fields():
    rmap = np.zeros((40, 40))
    fmap = rmap.copy()
    fields = []
    bs, cov = func(rmap, fmap, fields, arena_shape="s")
    assert bs == -1
    print("test_zero_fields() passed")


def test_perfect_fields():
    """Note: a perfect score (1) can only be achieved with a field that is full
    width and infinitesimal depth. Here we are testing with a 40x40 field, and
    a perfect score is 0.7288348...
    """
    bs, cov = func(*rmap1(), arena_shape="s", walls="B")
    assert bs >= 0.728 and bs <= 0.729


# if __name__ == '__main__':
#    test_zero_fields()
#    test_perfect_fields()
#    show(rmap1()[0])
