""" Tests for grid score"""
import opexebo.tests as th
import opexebo
from opexebo.analysis import grid_score as func

print("=== tests_analysis_grid_score ===")


###############################################################################
################                MAIN TESTS
###############################################################################


def test_perfect_grid_cell():
    firing_map = th.generate_2d_map(
        "rect",
        1,
        x=80,
        y=80,
        coverage=0.95,
        fields=th.generate_hexagonal_grid_fields_dict(),
    )
    acorr = opexebo.analysis.autocorrelation(firing_map)
    gs, stats = func(acorr, debug=True, search_method="default")
    assert stats["grid_ellipse_aspect_ratio"] > 0.985
    assert stats["grid_ellipse_aspect_ratio"] < 1.15


# if __name__ == '__main__':
#    test_perfect_grid_cell()
