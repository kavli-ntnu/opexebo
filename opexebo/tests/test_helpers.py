'''' Test helpers'''

test_data_square = r"N:\simoba\opexebo_working_area\test_data\generic\simple_square_input_vars.mat"
test_data_nonsquare = r"N:\simoba\opexebo_working_area\test_data\non-square\input_file_vars.mat"


def get_data_size(data):
    return data['allStatistics'].size