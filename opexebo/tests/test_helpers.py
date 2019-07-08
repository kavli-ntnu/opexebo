'''' Test helpers'''

test_data_square = r"N:\simoba\opexebo_working_area\test_data\generic\simple_square_input_vars.mat"
test_data_nonsquare = r"N:\simoba\opexebo_working_area\test_data\non-square\input_file_vars.mat"
test_data_big = r"C:\Users\simoba\Documents\_work\Kavli\bntComp\Output_2\auto_input_file_vars_2.mat"


def get_data_size(data):
    return data['allStatistics'].size

def get_time_map_bnt(data, key):
    time_smoothed = data['cellsData'][key,0]['epochs'][0,0][0,0]['map'][0,0]['time'][0,0]
    time_raw = data['cellsData'][key,0]['epochs'][0,0][0,0]['map'][0,0]['timeRaw'][0,0]
    return time_smoothed, time_raw

def get_ratemap_bnt(data, key):
    rmap_smooth = data['cellsData'][key,0]['epochs'][0,0][0,0]['map'][0,0]['z'][0,0]
    rmap_raw = data['cellsData'][key,0]['epochs'][0,0][0,0]['map'][0,0]['zRaw'][0,0]
    return rmap_smooth, rmap_raw