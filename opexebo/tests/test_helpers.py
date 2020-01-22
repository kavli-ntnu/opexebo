import numpy as np

'''' Test helpers'''

test_data_square = r"N:\simoba\opexebo_working_area\test_data\generic\simple_square_input_vars.mat"
test_data_nonsquare = r"N:\simoba\opexebo_working_area\test_data\non-square\input_file_vars.mat"


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



###############################################################################
################                HELPER FUNCTIONS
###############################################################################
def generate_2d_map(shape, bin_width, **kwargs):
    ''' Create a fake 2D histogram for testing functions that process 2D images
    e.g. SpatialOccupancy, RateMap, PlaceField, GridScore etc

    Parameters
    ----------
    shape : str
        Shape - either "rect" or "circ"
    bin_width : float
        Bins must be assumed square
    kwargs
        x : float
            x-dimension of a rectangular array
        y : float
            y-dimension of a rectangular array
        d : float
            diameter of a circular array
        coverage : float
            Probability of any given location being visited by the animal
            Given in the range [0,1]
        fields : various
            Define how firing fields are generated. Options:
                Undefined - no fields
                str - "random" - [5-15] randomly generated firing fields
                list of dicts - predefined firing fields per dictionary specifications
                    For keywords, see `generate_field`
    Returns
    -------
    arena : numpy.ndarray
        2D array
    '''
    if shape == "rect":
        array_size = np.array((kwargs.get("y", 80), kwargs.get("x", 80)))
        num_bins = np.ceil(array_size / bin_width).astype(int)   
        bin_width = array_size / num_bins
        bin_centres = [np.linspace(-1*(array_size[i]-bin_width[i])/2, 
                                   (array_size[i]-bin_width[i])/2, 
                                   num_bins[i]) for i in range(2)]
        arena = np.zeros(num_bins)
    elif shape == "circ":
        diameter = kwargs.get("d", 100)
        array_size = np.array([diameter, diameter])
        num_bins = np.ceil(diameter / bin_width).astype(int)        
        arena = np.zeros((num_bins, num_bins))
        
        bin_width = diameter/num_bins
        bin_centres = (np.linspace(-(diameter-bin_width)/2, (diameter-bin_width)/2, num_bins),
                       np.linspace(-(diameter-bin_width)/2, (diameter-bin_width)/2, num_bins))
        distance = generate_distance_map(bin_centres)
        outside = distance > (diameter / 2)
        arena = np.ma.masked_where(outside, arena)
    else:
        raise NotImplementedError(f"Shape '{shape}' is not supported. Only 'rect' or 'circ' are supported")
    # Generate 1 or more firing fields
    fields = kwargs.get("fields", None)
    if fields is None:
        print("No fields argument detected")
        # do nothing at all
        pass
    elif fields == "random":
        # Randomly generate firing fields
        num_fields = np.random.randint(5, 15)
        print(f"Randomfields: {num_fields}")
        for i in range(num_fields):
            x, y = np.random.randn(2) * array_size * 0.1        # randn: [-1, 1], normal dist
            s_major, s_minor = np.random.rand(2) * array_size * 0.4   # rand: [0, 1], linear dist
            theta = np.random.rand(1) * 2 * np.pi
            amplitude = np.random.rand(1) * 20
            noise = np.random.rand(1) * 0.2
            field = generate_field(bin_centres, x, y, s_major, s_minor, theta, amplitude, noise)
            arena += field
    elif isinstance(fields, list):
        print(f"Predefined fields detected: {len(fields)}")
        # A list of dictionaries of pre-defined fields:
        for field_dictionary in fields:
            field = generate_field(bin_centres, **field_dictionary)
            arena += field
    else:
        raise NotImplementedError(f"Fields definition not understood: <{fields}>")
    #field = generate_field(bin_centres, -5, 10, 10, 5, 0, 1, 0.2)
    #arena = arena + field

    # Generate non-perfect coverage
    coverage = kwargs.get("coverage", 1)
    visiting_probability = np.random.rand(*arena.shape)
    arena = np.ma.masked_where(visiting_probability>coverage, arena)
    return arena


def generate_distance_map(bin_centres):
    '''Generate a map of distances from the centre, given arrays of the Y, X bin centres

    Parameters
    ----------
    bin_centres : list of 1d np.ndarrays
        List (y, x) of the centre locations of the bin edges

    Returns
    -------
    distances : 2d np.ndarray
        2D Numpy array of size (len(y), len(x)) where the values are distances from the centre.
    '''
    X, Y = np.meshgrid(*bin_centres)
    distance = np.sqrt(np.square(X) + np.square(Y))
    return distance


def generate_field(bin_centres, x, y, s_major, s_minor, theta, amplitude, noise):
    '''Generate a noisy firing field based on a Gaussian distribution
    
    Parameters
    ----------
    bin_centres : list of 1d numpy.ndarray
        List (y, x) of the centre locations of the bin edges
    x : float
        x co-ord of centre of Gaussian
    y : float
        y co-ord of centre of Gaussian
    s_major : float
        Standard deviation of long axis
    s_minor : float
        Standard deviation of short axis
    theta : float
        Angle of long axis w.r.t x-axis in radians
    amplitude : float
        Peak firing rate of field
    noise : float
        Noisiness of field, relative to amplitude
    
    Returns
    -------
    field : numpy.ndarray
        2D array, matching the sizes of bin edges, containing a single firing
        field.
        
    '''
    sigma_x = (np.cos(theta) * s_major) + (np.sin(theta) * s_minor)
    sigma_y = (np.sin(theta) * s_major) - (np.cos(theta) * s_minor)
    X, Y = np.meshgrid(*bin_centres)
    
    field = np.exp(-0.5 * ((np.square(X-x) / (2*sigma_x**2))
                           + (np.square(Y-y) / (2*sigma_y**2))))
    if not noise == 0:
        noise = np.random.rand(*field.shape) * noise
        noise[field<0.05] = 0
    else:
        noise = 1
    
    field = field * noise * amplitude
    return field
    
def generate_hexagonal_grid_fields():
    '''Example code to generate a field definition list for a hexagonal grid of
    cells. Mimic a perfect grid cell.'''
    sma = 5
    smi = 5
    amp = 1
    fields = []
    offset = 20  
    for y_step in np.arange(-3, 4, 1):
        y = (y_step * offset * 0.866)
        for x_step in np.arange(-3, 4, 1):
            x = (x_step * offset) + (y_step%2 * 0.5 * offset)
            field = {"x":x, "y":y, "s_major":sma, "s_minor":smi, "theta": 0, "amplitude":amp, "noise":0}
            fields.append(field)
    return fields
