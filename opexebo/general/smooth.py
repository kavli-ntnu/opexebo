""" Provides a function for Gaussian smoothing """
import numpy as np
from astropy.convolution import convolve, Gaussian2DKernel, Gaussian1DKernel
#http://docs.astropy.org/en/stable/convolution/index.html

#Astropy appears to have an import problem - it routinely takes about 3 minutes 
#to import, apparently because it is looking for a file with a stupidly long timeout.
#It typically results in a warning: 
#        ConfigurationMissingWarning : Configuration defaults will be used due to FileNotFoundError:2 on None

#The root cause seems to be that Astropy expects to find a configuration file in
#a very specific location - in thise case, a network location (\\home.ansatt.ntnu.no\.astropy)
#Which doesn't exist. But the timeout for checking the network is very long

# A tempoary workaround is to crate the environment variables 
#    XDG_CACHE_HOME
#    XDG_CONFIG_HOME 
# pointing to local locations that do exist. Since we only use a very minor part
# of Astropy here, the fact that the config file may be ephermeral is not a problem
# See here for discussion of these variables
# https://github.com/astropy/astropy/issues/6511


def smooth(data, sigma):
    '''Smooth provided data with a Gaussian kernel 
    
    The smoothing is done with a routine from the astronomical package astropy
    Like scipy.ndimage, this package DOES NOT RESPECT MASKED ARRAYS. However, it
    replaces NaN values with an interpolation of nearby cells.
    
    Therefore, to handle masked arrays, the data at masked positions **is replaced by np.nan**
    prior to smoothing, and thus avoids influencing nearby, unmasked cells.
    The masked cells are then returned to their original values prior to return.
    
    The package is discussed at http://docs.astropy.org/en/stable/convolution/index.html
    
    Parameters:
    ----------
    data : np.ndarray or np.ma.MaskedArray
        Data that will be smoothed
    sigma : int
        Standard deviations for Gaussian kernel in units of pixels
    
    Returns
    -------
    smoothed_data : np.ndarray or np.ma.MaskedArray
        Smoothed data
    '''
    d = data.ndim
    if  d== 2:
        kernel = Gaussian2DKernel(x_stddev=sigma)
    elif d == 1:
        kernel = Gaussian1DKernel(stddev=sigma)
    else:
        raise ValueError("This function can only smooth 1D or 2D data. You provided data with %d dimensions" % d)
    
    working_data = data.copy()
    # This is in case the input data has to be modified
    # Since Python passes references instead of data, we create a new copy to work
    # with, rather than modifying the state of data that other functions may rely on
    
    if type(data) == np.ma.MaskedArray:
        
        working_data[data.mask] = np.nan
        
    smoothed_data = convolve(working_data, kernel)
    
    
    if type(data) == np.ma.MaskedArray:
#        data.data[data_copy.mask] = data_copy[data_copy.mask]
        smoothed_data = np.ma.masked_where(data.mask, smoothed_data)
        smoothed_data.data[data.mask] = data.data[data.mask]
        
    return smoothed_data


if __name__ == '__main__':
    import matplotlib.pyplot as plt
    a = np.nan
    test_data = [[0,0,0,0,5,3,0,4],
                 [0,0,0,0,1,1,2,4],
                 [0,0,3,6,6,4,4,0],
                 [0,5,2,5,a,4,2,0],
                 [0,0,1,a,a,2,0,0],
                 [0,0,a,a,a,2,6,0],
                 [0,0,3,5,6,6,6,6]]
    test_data = np.ma.MaskedArray(test_data)
    test_data.mask = np.zeros((8,7),dtype=bool)
    
    
    
    fig, (ax1, ax2) = plt.subplots(1,2)
    ax1.imshow(test_data)
    test_data.mask[5,2] = True
    test_data.mask[3,5] = True
    ax2.imshow(smooth(test_data, 1))
    plt.show()
    
    td2 = np.arange(20)
    std2 = smooth(td2, 2)
    plt.figure()
    plt.plot(td2)
    plt.plot(std2)
    plt.show()