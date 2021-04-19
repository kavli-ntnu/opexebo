'''
Exception classes for the Opexebo library
'''

# ----- Level 1 -----
class OpexeboError(Exception):
    '''
    Base class for exceptions specific to Opexebo
    '''


# ----- Level 2 -----
class ArgumentError(OpexeboError):
    '''
    Incorrect argument given in some way
    '''

class DimensionMismatchError(OpexeboError):
    """
    Raised if 1D and 2D data is mixed
    """

class SpeedBandwidthError(OpexeboError):
    """
    Raised if the Speedscore bandwidth calculation fails
    """
