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


class SpeedBandwidthError(OpexeboError):
    """
    Raised if the Speedscore bandwidth calculation fails
    """
