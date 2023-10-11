import os
import platform

"""Astropy insists on having a local temporary directory, and on on managed
NTNU computers, the default search location happens to be a non-responsive
network location with an excessively long (~1 minute) timeout. Therefore, fix
it to something harmless in advance"""
if "Windows" in platform.platform():
    os.environ["HOMESHARE"] = r"C:\temp\astropy"
elif "Linux" in platform.platform():
    os.environ["HOMESHARE"] = r"/tmp/astropy"
elif "macos" in platform.platform():
    os.environ["HOMESHARE"] = r"/tmp/astropy"


from .test_helpers import *
from . import test_analysis as analysis
from . import test_general as general
