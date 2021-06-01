# opexebo

[![Build Status](https://travis-ci.com/kavli-ntnu/opexebo.svg?branch=master&status=passed)](https://travis-ci.com/kavli-ntnu/opexebo)
[![codecov](https://codecov.io/gh/kavli-ntnu/opexebo/branch/master/graph/badge.svg)](https://codecov.io/gh/kavli-ntnu/opexebo)
[![Documentation Status](https://readthedocs.org/projects/opexebo/badge/?version=latest)](https://opexebo.readthedocs.io/en/latest/?badge=latest)

This library provides various analysis tools written by and for the Moser group  at the [Kavli Institute](https://www.ntnu.edu/kavli) in Trondheim. 

The package contains a mixture of translations of the older MatLab based [Behavioural Neurology Toolbox](https://bitbucket.org/cnc-ntnu/bnt), and other functions written only in Python 3, as part of the transition towards providing open access to research code. 


### Documentation

Specifications and documentation for `opexebo` are available on [Read the Docs](https://opexebo.readthedocs.io/en/latest/).


### Installation

Install by using pip:

```
pip install git+https://github.com/kavli-ntnu/opexebo.git -U
```

A specific revision, branch, tag, or release an be installed with the @ modifier:

```
pip install git+https://github.com/kavli-ntnu/opexebo.git@v0.5.2
```

Installing in this way requires having `git`, this can be acquired from::

    https://git-scm.com/download

Opexebo has an optional dependency, `sep`, that is not installed by default. To be able to install it, you require a C++ compiler installed on your system. On Linux, `gcc` will do the job. On Windows, the the Microsoft Visual C++ Build Tools fulfil the same role (https://www.microsoft.com/en-us/download/details.aspx?id=48159). To force installation of all optional dependencies, append `#egg[full]` to the install command, for example::

    pip install git+https://github.com/kavli-ntnu/opexebo.git@v0.5.2#egg[full]
