=======
opexebo
=======

Collection of python code in `Kavli lab <https://www.ntnu.edu/kavli>`_.

Installation
============

Install by using pip::

    pip install git+https://github.com/kavli-ntnu/opexebo.git -U

A specific revision, branch, tag, or release an be installed with the @ modifier::

    pip install git+https://github.com/kavli-ntnu/opexebo.git@v0.3.3

Installing in this way requires having `git`, this can be acquired from::

    https://git-scm.com/download/win

Opexebo has an optional dependency, `sep`, that is not installed by default. To be able to install it, you require a C++ compiler installed on your system. On Linux, `gcc` will do the job. On Windows, the the Microsoft Visual C++ Build Tools fulfil the same role (https://www.microsoft.com/en-us/download/details.aspx?id=48159). To force installation of all optional depencies, append `#egg[full]` to the install command, for example::

    pip install git+https://github.com/kavli-ntnu/opexebo.git@v0.3.3#egg[full]




    

