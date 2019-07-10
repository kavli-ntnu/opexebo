=======
opexebo
=======

Collection of python code in `Kavli lab <https://www.ntnu.edu/kavli>`_.

Installation
============

Install by using pip::

    pip install git+https://github.com/kavli-ntnu/opexebo.git#egg=opexebo

A specific revision, branch, tag, or release an be installed by targetting the git command:

    https://pip.pypa.io/en/stable/reference/pip_install/#vcs-support

Installing in this way requires having `git`, this can be acquired from

    https://git-scm.com/download/win

Installing the required python package `sep` requires the Microsoft Visual C++ Build Tools, these can be downloaded from

    https://www.microsoft.com/en-us/download/details.aspx?id=48159


Usage example
=============

.. highlight:: python

Some data (generated in Matlab) is available in ``sample_data/mouse_grids.mat``.
The file contains cell arrays ``aCorrs``, ``rateMaps`` and vectors ``gridScores``,
``gridScoresNew``. ``gridScoresNew`` is a version of gridness score as of BNT v1.0::

    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd

    from scipy.io import loadmat

    import opexebo

    matFile = ''  # put the path to mouse_grids.mat file here
    mouse_grids = loadmat(matFile, squeeze_me=True, struct_as_record=False)
    rate_maps = mouse_grids['rateMaps']
    acorrs = mouse_grids['aCorrs']
    bnt_grid_scores = mouse_grids['gridScoresNew']
    num_cells = len(acorrs)

    columns = ['Cell #', 'BNT score', 'Python score']
    df = pd.DataFrame(index=range(0, num_cells), columns=columns)
    for i in range(0, num_cells):
        print("Processing cell {}".format(i))
        map = rate_maps[i]
        aCorr = opexebo.analysis.autocorrelation(map)
        gs = opexebo.analysis.gridscore(aCorr)

        df['Cell #'][i] = i+1
        df['BNT score'][i] = bnt_grid_scores[i]
        df['Python score'][i] = gs

    same_scores = np.abs(df['BNT score'] - df['Python score']) < 0.001
    # see indices of elements that have different scores
    np.where(~same_scores)

    map = rate_maps[0]
    plt.imshow(map, cmap='jet', origin='lower')
    acorr = opexebo.analysis.autocorrelation(map)
    plt.imshow(acorr, cmap='jet', origin='lower')
