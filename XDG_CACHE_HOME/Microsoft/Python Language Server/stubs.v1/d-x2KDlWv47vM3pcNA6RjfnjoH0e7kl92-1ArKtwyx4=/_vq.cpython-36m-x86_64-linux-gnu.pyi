import builtins as _mod_builtins

__builtins__ = {}
__doc__ = '\nCython rewrite of the vector quantization module, originally written\nin C at src/vq.c and the wrapper at src/vq_module.c. This should be\neasier to maintain than old SWIG output.\n\nOriginal C version by Damian Eads.\nTranslated to Cython by David Warde-Farley, October 2009.\n'
__file__ = '/home/claudinei/.local/lib/python3.6/site-packages/scipy/cluster/_vq.cpython-36m-x86_64-linux-gnu.so'
__name__ = 'scipy.cluster._vq'
__package__ = 'scipy.cluster'
__test__ = _mod_builtins.dict()
def update_cluster_means():
    '\n    The update-step of K-means. Calculate the mean of observations in each\n    cluster.\n\n    Parameters\n    ----------\n    obs : ndarray\n        The observation matrix. Each row is an observation. Its dtype must be\n        float32 or float64.\n    labels : ndarray\n        The label of each observation. Must be an 1d array.\n    nc : int\n        The number of centroids.\n\n    Returns\n    -------\n    cb : ndarray\n        The new code book.\n    has_members : ndarray\n        A boolean array indicating which clusters have members.\n\n    Notes\n    -----\n    The empty clusters will be set to all zeros and the curresponding elements\n    in `has_members` will be `False`. The upper level function should decide\n    how to deal with them.\n    '
    pass

def vq():
    '\n    Vector quantization ndarray wrapper. Only support float32 and float64.\n\n    Parameters\n    ----------\n    obs : ndarray\n        The observation matrix. Each row is an observation.\n    codes : ndarray\n        The code book matrix.\n\n    Notes\n    -----\n    The observation matrix and code book matrix should have same ndim and\n    same number of columns (features). Only 1-dimensional and 2-dimensional\n    arrays are supported.\n    '
    pass

