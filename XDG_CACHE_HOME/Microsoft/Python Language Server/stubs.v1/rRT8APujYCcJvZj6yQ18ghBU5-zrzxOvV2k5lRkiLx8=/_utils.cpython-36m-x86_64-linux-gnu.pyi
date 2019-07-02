import builtins as _mod_builtins

__builtins__ = {}
__doc__ = None
__file__ = '/home/claudinei/.local/lib/python3.6/site-packages/sklearn/manifold/_utils.cpython-36m-x86_64-linux-gnu.so'
__name__ = 'sklearn.manifold._utils'
__package__ = 'sklearn.manifold'
__test__ = _mod_builtins.dict()
def _binary_search_perplexity():
    "Binary search for sigmas of conditional Gaussians.\n\n    This approximation reduces the computational complexity from O(N^2) to\n    O(uN). See the exact method '_binary_search_perplexity' for more details.\n\n    Parameters\n    ----------\n    affinities : array-like, shape (n_samples, k)\n        Distances between training samples and its k nearest neighbors.\n\n    neighbors : array-like, shape (n_samples, k) or None\n        Each row contains the indices to the k nearest neigbors. If this\n        array is None, then the perplexity is estimated over all data\n        not just the nearest neighbors.\n\n    desired_perplexity : float\n        Desired perplexity (2^entropy) of the conditional Gaussians.\n\n    verbose : int\n        Verbosity level.\n\n    Returns\n    -------\n    P : array, shape (n_samples, n_samples)\n        Probabilities of conditional Gaussian distributions p_i|j.\n    "
    pass

