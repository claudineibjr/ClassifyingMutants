import builtins as _mod_builtins

__builtins__ = {}
__doc__ = None
__file__ = '/home/claudinei/.local/lib/python3.6/site-packages/sklearn/cluster/_k_means_elkan.cpython-36m-x86_64-linux-gnu.so'
__name__ = 'sklearn.cluster._k_means_elkan'
__package__ = 'sklearn.cluster'
def __pyx_unpickle_Enum():
    pass

__test__ = _mod_builtins.dict()
def _centers_dense(X, sample_weight, labels, n_clusters, distances):
    'M step of the K-means EM algorithm\n\n    Computation of cluster centers / means.\n\n    Parameters\n    ----------\n    X : array-like, shape (n_samples, n_features)\n\n    sample_weight : array-like, shape (n_samples,)\n        The weights for each observation in X.\n\n    labels : array of integers, shape (n_samples)\n        Current label assignment\n\n    n_clusters : int\n        Number of desired clusters\n\n    distances : array-like, shape (n_samples)\n        Distance to closest cluster for each sample.\n\n    Returns\n    -------\n    centers : array, shape (n_clusters, n_features)\n        The resulting centers\n    '
    pass

def euclidean_distances(X, Y, Y_norm_squared, squared, X_norm_squared):
    '\n    Considering the rows of X (and Y=X) as vectors, compute the\n    distance matrix between each pair of vectors.\n\n    For efficiency reasons, the euclidean distance between a pair of row\n    vector x and y is computed as::\n\n        dist(x, y) = sqrt(dot(x, x) - 2 * dot(x, y) + dot(y, y))\n\n    This formulation has two advantages over other ways of computing distances.\n    First, it is computationally efficient when dealing with sparse data.\n    Second, if one argument varies but the other remains unchanged, then\n    `dot(x, x)` and/or `dot(y, y)` can be pre-computed.\n\n    However, this is not the most precise way of doing this computation, and\n    the distance matrix returned by this function may not be exactly\n    symmetric as required by, e.g., ``scipy.spatial.distance`` functions.\n\n    Read more in the :ref:`User Guide <metrics>`.\n\n    Parameters\n    ----------\n    X : {array-like, sparse matrix}, shape (n_samples_1, n_features)\n\n    Y : {array-like, sparse matrix}, shape (n_samples_2, n_features)\n\n    Y_norm_squared : array-like, shape (n_samples_2, ), optional\n        Pre-computed dot-products of vectors in Y (e.g.,\n        ``(Y**2).sum(axis=1)``)\n        May be ignored in some cases, see the note below.\n\n    squared : boolean, optional\n        Return squared Euclidean distances.\n\n    X_norm_squared : array-like, shape = [n_samples_1], optional\n        Pre-computed dot-products of vectors in X (e.g.,\n        ``(X**2).sum(axis=1)``)\n        May be ignored in some cases, see the note below.\n\n    Notes\n    -----\n    To achieve better accuracy, `X_norm_squared`\xa0and `Y_norm_squared` may be\n    unused if they are passed as ``float32``.\n\n    Returns\n    -------\n    distances : array, shape (n_samples_1, n_samples_2)\n\n    Examples\n    --------\n    >>> from sklearn.metrics.pairwise import euclidean_distances\n    >>> X = [[0, 1], [1, 1]]\n    >>> # distance between rows of X\n    >>> euclidean_distances(X, X)\n    array([[0., 1.],\n           [1., 0.]])\n    >>> # get distance to origin\n    >>> euclidean_distances(X, [[0, 0]])\n    array([[1.        ],\n           [1.41421356]])\n\n    See also\n    --------\n    paired_distances : distances betweens pairs of elements of X and Y.\n    '
    pass

def k_means_elkan():
    "Run Elkan's k-means.\n\n    Parameters\n    ----------\n    X_ : nd-array, shape (n_samples, n_features)\n\n    sample_weight : nd-array, shape (n_samples,)\n        The weights for each observation in X.\n\n    n_clusters : int\n        Number of clusters to find.\n\n    init : nd-array, shape (n_clusters, n_features)\n        Initial position of centers.\n\n    tol : float, default=1e-4\n        The relative increment in cluster means before declaring convergence.\n\n    max_iter : int, default=30\n    Maximum number of iterations of the k-means algorithm.\n\n    verbose : bool, default=False\n        Whether to be verbose.\n\n    "
    pass

