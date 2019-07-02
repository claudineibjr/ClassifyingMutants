import builtins as _mod_builtins

__builtins__ = {}
__doc__ = None
__file__ = '/home/claudinei/.local/lib/python3.6/site-packages/sklearn/cluster/_k_means.cpython-36m-x86_64-linux-gnu.so'
__name__ = 'sklearn.cluster._k_means'
__package__ = 'sklearn.cluster'
def __pyx_unpickle_Enum():
    pass

__test__ = _mod_builtins.dict()
def _assign_labels_array(X, sample_weight, x_squared_norms, centers, labels, distances):
    'Compute label assignment and inertia for a dense array\n\n    Return the inertia (sum of squared distances to the centers).\n    '
    pass

def _assign_labels_csr(X, sample_weight, x_squared_norms, centers, labels, distances):
    'Compute label assignment and inertia for a CSR input\n\n    Return the inertia (sum of squared distances to the centers).\n    '
    pass

def _centers_dense(X, sample_weight, labels, n_clusters, distances):
    'M step of the K-means EM algorithm\n\n    Computation of cluster centers / means.\n\n    Parameters\n    ----------\n    X : array-like, shape (n_samples, n_features)\n\n    sample_weight : array-like, shape (n_samples,)\n        The weights for each observation in X.\n\n    labels : array of integers, shape (n_samples)\n        Current label assignment\n\n    n_clusters : int\n        Number of desired clusters\n\n    distances : array-like, shape (n_samples)\n        Distance to closest cluster for each sample.\n\n    Returns\n    -------\n    centers : array, shape (n_clusters, n_features)\n        The resulting centers\n    '
    pass

def _centers_sparse(X, sample_weight, labels, n_clusters, distances):
    'M step of the K-means EM algorithm\n\n    Computation of cluster centers / means.\n\n    Parameters\n    ----------\n    X : scipy.sparse.csr_matrix, shape (n_samples, n_features)\n\n    sample_weight : array-like, shape (n_samples,)\n        The weights for each observation in X.\n\n    labels : array of integers, shape (n_samples)\n        Current label assignment\n\n    n_clusters : int\n        Number of desired clusters\n\n    distances : array-like, shape (n_samples)\n        Distance to closest cluster for each sample.\n\n    Returns\n    -------\n    centers : array, shape (n_clusters, n_features)\n        The resulting centers\n    '
    pass

def _mini_batch_update_csr(X, sample_weight, x_squared_norms, centers, weight_sums, nearest_center, old_center, compute_squared_diff):
    'Incremental update of the centers for sparse MiniBatchKMeans.\n\n    Parameters\n    ----------\n\n    X : CSR matrix, dtype float\n        The complete (pre allocated) training set as a CSR matrix.\n\n    centers : array, shape (n_clusters, n_features)\n        The cluster centers\n\n    counts : array, shape (n_clusters,)\n         The vector in which we keep track of the numbers of elements in a\n         cluster\n\n    Returns\n    -------\n    inertia : float\n        The inertia of the batch prior to centers update, i.e. the sum\n        of squared distances to the closest center for each sample. This\n        is the objective function being minimized by the k-means algorithm.\n\n    squared_diff : float\n        The sum of squared update (squared norm of the centers position\n        change). If compute_squared_diff is 0, this computation is skipped and\n        0.0 is returned instead.\n\n    Both squared diff and inertia are commonly used to monitor the convergence\n    of the algorithm.\n    '
    pass

def assign_rows_csr(X, X_rows, out_rows, out):
    'Densify selected rows of a CSR matrix into a preallocated array.\n\n    Like out[out_rows] = X[X_rows].toarray() but without copying.\n    No-copy supported for both dtype=np.float32 and dtype=np.float64.\n\n    Parameters\n    ----------\n    X : scipy.sparse.csr_matrix, shape=(n_samples, n_features)\n    X_rows : array, dtype=np.intp, shape=n_rows\n    out_rows : array, dtype=np.intp, shape=n_rows\n    out : array, shape=(arbitrary, n_features)\n    '
    pass

