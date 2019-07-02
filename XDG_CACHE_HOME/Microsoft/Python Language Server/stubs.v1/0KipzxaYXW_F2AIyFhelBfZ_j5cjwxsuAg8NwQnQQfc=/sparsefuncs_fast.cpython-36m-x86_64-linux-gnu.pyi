import builtins as _mod_builtins

__builtins__ = {}
__doc__ = None
__file__ = '/home/claudinei/.local/lib/python3.6/site-packages/sklearn/utils/sparsefuncs_fast.cpython-36m-x86_64-linux-gnu.so'
__name__ = 'sklearn.utils.sparsefuncs_fast'
__package__ = 'sklearn.utils'
def __pyx_unpickle_Enum():
    pass

__test__ = _mod_builtins.dict()
def _csc_mean_variance_axis0(X_data, n_samples, n_features, X_indices, X_indptr):
    pass

def _csr_mean_variance_axis0(X_data, n_samples, n_features, X_indices):
    pass

def _csr_row_norms(X_data, shape, X_indices, X_indptr):
    pass

def _incr_mean_variance_axis0(X_data, n_samples, n_features, X_indices, X_indptr, X_format, last_mean, last_var, last_n):
    pass

def _inplace_csr_row_normalize_l1(X_data, shape, X_indices, X_indptr):
    pass

def _inplace_csr_row_normalize_l2(X_data, shape, X_indices, X_indptr):
    pass

def assign_rows_csr(X, X_rows, out_rows, out):
    'Densify selected rows of a CSR matrix into a preallocated array.\n\n    Like out[out_rows] = X[X_rows].toarray() but without copying.\n    No-copy supported for both dtype=np.float32 and dtype=np.float64.\n\n    Parameters\n    ----------\n    X : scipy.sparse.csr_matrix, shape=(n_samples, n_features)\n    X_rows : array, dtype=np.intp, shape=n_rows\n    out_rows : array, dtype=np.intp, shape=n_rows\n    out : array, shape=(arbitrary, n_features)\n    '
    pass

def csc_mean_variance_axis0():
    'Compute mean and variance along axis 0 on a CSC matrix\n\n    Parameters\n    ----------\n    X : CSC sparse matrix, shape (n_samples, n_features)\n        Input data.\n\n    Returns\n    -------\n    means : float array with shape (n_features,)\n        Feature-wise means\n\n    variances : float array with shape (n_features,)\n        Feature-wise variances\n\n    '
    pass

def csr_mean_variance_axis0():
    'Compute mean and variance along axis 0 on a CSR matrix\n\n    Parameters\n    ----------\n    X : CSR sparse matrix, shape (n_samples, n_features)\n        Input data.\n\n    Returns\n    -------\n    means : float array with shape (n_features,)\n        Feature-wise means\n\n    variances : float array with shape (n_features,)\n        Feature-wise variances\n\n    '
    pass

def csr_row_norms():
    'L2 norm of each row in CSR matrix X.'
    pass

def incr_mean_variance_axis0():
    'Compute mean and variance along axis 0 on a CSR or CSC matrix.\n\n    last_mean, last_var are the statistics computed at the last step by this\n    function. Both must be initialized to 0.0. last_n is the\n    number of samples encountered until now and is initialized at 0.\n\n    Parameters\n    ----------\n    X : CSR or CSC sparse matrix, shape (n_samples, n_features)\n      Input data.\n\n    last_mean : float array with shape (n_features,)\n      Array of feature-wise means to update with the new data X.\n\n    last_var : float array with shape (n_features,)\n      Array of feature-wise var to update with the new data X.\n\n    last_n : int array with shape (n_features,)\n      Number of samples seen so far, before X.\n\n    Returns\n    -------\n    updated_mean : float array with shape (n_features,)\n      Feature-wise means\n\n    updated_variance : float array with shape (n_features,)\n      Feature-wise variances\n\n    updated_n : int array with shape (n_features,)\n      Updated number of samples seen\n\n    Notes\n    -----\n    NaNs are ignored during the computation.\n\n    References\n    ----------\n    T. Chan, G. Golub, R. LeVeque. Algorithms for computing the sample\n      variance: recommendations, The American Statistician, Vol. 37, No. 3,\n      pp. 242-247\n\n    Also, see the non-sparse implementation of this in\n    `utils.extmath._batch_mean_variance_update`.\n\n    '
    pass

def inplace_csr_row_normalize_l1():
    'Inplace row normalize using the l1 norm'
    pass

def inplace_csr_row_normalize_l2():
    'Inplace row normalize using the l2 norm'
    pass

