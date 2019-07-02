import builtins as _mod_builtins
import scipy.sparse.csr as _mod_scipy_sparse_csr

__builtins__ = {}
__doc__ = None
__file__ = '/home/claudinei/.local/lib/python3.6/site-packages/sklearn/preprocessing/_csr_polynomial_expansion.cpython-36m-x86_64-linux-gnu.so'
__name__ = 'sklearn.preprocessing._csr_polynomial_expansion'
__package__ = 'sklearn.preprocessing'
def __pyx_unpickle_Enum():
    pass

__test__ = _mod_builtins.dict()
def _csr_polynomial_expansion(data, indices, indptr, d, interaction_only, degree):
    '\n    Perform a second-degree polynomial or interaction expansion on a scipy\n    compressed sparse row (CSR) matrix. The method used only takes products of\n    non-zero features. For a matrix with density d, this results in a speedup\n    on the order of d^k where k is the degree of the expansion, assuming all\n    rows are of similar density.\n\n    Parameters\n    ----------\n    data : nd-array\n        The "data" attribute of the input CSR matrix.\n\n    indices : nd-array\n        The "indices" attribute of the input CSR matrix.\n\n    indptr : nd-array\n        The "indptr" attribute of the input CSR matrix.\n\n    d : int\n        The dimensionality of the input CSR matrix.\n\n    interaction_only : int\n        0 for a polynomial expansion, 1 for an interaction expansion.\n\n    degree : int\n        The degree of the expansion. This must be either 2 or 3.\n\n    References\n    ----------\n    "Leveraging Sparsity to Speed Up Polynomial Feature Expansions of CSR\n    Matrices Using K-Simplex Numbers" by Andrew Nystrom and John Hughes.\n    '
    pass

csr_matrix = _mod_scipy_sparse_csr.csr_matrix
