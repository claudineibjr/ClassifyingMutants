import builtins as _mod_builtins

__builtins__ = {}
__doc__ = None
__file__ = '/home/claudinei/.local/lib/python3.6/site-packages/sklearn/decomposition/_online_lda.cpython-36m-x86_64-linux-gnu.so'
__name__ = 'sklearn.decomposition._online_lda'
__package__ = 'sklearn.decomposition'
__test__ = _mod_builtins.dict()
def _dirichlet_expectation_1d():
    'Dirichlet expectation for a single sample:\n        exp(E[log(theta)]) for theta ~ Dir(doc_topic)\n    after adding doc_topic_prior to doc_topic, in-place.\n\n    Equivalent to\n        doc_topic += doc_topic_prior\n        out[:] = np.exp(psi(doc_topic) - psi(np.sum(doc_topic)))\n    '
    pass

def _dirichlet_expectation_2d():
    "Dirichlet expectation for multiple samples:\n    E[log(theta)] for theta ~ Dir(arr).\n\n    Equivalent to psi(arr) - psi(np.sum(arr, axis=1))[:, np.newaxis].\n\n    Note that unlike _dirichlet_expectation_1d, this function doesn't compute\n    the exp and doesn't add in the prior.\n    "
    pass

def mean_change():
    'Calculate the mean difference between two arrays.\n\n    Equivalent to np.abs(arr_1 - arr2).mean().\n    '
    pass

