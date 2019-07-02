import builtins as _mod_builtins
import sklearn.exceptions as _mod_sklearn_exceptions

ConvergenceWarning = _mod_sklearn_exceptions.ConvergenceWarning
__builtins__ = {}
__doc__ = None
__file__ = '/home/claudinei/.local/lib/python3.6/site-packages/sklearn/svm/libsvm_sparse.cpython-36m-x86_64-linux-gnu.so'
__name__ = 'sklearn.svm.libsvm_sparse'
__package__ = 'sklearn.svm'
__test__ = _mod_builtins.dict()
def libsvm_sparse_decision_function():
    '\n    Predict margin (libsvm name for this is predict_values)\n\n    We have to reconstruct model and parameters to make sure we stay\n    in sync with the python object.\n    '
    pass

def libsvm_sparse_predict():
    '\n    Predict values T given a model.\n\n    For speed, all real work is done at the C level in function\n    copy_predict (libsvm_helper.c).\n\n    We have to reconstruct model and parameters to make sure we stay\n    in sync with the python object.\n\n    See sklearn.svm.predict for a complete list of parameters.\n\n    Parameters\n    ----------\n    X : array-like, dtype=float\n    Y : array\n        target vector\n\n    Returns\n    -------\n    dec_values : array\n        predicted values.\n    '
    pass

def libsvm_sparse_predict_proba():
    '\n    Predict values T given a model.\n    '
    pass

def libsvm_sparse_train():
    '\n    Wrap svm_train from libsvm using a scipy.sparse.csr matrix\n\n    Work in progress.\n\n    Parameters\n    ----------\n    n_features : number of features.\n        XXX: can we retrieve this from any other parameter ?\n\n    X : array-like, dtype=float, size=[N, D]\n\n    Y : array, dtype=float, size=[N]\n        target vector\n\n    ...\n\n    Notes\n    -------------------\n    See sklearn.svm.predict for a complete list of parameters.\n\n    '
    pass

def set_verbosity_wrap():
    '\n    Control verbosity of libsvm library\n    '
    pass

