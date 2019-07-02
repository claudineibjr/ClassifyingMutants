import builtins as _mod_builtins
import sklearn.exceptions as _mod_sklearn_exceptions

ConvergenceWarning = _mod_sklearn_exceptions.ConvergenceWarning
__builtins__ = {}
__doc__ = None
__file__ = '/home/claudinei/.local/lib/python3.6/site-packages/sklearn/linear_model/cd_fast.cpython-36m-x86_64-linux-gnu.so'
__name__ = 'sklearn.linear_model.cd_fast'
__package__ = 'sklearn.linear_model'
def __pyx_unpickle_Enum():
    pass

__test__ = _mod_builtins.dict()
def enet_coordinate_descent():
    'Cython version of the coordinate descent algorithm\n        for Elastic-Net regression\n\n        We minimize\n\n        (1/2) * norm(y - X w, 2)^2 + alpha norm(w, 1) + (beta/2) norm(w, 2)^2\n\n    '
    pass

def enet_coordinate_descent_gram():
    'Cython version of the coordinate descent algorithm\n        for Elastic-Net regression\n\n        We minimize\n\n        (1/2) * w^T Q w - q^T w + alpha norm(w, 1) + (beta/2) * norm(w, 2)^2\n\n        which amount to the Elastic-Net problem when:\n        Q = X^T X (Gram matrix)\n        q = X^T y\n    '
    pass

def enet_coordinate_descent_multi_task():
    'Cython version of the coordinate descent algorithm\n        for Elastic-Net mult-task regression\n\n        We minimize\n\n        (1/2) * norm(y - X w, 2)^2 + l1_reg ||w||_21 + (1/2) * l2_reg norm(w, 2)^2\n\n    '
    pass

def sparse_enet_coordinate_descent():
    'Cython version of the coordinate descent algorithm for Elastic-Net\n\n    We minimize:\n\n        (1/2) * norm(y - X w, 2)^2 + alpha norm(w, 1) + (beta/2) * norm(w, 2)^2\n\n    '
    pass

