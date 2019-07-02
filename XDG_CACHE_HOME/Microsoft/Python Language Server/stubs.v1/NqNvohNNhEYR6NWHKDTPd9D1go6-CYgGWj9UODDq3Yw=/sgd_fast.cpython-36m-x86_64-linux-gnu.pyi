import builtins as _mod_builtins

class Classification(LossFunction):
    'Base class for loss functions for classification'
    __class__ = Classification
    def __init__(self, *args, **kwargs):
        'Base class for loss functions for classification'
        pass
    
    @classmethod
    def __init_subclass__(cls):
        'This method is called when a class is subclassed.\n\nThe default implementation does nothing. It may be\noverridden to extend subclasses.\n'
        return None
    
    __pyx_vtable__ = _mod_builtins.PyCapsule()
    def __reduce__(self):
        return ''; return ()
    
    def __setstate__(self, state):
        return None
    
    @classmethod
    def __subclasshook__(cls, subclass):
        'Abstract classes can override this to customize issubclass().\n\nThis is invoked early on by abc.ABCMeta.__subclasscheck__().\nIt should return True, False or NotImplemented.  If it returns\nNotImplemented, the normal algorithm is used.  Otherwise, it\noverrides the normal algorithm (and the outcome is cached).\n'
        return False
    

class EpsilonInsensitive(Regression):
    'Epsilon-Insensitive loss (used by SVR).\n\n    loss = max(0, |y - p| - epsilon)\n    '
    __class__ = EpsilonInsensitive
    def __init__(self, usedbySVR):
        'Epsilon-Insensitive loss (used by SVR).\n\n    loss = max(0, |y - p| - epsilon)\n    '
        pass
    
    @classmethod
    def __init_subclass__(cls):
        'This method is called when a class is subclassed.\n\nThe default implementation does nothing. It may be\noverridden to extend subclasses.\n'
        return None
    
    __pyx_vtable__ = _mod_builtins.PyCapsule()
    def __reduce__(self):
        return ''; return ()
    
    @classmethod
    def __subclasshook__(cls, subclass):
        'Abstract classes can override this to customize issubclass().\n\nThis is invoked early on by abc.ABCMeta.__subclasscheck__().\nIt should return True, False or NotImplemented.  If it returns\nNotImplemented, the normal algorithm is used.  Otherwise, it\noverrides the normal algorithm (and the outcome is cached).\n'
        return False
    

class Hinge(Classification):
    'Hinge loss for binary classification tasks with y in {-1,1}\n\n    Parameters\n    ----------\n\n    threshold : float > 0.0\n        Margin threshold. When threshold=1.0, one gets the loss used by SVM.\n        When threshold=0.0, one gets the loss used by the Perceptron.\n    '
    __class__ = Hinge
    def __init__(self, *args, **kwargs):
        'Hinge loss for binary classification tasks with y in {-1,1}\n\n    Parameters\n    ----------\n\n    threshold : float > 0.0\n        Margin threshold. When threshold=1.0, one gets the loss used by SVM.\n        When threshold=0.0, one gets the loss used by the Perceptron.\n    '
        pass
    
    @classmethod
    def __init_subclass__(cls):
        'This method is called when a class is subclassed.\n\nThe default implementation does nothing. It may be\noverridden to extend subclasses.\n'
        return None
    
    __pyx_vtable__ = _mod_builtins.PyCapsule()
    def __reduce__(self):
        return ''; return ()
    
    @classmethod
    def __subclasshook__(cls, subclass):
        'Abstract classes can override this to customize issubclass().\n\nThis is invoked early on by abc.ABCMeta.__subclasscheck__().\nIt should return True, False or NotImplemented.  If it returns\nNotImplemented, the normal algorithm is used.  Otherwise, it\noverrides the normal algorithm (and the outcome is cached).\n'
        return False
    

class Huber(Regression):
    'Huber regression loss\n\n    Variant of the SquaredLoss that is robust to outliers (quadratic near zero,\n    linear in for large errors).\n\n    https://en.wikipedia.org/wiki/Huber_Loss_Function\n    '
    __class__ = Huber
    def __init__(self, *args, **kwargs):
        'Huber regression loss\n\n    Variant of the SquaredLoss that is robust to outliers (quadratic near zero,\n    linear in for large errors).\n\n    https://en.wikipedia.org/wiki/Huber_Loss_Function\n    '
        pass
    
    @classmethod
    def __init_subclass__(cls):
        'This method is called when a class is subclassed.\n\nThe default implementation does nothing. It may be\noverridden to extend subclasses.\n'
        return None
    
    __pyx_vtable__ = _mod_builtins.PyCapsule()
    def __reduce__(self):
        return ''; return ()
    
    @classmethod
    def __subclasshook__(cls, subclass):
        'Abstract classes can override this to customize issubclass().\n\nThis is invoked early on by abc.ABCMeta.__subclasscheck__().\nIt should return True, False or NotImplemented.  If it returns\nNotImplemented, the normal algorithm is used.  Otherwise, it\noverrides the normal algorithm (and the outcome is cached).\n'
        return False
    

class Log(Classification):
    'Logistic regression loss for binary classification with y in {-1, 1}'
    __class__ = Log
    def __init__(self, *args, **kwargs):
        'Logistic regression loss for binary classification with y in {-1, 1}'
        pass
    
    @classmethod
    def __init_subclass__(cls):
        'This method is called when a class is subclassed.\n\nThe default implementation does nothing. It may be\noverridden to extend subclasses.\n'
        return None
    
    __pyx_vtable__ = _mod_builtins.PyCapsule()
    def __reduce__(self):
        return ''; return ()
    
    @classmethod
    def __subclasshook__(cls, subclass):
        'Abstract classes can override this to customize issubclass().\n\nThis is invoked early on by abc.ABCMeta.__subclasscheck__().\nIt should return True, False or NotImplemented.  If it returns\nNotImplemented, the normal algorithm is used.  Otherwise, it\noverrides the normal algorithm (and the outcome is cached).\n'
        return False
    

class LossFunction(_mod_builtins.object):
    'Base class for convex loss functions'
    __class__ = LossFunction
    def __init__(self, *args, **kwargs):
        'Base class for convex loss functions'
        pass
    
    @classmethod
    def __init_subclass__(cls):
        'This method is called when a class is subclassed.\n\nThe default implementation does nothing. It may be\noverridden to extend subclasses.\n'
        return None
    
    __pyx_vtable__ = _mod_builtins.PyCapsule()
    def __reduce__(self):
        return ''; return ()
    
    def __setstate__(self, state):
        return None
    
    @classmethod
    def __subclasshook__(cls, subclass):
        'Abstract classes can override this to customize issubclass().\n\nThis is invoked early on by abc.ABCMeta.__subclasscheck__().\nIt should return True, False or NotImplemented.  If it returns\nNotImplemented, the normal algorithm is used.  Otherwise, it\noverrides the normal algorithm (and the outcome is cached).\n'
        return False
    
    def dloss(self):
        'Evaluate the derivative of the loss function with respect to\n        the prediction `p`.\n\n        Parameters\n        ----------\n        p : double\n            The prediction, p = w^T x\n        y : double\n            The true value (aka target)\n        Returns\n        -------\n        double\n            The derivative of the loss function with regards to `p`.\n        '
        pass
    

class ModifiedHuber(Classification):
    "Modified Huber loss for binary classification with y in {-1, 1}\n\n    This is equivalent to quadratically smoothed SVM with gamma = 2.\n\n    See T. Zhang 'Solving Large Scale Linear Prediction Problems Using\n    Stochastic Gradient Descent', ICML'04.\n    "
    __class__ = ModifiedHuber
    def __init__(self, *args, **kwargs):
        "Modified Huber loss for binary classification with y in {-1, 1}\n\n    This is equivalent to quadratically smoothed SVM with gamma = 2.\n\n    See T. Zhang 'Solving Large Scale Linear Prediction Problems Using\n    Stochastic Gradient Descent', ICML'04.\n    "
        pass
    
    @classmethod
    def __init_subclass__(cls):
        'This method is called when a class is subclassed.\n\nThe default implementation does nothing. It may be\noverridden to extend subclasses.\n'
        return None
    
    __pyx_vtable__ = _mod_builtins.PyCapsule()
    def __reduce__(self):
        return ''; return ()
    
    @classmethod
    def __subclasshook__(cls, subclass):
        'Abstract classes can override this to customize issubclass().\n\nThis is invoked early on by abc.ABCMeta.__subclasscheck__().\nIt should return True, False or NotImplemented.  If it returns\nNotImplemented, the normal algorithm is used.  Otherwise, it\noverrides the normal algorithm (and the outcome is cached).\n'
        return False
    

class Regression(LossFunction):
    'Base class for loss functions for regression'
    __class__ = Regression
    def __init__(self, *args, **kwargs):
        'Base class for loss functions for regression'
        pass
    
    @classmethod
    def __init_subclass__(cls):
        'This method is called when a class is subclassed.\n\nThe default implementation does nothing. It may be\noverridden to extend subclasses.\n'
        return None
    
    __pyx_vtable__ = _mod_builtins.PyCapsule()
    def __reduce__(self):
        return ''; return ()
    
    def __setstate__(self, state):
        return None
    
    @classmethod
    def __subclasshook__(cls, subclass):
        'Abstract classes can override this to customize issubclass().\n\nThis is invoked early on by abc.ABCMeta.__subclasscheck__().\nIt should return True, False or NotImplemented.  If it returns\nNotImplemented, the normal algorithm is used.  Otherwise, it\noverrides the normal algorithm (and the outcome is cached).\n'
        return False
    

class SquaredEpsilonInsensitive(Regression):
    'Epsilon-Insensitive loss.\n\n    loss = max(0, |y - p| - epsilon)^2\n    '
    __class__ = SquaredEpsilonInsensitive
    def __init__(self, *args, **kwargs):
        'Epsilon-Insensitive loss.\n\n    loss = max(0, |y - p| - epsilon)^2\n    '
        pass
    
    @classmethod
    def __init_subclass__(cls):
        'This method is called when a class is subclassed.\n\nThe default implementation does nothing. It may be\noverridden to extend subclasses.\n'
        return None
    
    __pyx_vtable__ = _mod_builtins.PyCapsule()
    def __reduce__(self):
        return ''; return ()
    
    @classmethod
    def __subclasshook__(cls, subclass):
        'Abstract classes can override this to customize issubclass().\n\nThis is invoked early on by abc.ABCMeta.__subclasscheck__().\nIt should return True, False or NotImplemented.  If it returns\nNotImplemented, the normal algorithm is used.  Otherwise, it\noverrides the normal algorithm (and the outcome is cached).\n'
        return False
    

class SquaredHinge(Classification):
    'Squared Hinge loss for binary classification tasks with y in {-1,1}\n\n    Parameters\n    ----------\n\n    threshold : float > 0.0\n        Margin threshold. When threshold=1.0, one gets the loss used by\n        (quadratically penalized) SVM.\n    '
    __class__ = SquaredHinge
    def __init__(self, *args, **kwargs):
        'Squared Hinge loss for binary classification tasks with y in {-1,1}\n\n    Parameters\n    ----------\n\n    threshold : float > 0.0\n        Margin threshold. When threshold=1.0, one gets the loss used by\n        (quadratically penalized) SVM.\n    '
        pass
    
    @classmethod
    def __init_subclass__(cls):
        'This method is called when a class is subclassed.\n\nThe default implementation does nothing. It may be\noverridden to extend subclasses.\n'
        return None
    
    __pyx_vtable__ = _mod_builtins.PyCapsule()
    def __reduce__(self):
        return ''; return ()
    
    @classmethod
    def __subclasshook__(cls, subclass):
        'Abstract classes can override this to customize issubclass().\n\nThis is invoked early on by abc.ABCMeta.__subclasscheck__().\nIt should return True, False or NotImplemented.  If it returns\nNotImplemented, the normal algorithm is used.  Otherwise, it\noverrides the normal algorithm (and the outcome is cached).\n'
        return False
    

class SquaredLoss(Regression):
    'Squared loss traditional used in linear regression.'
    __class__ = SquaredLoss
    def __init__(self, *args, **kwargs):
        'Squared loss traditional used in linear regression.'
        pass
    
    @classmethod
    def __init_subclass__(cls):
        'This method is called when a class is subclassed.\n\nThe default implementation does nothing. It may be\noverridden to extend subclasses.\n'
        return None
    
    __pyx_vtable__ = _mod_builtins.PyCapsule()
    def __reduce__(self):
        return ''; return ()
    
    @classmethod
    def __subclasshook__(cls, subclass):
        'Abstract classes can override this to customize issubclass().\n\nThis is invoked early on by abc.ABCMeta.__subclasscheck__().\nIt should return True, False or NotImplemented.  If it returns\nNotImplemented, the normal algorithm is used.  Otherwise, it\noverrides the normal algorithm (and the outcome is cached).\n'
        return False
    

__builtins__ = {}
__doc__ = None
__file__ = '/home/claudinei/.local/lib/python3.6/site-packages/sklearn/linear_model/sgd_fast.cpython-36m-x86_64-linux-gnu.so'
__name__ = 'sklearn.linear_model.sgd_fast'
__package__ = 'sklearn.linear_model'
def __pyx_unpickle_Classification():
    pass

def __pyx_unpickle_Enum():
    pass

def __pyx_unpickle_LossFunction():
    pass

def __pyx_unpickle_Regression():
    pass

__test__ = _mod_builtins.dict()
def _plain_sgd():
    pass

def average_sgd():
    'Average SGD for generic loss functions and penalties.\n\n    Parameters\n    ----------\n    weights : ndarray[double, ndim=1]\n        The allocated coef_ vector.\n    intercept : double\n        The initial intercept.\n    average_weights : ndarray[double, ndim=1]\n        The average weights as computed for ASGD\n    average_intercept : double\n        The average intercept for ASGD\n    loss : LossFunction\n        A concrete ``LossFunction`` object.\n    penalty_type : int\n        The penalty 2 for L2, 1 for L1, and 3 for Elastic-Net.\n    alpha : float\n        The regularization parameter.\n    C : float\n        Maximum step size for passive aggressive.\n    l1_ratio : float\n        The Elastic Net mixing parameter, with 0 <= l1_ratio <= 1.\n        l1_ratio=0 corresponds to L2 penalty, l1_ratio=1 to L1.\n    dataset : SequentialDataset\n        A concrete ``SequentialDataset`` object.\n    validation_mask : ndarray[unsigned char, ndim=1]\n        Equal to True on the validation set.\n    early_stopping : boolean\n        Whether to use a stopping criterion based on the validation set.\n    validation_score_cb : callable\n        A callable to compute a validation score given the current\n        coefficients and intercept values.\n        Used only if early_stopping is True.\n    n_iter_no_change : int\n        Number of iteration with no improvement to wait before stopping.\n    max_iter : int\n        The maximum number of iterations (epochs).\n    tol: double\n        The tolerance for the stopping criterion.\n    dataset : SequentialDataset\n        A concrete ``SequentialDataset`` object.\n    fit_intercept : int\n        Whether or not to fit the intercept (1 or 0).\n    verbose : int\n        Print verbose output; 0 for quite.\n    shuffle : boolean\n        Whether to shuffle the training data before each epoch.\n    weight_pos : float\n        The weight of the positive class.\n    weight_neg : float\n        The weight of the negative class.\n    seed : np.uint32_t\n        Seed of the pseudorandom number generator used to shuffle the data.\n    learning_rate : int\n        The learning rate:\n        (1) constant, eta = eta0\n        (2) optimal, eta = 1.0/(alpha * t).\n        (3) inverse scaling, eta = eta0 / pow(t, power_t)\n        (4) adaptive decrease\n        (5) Passive Aggressive-I, eta = min(alpha, loss/norm(x))\n        (6) Passive Aggressive-II, eta = 1.0 / (norm(x) + 0.5*alpha)\n    eta0 : double\n        The initial learning rate.\n    power_t : double\n        The exponent for inverse scaling learning rate.\n    t : double\n        Initial state of the learning rate. This value is equal to the\n        iteration count except when the learning rate is set to `optimal`.\n        Default: 1.0.\n    average : int\n        The number of iterations before averaging starts. average=1 is\n        equivalent to averaging for all iterations.\n\n    Returns\n    -------\n    weights : array, shape=[n_features]\n        The fitted weight vector.\n    intercept : float\n        The fitted intercept term.\n    average_weights : array shape=[n_features]\n        The averaged weights across iterations\n    average_intercept : float\n        The averaged intercept across iterations\n    n_iter_ : int\n        The actual number of iter (epochs).\n    '
    pass

def plain_sgd():
    'Plain SGD for generic loss functions and penalties.\n\n    Parameters\n    ----------\n    weights : ndarray[double, ndim=1]\n        The allocated coef_ vector.\n    intercept : double\n        The initial intercept.\n    loss : LossFunction\n        A concrete ``LossFunction`` object.\n    penalty_type : int\n        The penalty 2 for L2, 1 for L1, and 3 for Elastic-Net.\n    alpha : float\n        The regularization parameter.\n    C : float\n        Maximum step size for passive aggressive.\n    l1_ratio : float\n        The Elastic Net mixing parameter, with 0 <= l1_ratio <= 1.\n        l1_ratio=0 corresponds to L2 penalty, l1_ratio=1 to L1.\n    dataset : SequentialDataset\n        A concrete ``SequentialDataset`` object.\n    validation_mask : ndarray[unsigned char, ndim=1]\n        Equal to True on the validation set.\n    early_stopping : boolean\n        Whether to use a stopping criterion based on the validation set.\n    validation_score_cb : callable\n        A callable to compute a validation score given the current\n        coefficients and intercept values.\n        Used only if early_stopping is True.\n    n_iter_no_change : int\n        Number of iteration with no improvement to wait before stopping.\n    max_iter : int\n        The maximum number of iterations (epochs).\n    tol: double\n        The tolerance for the stopping criterion.\n    fit_intercept : int\n        Whether or not to fit the intercept (1 or 0).\n    verbose : int\n        Print verbose output; 0 for quite.\n    shuffle : boolean\n        Whether to shuffle the training data before each epoch.\n    weight_pos : float\n        The weight of the positive class.\n    weight_neg : float\n        The weight of the negative class.\n    seed : np.uint32_t\n        Seed of the pseudorandom number generator used to shuffle the data.\n    learning_rate : int\n        The learning rate:\n        (1) constant, eta = eta0\n        (2) optimal, eta = 1.0/(alpha * t).\n        (3) inverse scaling, eta = eta0 / pow(t, power_t)\n        (4) adaptive decrease\n        (5) Passive Aggressive-I, eta = min(alpha, loss/norm(x))\n        (6) Passive Aggressive-II, eta = 1.0 / (norm(x) + 0.5*alpha)\n    eta0 : double\n        The initial learning rate.\n    power_t : double\n        The exponent for inverse scaling learning rate.\n    t : double\n        Initial state of the learning rate. This value is equal to the\n        iteration count except when the learning rate is set to `optimal`.\n        Default: 1.0.\n    intercept_decay : double\n        The decay ratio of intercept, used in updating intercept.\n\n    Returns\n    -------\n    weights : array, shape=[n_features]\n        The fitted weight vector.\n    intercept : float\n        The fitted intercept term.\n    n_iter_ : int\n        The actual number of iter (epochs).\n    '
    pass

def time():
    'time() -> floating point number\n\nReturn the current time in seconds since the Epoch.\nFractions of a second may be present if the system clock provides them.'
    return 1.0

