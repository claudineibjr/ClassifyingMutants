import builtins as _mod_builtins

class MultinomialLogLoss32(_mod_builtins.object):
    __class__ = MultinomialLogLoss32
    def __init__(self, *args, **kwargs):
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
    

class MultinomialLogLoss64(_mod_builtins.object):
    __class__ = MultinomialLogLoss64
    def __init__(self, *args, **kwargs):
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
__doc__ = '\nSAG and SAGA implementation\nWARNING: Do not edit .pyx file directly, it is generated from .pyx.tp\n'
__file__ = '/home/claudinei/.local/lib/python3.6/site-packages/sklearn/linear_model/sag_fast.cpython-36m-x86_64-linux-gnu.so'
__name__ = 'sklearn.linear_model.sag_fast'
__package__ = 'sklearn.linear_model'
__test__ = _mod_builtins.dict()
def _multinomial_grad_loss_all_samples():
    'Compute multinomial gradient and loss across all samples.\n\n    Used for testing purpose only.\n    '
    pass

def sag32():
    'Stochastic Average Gradient (SAG) and SAGA solvers.\n\n    Used in Ridge and LogisticRegression.\n\n    Reference\n    ---------\n    Schmidt, M., Roux, N. L., & Bach, F. (2013).\n    Minimizing finite sums with the stochastic average gradient\n    https://hal.inria.fr/hal-00860051/document\n    (section 4.3)\n\n    Defazio, A., Bach, F., Lacoste-Julien, S. (2014),\n    SAGA: A Fast Incremental Gradient Method With Support\n    for Non-Strongly Convex Composite Objectives\n    https://arxiv.org/abs/1407.0202\n\n    '
    pass

def sag64():
    'Stochastic Average Gradient (SAG) and SAGA solvers.\n\n    Used in Ridge and LogisticRegression.\n\n    Reference\n    ---------\n    Schmidt, M., Roux, N. L., & Bach, F. (2013).\n    Minimizing finite sums with the stochastic average gradient\n    https://hal.inria.fr/hal-00860051/document\n    (section 4.3)\n\n    Defazio, A., Bach, F., Lacoste-Julien, S. (2014),\n    SAGA: A Fast Incremental Gradient Method With Support\n    for Non-Strongly Convex Composite Objectives\n    https://arxiv.org/abs/1407.0202\n\n    '
    pass

