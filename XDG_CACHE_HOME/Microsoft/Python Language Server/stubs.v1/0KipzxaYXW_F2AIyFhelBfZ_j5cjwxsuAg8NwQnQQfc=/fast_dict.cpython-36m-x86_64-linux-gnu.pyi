import builtins as _mod_builtins

class IntFloatDict(_mod_builtins.object):
    __class__ = IntFloatDict
    def __delitem__(self, key):
        'Delete self[key].'
        return None
    
    def __getitem__(self, key):
        'Return self[key].'
        pass
    
    def __init__(self, *args, **kwargs):
        pass
    
    @classmethod
    def __init_subclass__(cls):
        'This method is called when a class is subclassed.\n\nThe default implementation does nothing. It may be\noverridden to extend subclasses.\n'
        return None
    
    def __iter__(self):
        'Implement iter(self).'
        return IntFloatDict()
    
    def __len__(self):
        'Return len(self).'
        return 0
    
    __pyx_vtable__ = _mod_builtins.PyCapsule()
    def __reduce__(self):
        return ''; return ()
    
    def __setitem__(self, key, value):
        'Set self[key] to value.'
        return None
    
    def __setstate__(self, state):
        return None
    
    @classmethod
    def __subclasshook__(cls, subclass):
        'Abstract classes can override this to customize issubclass().\n\nThis is invoked early on by abc.ABCMeta.__subclasscheck__().\nIt should return True, False or NotImplemented.  If it returns\nNotImplemented, the normal algorithm is used.  Otherwise, it\noverrides the normal algorithm (and the outcome is cached).\n'
        return False
    
    def append(self):
        pass
    
    def copy(self):
        pass
    
    def to_arrays(self):
        'Return the key, value representation of the IntFloatDict\n           object.\n\n           Returns\n           =======\n           keys : ndarray, shape (n_items, ), dtype=int\n                The indices of the data points\n           values : ndarray, shape (n_items, ), dtype=float\n                The values of the data points\n        '
        pass
    
    def update(self):
        pass
    

__builtins__ = {}
__doc__ = '\nUses C++ map containers for fast dict-like behavior with keys being\nintegers, and values float.\n'
__file__ = '/home/claudinei/.local/lib/python3.6/site-packages/sklearn/utils/fast_dict.cpython-36m-x86_64-linux-gnu.so'
__name__ = 'sklearn.utils.fast_dict'
__package__ = 'sklearn.utils'
def __pyx_unpickle_Enum():
    pass

def __pyx_unpickle_IntFloatDict():
    pass

__test__ = _mod_builtins.dict()
def argmin():
    pass

