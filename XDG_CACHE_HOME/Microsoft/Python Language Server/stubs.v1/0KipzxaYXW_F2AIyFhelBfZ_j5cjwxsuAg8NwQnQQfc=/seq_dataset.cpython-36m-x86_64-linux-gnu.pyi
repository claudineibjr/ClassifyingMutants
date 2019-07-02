import builtins as _mod_builtins

class ArrayDataset32(SequentialDataset32):
    'Dataset backed by a two-dimensional numpy array.\n\n    The dtype of the numpy array is expected to be ``np.float32`` (float)\n    and C-style memory layout.\n    '
    __class__ = ArrayDataset32
    def __init__(self, *args, **kwargs):
        'Dataset backed by a two-dimensional numpy array.\n\n    The dtype of the numpy array is expected to be ``np.float32`` (float)\n    and C-style memory layout.\n    '
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
    

class ArrayDataset64(SequentialDataset64):
    'Dataset backed by a two-dimensional numpy array.\n\n    The dtype of the numpy array is expected to be ``np.float64`` (double)\n    and C-style memory layout.\n    '
    __class__ = ArrayDataset64
    def __init__(self, *args, **kwargs):
        'Dataset backed by a two-dimensional numpy array.\n\n    The dtype of the numpy array is expected to be ``np.float64`` (double)\n    and C-style memory layout.\n    '
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
    

class CSRDataset32(SequentialDataset32):
    'A ``SequentialDataset`` backed by a scipy sparse CSR matrix. '
    __class__ = CSRDataset32
    def __init__(self, *args, **kwargs):
        'A ``SequentialDataset`` backed by a scipy sparse CSR matrix. '
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
    

class CSRDataset64(SequentialDataset64):
    'A ``SequentialDataset`` backed by a scipy sparse CSR matrix. '
    __class__ = CSRDataset64
    def __init__(self, *args, **kwargs):
        'A ``SequentialDataset`` backed by a scipy sparse CSR matrix. '
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
    

class SequentialDataset32(_mod_builtins.object):
    'Base class for datasets with sequential data access.\n\n    SequentialDataset is used to iterate over the rows of a matrix X and\n    corresponding target values y, i.e. to iterate over samples.\n    There are two methods to get the next sample:\n        - next : Iterate sequentially (optionally randomized)\n        - random : Iterate randomly (with replacement)\n\n    Attributes\n    ----------\n    index : np.ndarray\n        Index array for fast shuffling.\n\n    index_data_ptr : int\n        Pointer to the index array.\n\n    current_index : int\n        Index of current sample in ``index``.\n        The index of current sample in the data is given by\n        index_data_ptr[current_index].\n\n    n_samples : Py_ssize_t\n        Number of samples in the dataset.\n\n    seed : np.uint32_t\n        Seed used for random sampling.\n\n    '
    __class__ = SequentialDataset32
    def __init__(self, *args, **kwargs):
        'Base class for datasets with sequential data access.\n\n    SequentialDataset is used to iterate over the rows of a matrix X and\n    corresponding target values y, i.e. to iterate over samples.\n    There are two methods to get the next sample:\n        - next : Iterate sequentially (optionally randomized)\n        - random : Iterate randomly (with replacement)\n\n    Attributes\n    ----------\n    index : np.ndarray\n        Index array for fast shuffling.\n\n    index_data_ptr : int\n        Pointer to the index array.\n\n    current_index : int\n        Index of current sample in ``index``.\n        The index of current sample in the data is given by\n        index_data_ptr[current_index].\n\n    n_samples : Py_ssize_t\n        Number of samples in the dataset.\n\n    seed : np.uint32_t\n        Seed used for random sampling.\n\n    '
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
    
    def _next_py(self):
        'python function used for easy testing'
        pass
    
    def _random_py(self):
        'python function used for easy testing'
        pass
    
    def _sample_py(self):
        'python function used for easy testing'
        pass
    
    def _shuffle_py(self):
        'python function used for easy testing'
        pass
    

class SequentialDataset64(_mod_builtins.object):
    'Base class for datasets with sequential data access.\n\n    SequentialDataset is used to iterate over the rows of a matrix X and\n    corresponding target values y, i.e. to iterate over samples.\n    There are two methods to get the next sample:\n        - next : Iterate sequentially (optionally randomized)\n        - random : Iterate randomly (with replacement)\n\n    Attributes\n    ----------\n    index : np.ndarray\n        Index array for fast shuffling.\n\n    index_data_ptr : int\n        Pointer to the index array.\n\n    current_index : int\n        Index of current sample in ``index``.\n        The index of current sample in the data is given by\n        index_data_ptr[current_index].\n\n    n_samples : Py_ssize_t\n        Number of samples in the dataset.\n\n    seed : np.uint32_t\n        Seed used for random sampling.\n\n    '
    __class__ = SequentialDataset64
    def __init__(self, *args, **kwargs):
        'Base class for datasets with sequential data access.\n\n    SequentialDataset is used to iterate over the rows of a matrix X and\n    corresponding target values y, i.e. to iterate over samples.\n    There are two methods to get the next sample:\n        - next : Iterate sequentially (optionally randomized)\n        - random : Iterate randomly (with replacement)\n\n    Attributes\n    ----------\n    index : np.ndarray\n        Index array for fast shuffling.\n\n    index_data_ptr : int\n        Pointer to the index array.\n\n    current_index : int\n        Index of current sample in ``index``.\n        The index of current sample in the data is given by\n        index_data_ptr[current_index].\n\n    n_samples : Py_ssize_t\n        Number of samples in the dataset.\n\n    seed : np.uint32_t\n        Seed used for random sampling.\n\n    '
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
    
    def _next_py(self):
        'python function used for easy testing'
        pass
    
    def _random_py(self):
        'python function used for easy testing'
        pass
    
    def _sample_py(self):
        'python function used for easy testing'
        pass
    
    def _shuffle_py(self):
        'python function used for easy testing'
        pass
    

__builtins__ = {}
__doc__ = '\nDataset abstractions for sequential data access.\nWARNING: Do not edit .pyx file directly, it is generated from .pyx.tp\n'
__file__ = '/home/claudinei/.local/lib/python3.6/site-packages/sklearn/utils/seq_dataset.cpython-36m-x86_64-linux-gnu.so'
__name__ = 'sklearn.utils.seq_dataset'
__package__ = 'sklearn.utils'
__test__ = _mod_builtins.dict()
