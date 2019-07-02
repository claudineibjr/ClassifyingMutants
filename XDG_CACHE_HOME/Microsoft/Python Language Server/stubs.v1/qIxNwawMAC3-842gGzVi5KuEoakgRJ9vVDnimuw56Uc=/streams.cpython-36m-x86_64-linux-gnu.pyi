import builtins as _mod_builtins

class FileStream(GenericStream):
    __class__ = FileStream
    def __del__(self):
        return None
    
    def __init__(self, *args, **kwargs):
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
    
    def seek(self):
        pass
    
    def tell(self):
        pass
    

class GenericStream(_mod_builtins.object):
    __class__ = GenericStream
    def __init__(self, *args, **kwargs):
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
    
    def read(self):
        pass
    
    def seek(self):
        pass
    
    def tell(self):
        pass
    

class ZlibInputStream(GenericStream):
    '\n    File-like object uncompressing bytes from a zlib compressed stream.\n\n    Parameters\n    ----------\n    stream : file-like\n        Stream to read compressed data from.\n    max_length : int\n        Maximum number of bytes to read from the stream.\n\n    Notes\n    -----\n    Some matlab files contain zlib streams without valid Z_STREAM_END\n    termination.  To get round this, we use the decompressobj object, that\n    allows you to decode an incomplete stream.  See discussion at\n    https://bugs.python.org/issue8672\n\n    '
    __class__ = ZlibInputStream
    def __init__(self, *args, **kwargs):
        '\n    File-like object uncompressing bytes from a zlib compressed stream.\n\n    Parameters\n    ----------\n    stream : file-like\n        Stream to read compressed data from.\n    max_length : int\n        Maximum number of bytes to read from the stream.\n\n    Notes\n    -----\n    Some matlab files contain zlib streams without valid Z_STREAM_END\n    termination.  To get round this, we use the decompressobj object, that\n    allows you to decode an incomplete stream.  See discussion at\n    https://bugs.python.org/issue8672\n\n    '
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
    
    def all_data_read(self):
        pass
    
    def read(self):
        pass
    
    def seek(self):
        pass
    
    def tell(self):
        pass
    

__builtins__ = {}
__doc__ = None
__file__ = '/home/claudinei/.local/lib/python3.6/site-packages/scipy/io/matlab/streams.cpython-36m-x86_64-linux-gnu.so'
__name__ = 'scipy.io.matlab.streams'
__package__ = 'scipy.io.matlab'
__pyx_capi__ = _mod_builtins.dict()
def __pyx_unpickle_GenericStream():
    pass

def __pyx_unpickle_ZlibInputStream():
    pass

def __pyx_unpickle_cStringStream():
    pass

__test__ = _mod_builtins.dict()
def _read_into():
    pass

def _read_string():
    pass

class cStringStream(GenericStream):
    __class__ = cStringStream
    def __init__(self, *args, **kwargs):
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
    
    def seek(self):
        pass
    

def make_stream():
    ' Make stream of correct type for file-like `fobj`\n    '
    pass

