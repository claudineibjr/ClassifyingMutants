import builtins as _mod_builtins
import scipy.sparse.csc as _mod_scipy_sparse_csc

class VarHeader5(_mod_builtins.object):
    __class__ = VarHeader5
    def __init__(self, *args, **kwargs):
        pass
    
    @classmethod
    def __init_subclass__(cls):
        'This method is called when a class is subclassed.\n\nThe default implementation does nothing. It may be\noverridden to extend subclasses.\n'
        return None
    
    def __reduce__(self):
        return ''; return ()
    
    def __setstate__(self, state):
        return None
    
    @classmethod
    def __subclasshook__(cls, subclass):
        'Abstract classes can override this to customize issubclass().\n\nThis is invoked early on by abc.ABCMeta.__subclasscheck__().\nIt should return True, False or NotImplemented.  If it returns\nNotImplemented, the normal algorithm is used.  Otherwise, it\noverrides the normal algorithm (and the outcome is cached).\n'
        return False
    
    @property
    def dims(self):
        pass
    
    @property
    def is_global(self):
        pass
    
    @property
    def is_logical(self):
        pass
    
    @property
    def mclass(self):
        pass
    
    @property
    def name(self):
        pass
    
    @property
    def nzmax(self):
        pass
    
    def set_dims(self):
        ' Allow setting of dimensions from python\n\n        This is for constructing headers for tests\n        '
        pass
    

class VarReader5(_mod_builtins.object):
    __class__ = VarReader5
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
    
    def array_from_header(self):
        ' Read array of any class, given matrix `header`\n\n        Parameters\n        ----------\n        header : VarHeader5\n           array header object\n        process : int, optional\n           If not zero, apply post-processing on returned array\n           \n        Returns\n        -------\n        arr : array or sparse array\n           read array\n        '
        pass
    
    @property
    def is_swapped(self):
        pass
    
    @property
    def little_endian(self):
        pass
    
    def read_cells(self):
        ' Read cell array from stream '
        pass
    
    def read_char(self):
        ' Read char matrices from stream as arrays\n\n        Matrices of char are likely to be converted to matrices of\n        string by later processing in ``array_from_header``\n        '
        pass
    
    def read_fieldnames(self):
        'Read fieldnames for struct-like matrix.'
        pass
    
    def read_full_tag(self):
        ' Python method for reading full u4, u4 tag from stream\n\n        Returns\n        -------\n        mdtype : int32\n           matlab data type code\n        byte_count : int32\n           number of data bytes following\n\n        Notes\n        -----\n        Assumes tag is in fact full, that is, is not a small data\n        element.  This means it can skip some checks and makes it\n        slightly faster than ``read_tag``\n        '
        pass
    
    def read_header(self):
        ' Return matrix header for current stream position\n\n        Returns matrix headers at top level and sub levels\n\n        Parameters\n        ----------\n        check_stream_limit : if True, then if the returned header\n        is passed to array_from_header, it will be verified that\n        the length of the uncompressed data is not overlong (which\n        can indicate .mat file corruption)\n        '
        pass
    
    def read_numeric(self):
        ' Read numeric data element into ndarray\n\n        Reads element, then casts to ndarray.\n\n        The type of the array is usually given by the ``mdtype`` returned via\n        ``read_element``.  Sparse logical arrays are an exception, where the\n        type of the array may be ``np.bool`` even if the ``mdtype`` claims the\n        data is of float64 type.\n\n        Parameters\n        ----------\n        copy : bool, optional\n            Whether to copy the array before returning.  If False, return array\n            backed by bytes read from file.\n        nnz : int, optional\n            Number of non-zero values when reading numeric data from sparse\n            matrices.  -1 if not reading sparse matrices, or to disable check\n            for bytes data instead of declared data type (see Notes).\n\n        Returns\n        -------\n        arr : array\n            Numeric array\n\n        Notes\n        -----\n        MATLAB apparently likes to store sparse logical matrix data as bytes\n        instead of miDOUBLE (float64) data type, even though the data element\n        still declares its type as miDOUBLE.  We can guess this has happened by\n        looking for the length of the data compared to the expected number of\n        elements, using the `nnz` input parameter.\n        '
        pass
    
    def read_opaque(self):
        " Read opaque (function workspace) type\n\n        Looking at some mat files, the structure of this type seems to\n        be:\n\n        * array flags as usual (already read into `hdr`)\n        * 3 int8 strings\n        * a matrix\n\n        Then there's a matrix at the end of the mat file that seems have\n        the anonymous founction workspaces - we load it as\n        ``__function_workspace__``\n\n        See the comments at the beginning of ``mio5.py``\n        "
        pass
    
    def read_real_complex(self):
        ' Read real / complex matrices from stream '
        pass
    
    def read_struct(self):
        ' Read struct or object array from stream\n\n        Objects are just structs with an extra field *classname*,\n        defined before (this here) struct format structure\n        '
        pass
    
    def read_tag(self):
        ' Read tag mdtype and byte_count\n\n        Does necessary swapping and takes account of SDE formats.\n\n        See also ``read_full_tag`` method.\n        \n        Returns\n        -------\n        mdtype : int\n           matlab data type code\n        byte_count : int\n           number of bytes following that comprise the data\n        tag_data : None or str\n           Any data from the tag itself.  This is None for a full tag,\n           and string length `byte_count` if this is a small data\n           element.\n        '
        pass
    
    def set_stream(self):
        ' Set stream of best type from file-like `fobj`\n\n        Called from Python when initiating a variable read\n        '
        pass
    
    def shape_from_header(self):
        pass
    

__builtins__ = {}
__doc__ = ' Cython mio5 utility routines (-*- python -*- like)\n\n'
__file__ = '/home/claudinei/.local/lib/python3.6/site-packages/scipy/io/matlab/mio5_utils.cpython-36m-x86_64-linux-gnu.so'
__name__ = 'scipy.io.matlab.mio5_utils'
__package__ = 'scipy.io.matlab'
def __pyx_unpickle_VarHeader5():
    pass

__test__ = _mod_builtins.dict()
def asbytes(s):
    pass

def asstr(s):
    pass

def byteswap_u4():
    pass

def chars_to_strings():
    " Convert final axis of char array to strings\n\n    Parameters\n    ----------\n    in_arr : array\n       dtype of 'U1'\n\n    Returns\n    -------\n    str_arr : array\n       dtype of 'UN' where N is the length of the last dimension of\n       ``arr``\n    "
    pass

csc_matrix = _mod_scipy_sparse_csc.csc_matrix
def pycopy(x):
    "Shallow copy operation on arbitrary Python objects.\n\n    See the module's __doc__ string for more info.\n    "
    pass

def squeeze_element():
    ' Return squeezed element\n\n    The returned object may not be an ndarray - for example if we do\n    ``arr.item`` to return a ``mat_struct`` object from a struct array '
    pass

swapped_code = '>'
