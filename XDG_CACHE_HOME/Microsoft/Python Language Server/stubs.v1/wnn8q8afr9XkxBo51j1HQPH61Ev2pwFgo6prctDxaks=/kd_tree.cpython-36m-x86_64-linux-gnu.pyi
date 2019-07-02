import builtins as _mod_builtins
import numpy as _mod_numpy

class dtype(_mod_builtins.object):
    'dtype(obj, align=False, copy=False)\n\n    Create a data type object.\n\n    A numpy array is homogeneous, and contains elements described by a\n    dtype object. A dtype object can be constructed from different\n    combinations of fundamental numeric types.\n\n    Parameters\n    ----------\n    obj\n        Object to be converted to a data type object.\n    align : bool, optional\n        Add padding to the fields to match what a C compiler would output\n        for a similar C-struct. Can be ``True`` only if `obj` is a dictionary\n        or a comma-separated string. If a struct dtype is being created,\n        this also sets a sticky alignment flag ``isalignedstruct``.\n    copy : bool, optional\n        Make a new copy of the data-type object. If ``False``, the result\n        may just be a reference to a built-in data-type object.\n\n    See also\n    --------\n    result_type\n\n    Examples\n    --------\n    Using array-scalar type:\n\n    >>> np.dtype(np.int16)\n    dtype(\'int16\')\n\n    Structured type, one field name \'f1\', containing int16:\n\n    >>> np.dtype([(\'f1\', np.int16)])\n    dtype([(\'f1\', \'<i2\')])\n\n    Structured type, one field named \'f1\', in itself containing a structured\n    type with one field:\n\n    >>> np.dtype([(\'f1\', [(\'f1\', np.int16)])])\n    dtype([(\'f1\', [(\'f1\', \'<i2\')])])\n\n    Structured type, two fields: the first field contains an unsigned int, the\n    second an int32:\n\n    >>> np.dtype([(\'f1\', np.uint), (\'f2\', np.int32)])\n    dtype([(\'f1\', \'<u4\'), (\'f2\', \'<i4\')])\n\n    Using array-protocol type strings:\n\n    >>> np.dtype([(\'a\',\'f8\'),(\'b\',\'S10\')])\n    dtype([(\'a\', \'<f8\'), (\'b\', \'|S10\')])\n\n    Using comma-separated field formats.  The shape is (2,3):\n\n    >>> np.dtype("i4, (2,3)f8")\n    dtype([(\'f0\', \'<i4\'), (\'f1\', \'<f8\', (2, 3))])\n\n    Using tuples.  ``int`` is a fixed type, 3 the field\'s shape.  ``void``\n    is a flexible type, here of size 10:\n\n    >>> np.dtype([(\'hello\',(int,3)),(\'world\',np.void,10)])\n    dtype([(\'hello\', \'<i4\', 3), (\'world\', \'|V10\')])\n\n    Subdivide ``int16`` into 2 ``int8``\'s, called x and y.  0 and 1 are\n    the offsets in bytes:\n\n    >>> np.dtype((np.int16, {\'x\':(np.int8,0), \'y\':(np.int8,1)}))\n    dtype((\'<i2\', [(\'x\', \'|i1\'), (\'y\', \'|i1\')]))\n\n    Using dictionaries.  Two fields named \'gender\' and \'age\':\n\n    >>> np.dtype({\'names\':[\'gender\',\'age\'], \'formats\':[\'S1\',np.uint8]})\n    dtype([(\'gender\', \'|S1\'), (\'age\', \'|u1\')])\n\n    Offsets in bytes, here 0 and 25:\n\n    >>> np.dtype({\'surname\':(\'S25\',0),\'age\':(np.uint8,25)})\n    dtype([(\'surname\', \'|S25\'), (\'age\', \'|u1\')])'
    def __bool__(self):
        'self != 0'
        return False
    
    __class__ = dtype
    def __eq__(self, value):
        'Return self==value.'
        return False
    
    def __ge__(self, value):
        'Return self>=value.'
        return False
    
    def __getitem__(self, key):
        'Return self[key].'
        pass
    
    def __gt__(self, value):
        'Return self>value.'
        return False
    
    def __hash__(self):
        'Return hash(self).'
        return 0
    
    def __init__(self, obj, align=False, copy=False):
        'dtype(obj, align=False, copy=False)\n\n    Create a data type object.\n\n    A numpy array is homogeneous, and contains elements described by a\n    dtype object. A dtype object can be constructed from different\n    combinations of fundamental numeric types.\n\n    Parameters\n    ----------\n    obj\n        Object to be converted to a data type object.\n    align : bool, optional\n        Add padding to the fields to match what a C compiler would output\n        for a similar C-struct. Can be ``True`` only if `obj` is a dictionary\n        or a comma-separated string. If a struct dtype is being created,\n        this also sets a sticky alignment flag ``isalignedstruct``.\n    copy : bool, optional\n        Make a new copy of the data-type object. If ``False``, the result\n        may just be a reference to a built-in data-type object.\n\n    See also\n    --------\n    result_type\n\n    Examples\n    --------\n    Using array-scalar type:\n\n    >>> np.dtype(np.int16)\n    dtype(\'int16\')\n\n    Structured type, one field name \'f1\', containing int16:\n\n    >>> np.dtype([(\'f1\', np.int16)])\n    dtype([(\'f1\', \'<i2\')])\n\n    Structured type, one field named \'f1\', in itself containing a structured\n    type with one field:\n\n    >>> np.dtype([(\'f1\', [(\'f1\', np.int16)])])\n    dtype([(\'f1\', [(\'f1\', \'<i2\')])])\n\n    Structured type, two fields: the first field contains an unsigned int, the\n    second an int32:\n\n    >>> np.dtype([(\'f1\', np.uint), (\'f2\', np.int32)])\n    dtype([(\'f1\', \'<u4\'), (\'f2\', \'<i4\')])\n\n    Using array-protocol type strings:\n\n    >>> np.dtype([(\'a\',\'f8\'),(\'b\',\'S10\')])\n    dtype([(\'a\', \'<f8\'), (\'b\', \'|S10\')])\n\n    Using comma-separated field formats.  The shape is (2,3):\n\n    >>> np.dtype("i4, (2,3)f8")\n    dtype([(\'f0\', \'<i4\'), (\'f1\', \'<f8\', (2, 3))])\n\n    Using tuples.  ``int`` is a fixed type, 3 the field\'s shape.  ``void``\n    is a flexible type, here of size 10:\n\n    >>> np.dtype([(\'hello\',(int,3)),(\'world\',np.void,10)])\n    dtype([(\'hello\', \'<i4\', 3), (\'world\', \'|V10\')])\n\n    Subdivide ``int16`` into 2 ``int8``\'s, called x and y.  0 and 1 are\n    the offsets in bytes:\n\n    >>> np.dtype((np.int16, {\'x\':(np.int8,0), \'y\':(np.int8,1)}))\n    dtype((\'<i2\', [(\'x\', \'|i1\'), (\'y\', \'|i1\')]))\n\n    Using dictionaries.  Two fields named \'gender\' and \'age\':\n\n    >>> np.dtype({\'names\':[\'gender\',\'age\'], \'formats\':[\'S1\',np.uint8]})\n    dtype([(\'gender\', \'|S1\'), (\'age\', \'|u1\')])\n\n    Offsets in bytes, here 0 and 25:\n\n    >>> np.dtype({\'surname\':(\'S25\',0),\'age\':(np.uint8,25)})\n    dtype([(\'surname\', \'|S25\'), (\'age\', \'|u1\')])'
        pass
    
    @classmethod
    def __init_subclass__(cls):
        'This method is called when a class is subclassed.\n\nThe default implementation does nothing. It may be\noverridden to extend subclasses.\n'
        return None
    
    def __le__(self, value):
        'Return self<=value.'
        return False
    
    def __len__(self):
        'Return len(self).'
        return 0
    
    def __lt__(self, value):
        'Return self<value.'
        return False
    
    def __mul__(self, value):
        'Return self*value.'
        return dtype()
    
    def __ne__(self, value):
        'Return self!=value.'
        return False
    
    def __reduce__(self):
        return ''; return ()
    
    def __repr__(self):
        'Return repr(self).'
        return ''
    
    def __rmul__(self, value):
        'Return value*self.'
        return dtype()
    
    def __setstate__(self, state):
        return None
    
    def __str__(self):
        'Return str(self).'
        return ''
    
    @classmethod
    def __subclasshook__(cls, subclass):
        'Abstract classes can override this to customize issubclass().\n\nThis is invoked early on by abc.ABCMeta.__subclasscheck__().\nIt should return True, False or NotImplemented.  If it returns\nNotImplemented, the normal algorithm is used.  Otherwise, it\noverrides the normal algorithm (and the outcome is cached).\n'
        return False
    
    @property
    def alignment(self):
        'The required alignment (bytes) of this data-type according to the compiler.\n\n    More information is available in the C-API section of the manual.'
        pass
    
    @property
    def base(self):
        pass
    
    @property
    def byteorder(self):
        "A character indicating the byte-order of this data-type object.\n\n    One of:\n\n    ===  ==============\n    '='  native\n    '<'  little-endian\n    '>'  big-endian\n    '|'  not applicable\n    ===  ==============\n\n    All built-in data-type objects have byteorder either '=' or '|'.\n\n    Examples\n    --------\n\n    >>> dt = np.dtype('i2')\n    >>> dt.byteorder\n    '='\n    >>> # endian is not relevant for 8 bit numbers\n    >>> np.dtype('i1').byteorder\n    '|'\n    >>> # or ASCII strings\n    >>> np.dtype('S2').byteorder\n    '|'\n    >>> # Even if specific code is given, and it is native\n    >>> # '=' is the byteorder\n    >>> import sys\n    >>> sys_is_le = sys.byteorder == 'little'\n    >>> native_code = sys_is_le and '<' or '>'\n    >>> swapped_code = sys_is_le and '>' or '<'\n    >>> dt = np.dtype(native_code + 'i2')\n    >>> dt.byteorder\n    '='\n    >>> # Swapped code shows up as itself\n    >>> dt = np.dtype(swapped_code + 'i2')\n    >>> dt.byteorder == swapped_code\n    True"
        pass
    
    @property
    def char(self):
        'A unique character code for each of the 21 different built-in types.'
        pass
    
    @property
    def descr(self):
        "`__array_interface__` description of the data-type.\n\n    The format is that required by the 'descr' key in the\n    `__array_interface__` attribute.\n\n    Warning: This attribute exists specifically for `__array_interface__`,\n    and is not a datatype description compatible with `np.dtype`."
        pass
    
    @property
    def fields(self):
        "Dictionary of named fields defined for this data type, or ``None``.\n\n    The dictionary is indexed by keys that are the names of the fields.\n    Each entry in the dictionary is a tuple fully describing the field::\n\n      (dtype, offset[, title])\n\n    Offset is limited to C int, which is signed and usually 32 bits.\n    If present, the optional title can be any object (if it is a string\n    or unicode then it will also be a key in the fields dictionary,\n    otherwise it's meta-data). Notice also that the first two elements\n    of the tuple can be passed directly as arguments to the ``ndarray.getfield``\n    and ``ndarray.setfield`` methods.\n\n    See Also\n    --------\n    ndarray.getfield, ndarray.setfield\n\n    Examples\n    --------\n    >>> dt = np.dtype([('name', np.str_, 16), ('grades', np.float64, (2,))])\n    >>> print(dt.fields)\n    {'grades': (dtype(('float64',(2,))), 16), 'name': (dtype('|S16'), 0)}"
        pass
    
    @property
    def flags(self):
        'Bit-flags describing how this data type is to be interpreted.\n\n    Bit-masks are in `numpy.core.multiarray` as the constants\n    `ITEM_HASOBJECT`, `LIST_PICKLE`, `ITEM_IS_POINTER`, `NEEDS_INIT`,\n    `NEEDS_PYAPI`, `USE_GETITEM`, `USE_SETITEM`. A full explanation\n    of these flags is in C-API documentation; they are largely useful\n    for user-defined data-types.'
        pass
    
    @property
    def hasobject(self):
        "Boolean indicating whether this dtype contains any reference-counted\n    objects in any fields or sub-dtypes.\n\n    Recall that what is actually in the ndarray memory representing\n    the Python object is the memory address of that object (a pointer).\n    Special handling may be required, and this attribute is useful for\n    distinguishing data types that may contain arbitrary Python objects\n    and data-types that won't."
        pass
    
    @property
    def isalignedstruct(self):
        'Boolean indicating whether the dtype is a struct which maintains\n    field alignment. This flag is sticky, so when combining multiple\n    structs together, it is preserved and produces new dtypes which\n    are also aligned.'
        pass
    
    @property
    def isbuiltin(self):
        "Integer indicating how this dtype relates to the built-in dtypes.\n\n    Read-only.\n\n    =  ========================================================================\n    0  if this is a structured array type, with fields\n    1  if this is a dtype compiled into numpy (such as ints, floats etc)\n    2  if the dtype is for a user-defined numpy type\n       A user-defined type uses the numpy C-API machinery to extend\n       numpy to handle a new array type. See\n       :ref:`user.user-defined-data-types` in the NumPy manual.\n    =  ========================================================================\n\n    Examples\n    --------\n    >>> dt = np.dtype('i2')\n    >>> dt.isbuiltin\n    1\n    >>> dt = np.dtype('f8')\n    >>> dt.isbuiltin\n    1\n    >>> dt = np.dtype([('field1', 'f8')])\n    >>> dt.isbuiltin\n    0"
        pass
    
    @property
    def isnative(self):
        'Boolean indicating whether the byte order of this dtype is native\n    to the platform.'
        pass
    
    @property
    def itemsize(self):
        'The element size of this data-type object.\n\n    For 18 of the 21 types this number is fixed by the data-type.\n    For the flexible data-types, this number can be anything.'
        pass
    
    @property
    def kind(self):
        "A character code (one of 'biufcmMOSUV') identifying the general kind of data.\n\n    =  ======================\n    b  boolean\n    i  signed integer\n    u  unsigned integer\n    f  floating-point\n    c  complex floating-point\n    m  timedelta\n    M  datetime\n    O  object\n    S  (byte-)string\n    U  Unicode\n    V  void\n    =  ======================"
        pass
    
    @property
    def metadata(self):
        pass
    
    @property
    def name(self):
        'A bit-width name for this data-type.\n\n    Un-sized flexible data-type objects do not have this attribute.'
        pass
    
    @property
    def names(self):
        "Ordered list of field names, or ``None`` if there are no fields.\n\n    The names are ordered according to increasing byte offset. This can be\n    used, for example, to walk through all of the named fields in offset order.\n\n    Examples\n    --------\n    >>> dt = np.dtype([('name', np.str_, 16), ('grades', np.float64, (2,))])\n    >>> dt.names\n    ('name', 'grades')"
        pass
    
    @property
    def ndim(self):
        'Number of dimensions of the sub-array if this data type describes a\n    sub-array, and ``0`` otherwise.\n\n    .. versionadded:: 1.13.0'
        pass
    
    def newbyteorder(self, new_order='S'):
        "newbyteorder(new_order='S')\n\n    Return a new dtype with a different byte order.\n\n    Changes are also made in all fields and sub-arrays of the data type.\n\n    Parameters\n    ----------\n    new_order : string, optional\n        Byte order to force; a value from the byte order specifications\n        below.  The default value ('S') results in swapping the current\n        byte order.  `new_order` codes can be any of:\n\n        * 'S' - swap dtype from current to opposite endian\n        * {'<', 'L'} - little endian\n        * {'>', 'B'} - big endian\n        * {'=', 'N'} - native order\n        * {'|', 'I'} - ignore (no change to byte order)\n\n        The code does a case-insensitive check on the first letter of\n        `new_order` for these alternatives.  For example, any of '>'\n        or 'B' or 'b' or 'brian' are valid to specify big-endian.\n\n    Returns\n    -------\n    new_dtype : dtype\n        New dtype object with the given change to the byte order.\n\n    Notes\n    -----\n    Changes are also made in all fields and sub-arrays of the data type.\n\n    Examples\n    --------\n    >>> import sys\n    >>> sys_is_le = sys.byteorder == 'little'\n    >>> native_code = sys_is_le and '<' or '>'\n    >>> swapped_code = sys_is_le and '>' or '<'\n    >>> native_dt = np.dtype(native_code+'i2')\n    >>> swapped_dt = np.dtype(swapped_code+'i2')\n    >>> native_dt.newbyteorder('S') == swapped_dt\n    True\n    >>> native_dt.newbyteorder() == swapped_dt\n    True\n    >>> native_dt == swapped_dt.newbyteorder('S')\n    True\n    >>> native_dt == swapped_dt.newbyteorder('=')\n    True\n    >>> native_dt == swapped_dt.newbyteorder('N')\n    True\n    >>> native_dt == native_dt.newbyteorder('|')\n    True\n    >>> np.dtype('<i2') == native_dt.newbyteorder('<')\n    True\n    >>> np.dtype('<i2') == native_dt.newbyteorder('L')\n    True\n    >>> np.dtype('>i2') == native_dt.newbyteorder('>')\n    True\n    >>> np.dtype('>i2') == native_dt.newbyteorder('B')\n    True"
        pass
    
    @property
    def num(self):
        'A unique number for each of the 21 different built-in types.\n\n    These are roughly ordered from least-to-most precision.'
        pass
    
    @property
    def shape(self):
        'Shape tuple of the sub-array if this data type describes a sub-array,\n    and ``()`` otherwise.'
        pass
    
    @property
    def str(self):
        'The array-protocol typestring of this data-type object.'
        pass
    
    @property
    def subdtype(self):
        'Tuple ``(item_dtype, shape)`` if this `dtype` describes a sub-array, and\n    None otherwise.\n\n    The *shape* is the fixed shape of the sub-array described by this\n    data type, and *item_dtype* the data type of the array.\n\n    If a field whose dtype object has this attribute is retrieved,\n    then the extra dimensions implied by *shape* are tacked on to\n    the end of the retrieved array.'
        pass
    
    @property
    def type(self):
        'The type object used to instantiate a scalar of this data-type.'
        pass
    

class dtype(_mod_builtins.object):
    'dtype(obj, align=False, copy=False)\n\n    Create a data type object.\n\n    A numpy array is homogeneous, and contains elements described by a\n    dtype object. A dtype object can be constructed from different\n    combinations of fundamental numeric types.\n\n    Parameters\n    ----------\n    obj\n        Object to be converted to a data type object.\n    align : bool, optional\n        Add padding to the fields to match what a C compiler would output\n        for a similar C-struct. Can be ``True`` only if `obj` is a dictionary\n        or a comma-separated string. If a struct dtype is being created,\n        this also sets a sticky alignment flag ``isalignedstruct``.\n    copy : bool, optional\n        Make a new copy of the data-type object. If ``False``, the result\n        may just be a reference to a built-in data-type object.\n\n    See also\n    --------\n    result_type\n\n    Examples\n    --------\n    Using array-scalar type:\n\n    >>> np.dtype(np.int16)\n    dtype(\'int16\')\n\n    Structured type, one field name \'f1\', containing int16:\n\n    >>> np.dtype([(\'f1\', np.int16)])\n    dtype([(\'f1\', \'<i2\')])\n\n    Structured type, one field named \'f1\', in itself containing a structured\n    type with one field:\n\n    >>> np.dtype([(\'f1\', [(\'f1\', np.int16)])])\n    dtype([(\'f1\', [(\'f1\', \'<i2\')])])\n\n    Structured type, two fields: the first field contains an unsigned int, the\n    second an int32:\n\n    >>> np.dtype([(\'f1\', np.uint), (\'f2\', np.int32)])\n    dtype([(\'f1\', \'<u4\'), (\'f2\', \'<i4\')])\n\n    Using array-protocol type strings:\n\n    >>> np.dtype([(\'a\',\'f8\'),(\'b\',\'S10\')])\n    dtype([(\'a\', \'<f8\'), (\'b\', \'|S10\')])\n\n    Using comma-separated field formats.  The shape is (2,3):\n\n    >>> np.dtype("i4, (2,3)f8")\n    dtype([(\'f0\', \'<i4\'), (\'f1\', \'<f8\', (2, 3))])\n\n    Using tuples.  ``int`` is a fixed type, 3 the field\'s shape.  ``void``\n    is a flexible type, here of size 10:\n\n    >>> np.dtype([(\'hello\',(int,3)),(\'world\',np.void,10)])\n    dtype([(\'hello\', \'<i4\', 3), (\'world\', \'|V10\')])\n\n    Subdivide ``int16`` into 2 ``int8``\'s, called x and y.  0 and 1 are\n    the offsets in bytes:\n\n    >>> np.dtype((np.int16, {\'x\':(np.int8,0), \'y\':(np.int8,1)}))\n    dtype((\'<i2\', [(\'x\', \'|i1\'), (\'y\', \'|i1\')]))\n\n    Using dictionaries.  Two fields named \'gender\' and \'age\':\n\n    >>> np.dtype({\'names\':[\'gender\',\'age\'], \'formats\':[\'S1\',np.uint8]})\n    dtype([(\'gender\', \'|S1\'), (\'age\', \'|u1\')])\n\n    Offsets in bytes, here 0 and 25:\n\n    >>> np.dtype({\'surname\':(\'S25\',0),\'age\':(np.uint8,25)})\n    dtype([(\'surname\', \'|S25\'), (\'age\', \'|u1\')])'
    def __bool__(self):
        'self != 0'
        return False
    
    __class__ = dtype
    def __eq__(self, value):
        'Return self==value.'
        return False
    
    def __ge__(self, value):
        'Return self>=value.'
        return False
    
    def __getitem__(self, key):
        'Return self[key].'
        pass
    
    def __gt__(self, value):
        'Return self>value.'
        return False
    
    def __hash__(self):
        'Return hash(self).'
        return 0
    
    def __init__(self, obj, align=False, copy=False):
        'dtype(obj, align=False, copy=False)\n\n    Create a data type object.\n\n    A numpy array is homogeneous, and contains elements described by a\n    dtype object. A dtype object can be constructed from different\n    combinations of fundamental numeric types.\n\n    Parameters\n    ----------\n    obj\n        Object to be converted to a data type object.\n    align : bool, optional\n        Add padding to the fields to match what a C compiler would output\n        for a similar C-struct. Can be ``True`` only if `obj` is a dictionary\n        or a comma-separated string. If a struct dtype is being created,\n        this also sets a sticky alignment flag ``isalignedstruct``.\n    copy : bool, optional\n        Make a new copy of the data-type object. If ``False``, the result\n        may just be a reference to a built-in data-type object.\n\n    See also\n    --------\n    result_type\n\n    Examples\n    --------\n    Using array-scalar type:\n\n    >>> np.dtype(np.int16)\n    dtype(\'int16\')\n\n    Structured type, one field name \'f1\', containing int16:\n\n    >>> np.dtype([(\'f1\', np.int16)])\n    dtype([(\'f1\', \'<i2\')])\n\n    Structured type, one field named \'f1\', in itself containing a structured\n    type with one field:\n\n    >>> np.dtype([(\'f1\', [(\'f1\', np.int16)])])\n    dtype([(\'f1\', [(\'f1\', \'<i2\')])])\n\n    Structured type, two fields: the first field contains an unsigned int, the\n    second an int32:\n\n    >>> np.dtype([(\'f1\', np.uint), (\'f2\', np.int32)])\n    dtype([(\'f1\', \'<u4\'), (\'f2\', \'<i4\')])\n\n    Using array-protocol type strings:\n\n    >>> np.dtype([(\'a\',\'f8\'),(\'b\',\'S10\')])\n    dtype([(\'a\', \'<f8\'), (\'b\', \'|S10\')])\n\n    Using comma-separated field formats.  The shape is (2,3):\n\n    >>> np.dtype("i4, (2,3)f8")\n    dtype([(\'f0\', \'<i4\'), (\'f1\', \'<f8\', (2, 3))])\n\n    Using tuples.  ``int`` is a fixed type, 3 the field\'s shape.  ``void``\n    is a flexible type, here of size 10:\n\n    >>> np.dtype([(\'hello\',(int,3)),(\'world\',np.void,10)])\n    dtype([(\'hello\', \'<i4\', 3), (\'world\', \'|V10\')])\n\n    Subdivide ``int16`` into 2 ``int8``\'s, called x and y.  0 and 1 are\n    the offsets in bytes:\n\n    >>> np.dtype((np.int16, {\'x\':(np.int8,0), \'y\':(np.int8,1)}))\n    dtype((\'<i2\', [(\'x\', \'|i1\'), (\'y\', \'|i1\')]))\n\n    Using dictionaries.  Two fields named \'gender\' and \'age\':\n\n    >>> np.dtype({\'names\':[\'gender\',\'age\'], \'formats\':[\'S1\',np.uint8]})\n    dtype([(\'gender\', \'|S1\'), (\'age\', \'|u1\')])\n\n    Offsets in bytes, here 0 and 25:\n\n    >>> np.dtype({\'surname\':(\'S25\',0),\'age\':(np.uint8,25)})\n    dtype([(\'surname\', \'|S25\'), (\'age\', \'|u1\')])'
        pass
    
    @classmethod
    def __init_subclass__(cls):
        'This method is called when a class is subclassed.\n\nThe default implementation does nothing. It may be\noverridden to extend subclasses.\n'
        return None
    
    def __le__(self, value):
        'Return self<=value.'
        return False
    
    def __len__(self):
        'Return len(self).'
        return 0
    
    def __lt__(self, value):
        'Return self<value.'
        return False
    
    def __mul__(self, value):
        'Return self*value.'
        return dtype()
    
    def __ne__(self, value):
        'Return self!=value.'
        return False
    
    def __reduce__(self):
        return ''; return ()
    
    def __repr__(self):
        'Return repr(self).'
        return ''
    
    def __rmul__(self, value):
        'Return value*self.'
        return dtype()
    
    def __setstate__(self, state):
        return None
    
    def __str__(self):
        'Return str(self).'
        return ''
    
    @classmethod
    def __subclasshook__(cls, subclass):
        'Abstract classes can override this to customize issubclass().\n\nThis is invoked early on by abc.ABCMeta.__subclasscheck__().\nIt should return True, False or NotImplemented.  If it returns\nNotImplemented, the normal algorithm is used.  Otherwise, it\noverrides the normal algorithm (and the outcome is cached).\n'
        return False
    
    @property
    def alignment(self):
        'The required alignment (bytes) of this data-type according to the compiler.\n\n    More information is available in the C-API section of the manual.'
        pass
    
    @property
    def base(self):
        pass
    
    @property
    def byteorder(self):
        "A character indicating the byte-order of this data-type object.\n\n    One of:\n\n    ===  ==============\n    '='  native\n    '<'  little-endian\n    '>'  big-endian\n    '|'  not applicable\n    ===  ==============\n\n    All built-in data-type objects have byteorder either '=' or '|'.\n\n    Examples\n    --------\n\n    >>> dt = np.dtype('i2')\n    >>> dt.byteorder\n    '='\n    >>> # endian is not relevant for 8 bit numbers\n    >>> np.dtype('i1').byteorder\n    '|'\n    >>> # or ASCII strings\n    >>> np.dtype('S2').byteorder\n    '|'\n    >>> # Even if specific code is given, and it is native\n    >>> # '=' is the byteorder\n    >>> import sys\n    >>> sys_is_le = sys.byteorder == 'little'\n    >>> native_code = sys_is_le and '<' or '>'\n    >>> swapped_code = sys_is_le and '>' or '<'\n    >>> dt = np.dtype(native_code + 'i2')\n    >>> dt.byteorder\n    '='\n    >>> # Swapped code shows up as itself\n    >>> dt = np.dtype(swapped_code + 'i2')\n    >>> dt.byteorder == swapped_code\n    True"
        pass
    
    @property
    def char(self):
        'A unique character code for each of the 21 different built-in types.'
        pass
    
    @property
    def descr(self):
        "`__array_interface__` description of the data-type.\n\n    The format is that required by the 'descr' key in the\n    `__array_interface__` attribute.\n\n    Warning: This attribute exists specifically for `__array_interface__`,\n    and is not a datatype description compatible with `np.dtype`."
        pass
    
    @property
    def fields(self):
        "Dictionary of named fields defined for this data type, or ``None``.\n\n    The dictionary is indexed by keys that are the names of the fields.\n    Each entry in the dictionary is a tuple fully describing the field::\n\n      (dtype, offset[, title])\n\n    Offset is limited to C int, which is signed and usually 32 bits.\n    If present, the optional title can be any object (if it is a string\n    or unicode then it will also be a key in the fields dictionary,\n    otherwise it's meta-data). Notice also that the first two elements\n    of the tuple can be passed directly as arguments to the ``ndarray.getfield``\n    and ``ndarray.setfield`` methods.\n\n    See Also\n    --------\n    ndarray.getfield, ndarray.setfield\n\n    Examples\n    --------\n    >>> dt = np.dtype([('name', np.str_, 16), ('grades', np.float64, (2,))])\n    >>> print(dt.fields)\n    {'grades': (dtype(('float64',(2,))), 16), 'name': (dtype('|S16'), 0)}"
        pass
    
    @property
    def flags(self):
        'Bit-flags describing how this data type is to be interpreted.\n\n    Bit-masks are in `numpy.core.multiarray` as the constants\n    `ITEM_HASOBJECT`, `LIST_PICKLE`, `ITEM_IS_POINTER`, `NEEDS_INIT`,\n    `NEEDS_PYAPI`, `USE_GETITEM`, `USE_SETITEM`. A full explanation\n    of these flags is in C-API documentation; they are largely useful\n    for user-defined data-types.'
        pass
    
    @property
    def hasobject(self):
        "Boolean indicating whether this dtype contains any reference-counted\n    objects in any fields or sub-dtypes.\n\n    Recall that what is actually in the ndarray memory representing\n    the Python object is the memory address of that object (a pointer).\n    Special handling may be required, and this attribute is useful for\n    distinguishing data types that may contain arbitrary Python objects\n    and data-types that won't."
        pass
    
    @property
    def isalignedstruct(self):
        'Boolean indicating whether the dtype is a struct which maintains\n    field alignment. This flag is sticky, so when combining multiple\n    structs together, it is preserved and produces new dtypes which\n    are also aligned.'
        pass
    
    @property
    def isbuiltin(self):
        "Integer indicating how this dtype relates to the built-in dtypes.\n\n    Read-only.\n\n    =  ========================================================================\n    0  if this is a structured array type, with fields\n    1  if this is a dtype compiled into numpy (such as ints, floats etc)\n    2  if the dtype is for a user-defined numpy type\n       A user-defined type uses the numpy C-API machinery to extend\n       numpy to handle a new array type. See\n       :ref:`user.user-defined-data-types` in the NumPy manual.\n    =  ========================================================================\n\n    Examples\n    --------\n    >>> dt = np.dtype('i2')\n    >>> dt.isbuiltin\n    1\n    >>> dt = np.dtype('f8')\n    >>> dt.isbuiltin\n    1\n    >>> dt = np.dtype([('field1', 'f8')])\n    >>> dt.isbuiltin\n    0"
        pass
    
    @property
    def isnative(self):
        'Boolean indicating whether the byte order of this dtype is native\n    to the platform.'
        pass
    
    @property
    def itemsize(self):
        'The element size of this data-type object.\n\n    For 18 of the 21 types this number is fixed by the data-type.\n    For the flexible data-types, this number can be anything.'
        pass
    
    @property
    def kind(self):
        "A character code (one of 'biufcmMOSUV') identifying the general kind of data.\n\n    =  ======================\n    b  boolean\n    i  signed integer\n    u  unsigned integer\n    f  floating-point\n    c  complex floating-point\n    m  timedelta\n    M  datetime\n    O  object\n    S  (byte-)string\n    U  Unicode\n    V  void\n    =  ======================"
        pass
    
    @property
    def metadata(self):
        pass
    
    @property
    def name(self):
        'A bit-width name for this data-type.\n\n    Un-sized flexible data-type objects do not have this attribute.'
        pass
    
    @property
    def names(self):
        "Ordered list of field names, or ``None`` if there are no fields.\n\n    The names are ordered according to increasing byte offset. This can be\n    used, for example, to walk through all of the named fields in offset order.\n\n    Examples\n    --------\n    >>> dt = np.dtype([('name', np.str_, 16), ('grades', np.float64, (2,))])\n    >>> dt.names\n    ('name', 'grades')"
        pass
    
    @property
    def ndim(self):
        'Number of dimensions of the sub-array if this data type describes a\n    sub-array, and ``0`` otherwise.\n\n    .. versionadded:: 1.13.0'
        pass
    
    def newbyteorder(self, new_order='S'):
        "newbyteorder(new_order='S')\n\n    Return a new dtype with a different byte order.\n\n    Changes are also made in all fields and sub-arrays of the data type.\n\n    Parameters\n    ----------\n    new_order : string, optional\n        Byte order to force; a value from the byte order specifications\n        below.  The default value ('S') results in swapping the current\n        byte order.  `new_order` codes can be any of:\n\n        * 'S' - swap dtype from current to opposite endian\n        * {'<', 'L'} - little endian\n        * {'>', 'B'} - big endian\n        * {'=', 'N'} - native order\n        * {'|', 'I'} - ignore (no change to byte order)\n\n        The code does a case-insensitive check on the first letter of\n        `new_order` for these alternatives.  For example, any of '>'\n        or 'B' or 'b' or 'brian' are valid to specify big-endian.\n\n    Returns\n    -------\n    new_dtype : dtype\n        New dtype object with the given change to the byte order.\n\n    Notes\n    -----\n    Changes are also made in all fields and sub-arrays of the data type.\n\n    Examples\n    --------\n    >>> import sys\n    >>> sys_is_le = sys.byteorder == 'little'\n    >>> native_code = sys_is_le and '<' or '>'\n    >>> swapped_code = sys_is_le and '>' or '<'\n    >>> native_dt = np.dtype(native_code+'i2')\n    >>> swapped_dt = np.dtype(swapped_code+'i2')\n    >>> native_dt.newbyteorder('S') == swapped_dt\n    True\n    >>> native_dt.newbyteorder() == swapped_dt\n    True\n    >>> native_dt == swapped_dt.newbyteorder('S')\n    True\n    >>> native_dt == swapped_dt.newbyteorder('=')\n    True\n    >>> native_dt == swapped_dt.newbyteorder('N')\n    True\n    >>> native_dt == native_dt.newbyteorder('|')\n    True\n    >>> np.dtype('<i2') == native_dt.newbyteorder('<')\n    True\n    >>> np.dtype('<i2') == native_dt.newbyteorder('L')\n    True\n    >>> np.dtype('>i2') == native_dt.newbyteorder('>')\n    True\n    >>> np.dtype('>i2') == native_dt.newbyteorder('B')\n    True"
        pass
    
    @property
    def num(self):
        'A unique number for each of the 21 different built-in types.\n\n    These are roughly ordered from least-to-most precision.'
        pass
    
    @property
    def shape(self):
        'Shape tuple of the sub-array if this data type describes a sub-array,\n    and ``()`` otherwise.'
        pass
    
    @property
    def str(self):
        'The array-protocol typestring of this data-type object.'
        pass
    
    @property
    def subdtype(self):
        'Tuple ``(item_dtype, shape)`` if this `dtype` describes a sub-array, and\n    None otherwise.\n\n    The *shape* is the fixed shape of the sub-array described by this\n    data type, and *item_dtype* the data type of the array.\n\n    If a field whose dtype object has this attribute is retrieved,\n    then the extra dimensions implied by *shape* are tacked on to\n    the end of the retrieved array.'
        pass
    
    @property
    def type(self):
        'The type object used to instantiate a scalar of this data-type.'
        pass
    

class BinaryTree(_mod_builtins.object):
    __class__ = BinaryTree
    def __getstate__(self):
        '\n        get state for pickling\n        '
        pass
    
    def __init__(self, *args, **kwargs):
        pass
    
    @classmethod
    def __init_subclass__(cls):
        'This method is called when a class is subclassed.\n\nThe default implementation does nothing. It may be\noverridden to extend subclasses.\n'
        return None
    
    __pyx_vtable__ = _mod_builtins.PyCapsule()
    def __reduce__(self):
        '\n        reduce method used for pickling\n        '
        return ''; return ()
    
    def __setstate__(self, state):
        '\n        set state for pickling\n        '
        return None
    
    @classmethod
    def __subclasshook__(cls, subclass):
        'Abstract classes can override this to customize issubclass().\n\nThis is invoked early on by abc.ABCMeta.__subclasscheck__().\nIt should return True, False or NotImplemented.  If it returns\nNotImplemented, the normal algorithm is used.  Otherwise, it\noverrides the normal algorithm (and the outcome is cached).\n'
        return False
    
    def _update_memviews(self):
        pass
    
    def _update_sample_weight(self):
        pass
    
    @property
    def data(self):
        pass
    
    def get_arrays(self):
        pass
    
    def get_n_calls(self):
        pass
    
    def get_tree_stats(self):
        pass
    
    @property
    def idx_array(self):
        pass
    
    def kernel_density(self):
        "\n        kernel_density(self, X, h, kernel='gaussian', atol=0, rtol=1E-8,\n                       breadth_first=True, return_log=False)\n\n        Compute the kernel density estimate at points X with the given kernel,\n        using the distance metric specified at tree creation.\n\n        Parameters\n        ----------\n        X : array-like, shape = [n_samples, n_features]\n            An array of points to query.  Last dimension should match dimension\n            of training data.\n        h : float\n            the bandwidth of the kernel\n        kernel : string\n            specify the kernel to use.  Options are\n            - 'gaussian'\n            - 'tophat'\n            - 'epanechnikov'\n            - 'exponential'\n            - 'linear'\n            - 'cosine'\n            Default is kernel = 'gaussian'\n        atol, rtol : float (default = 0)\n            Specify the desired relative and absolute tolerance of the result.\n            If the true result is K_true, then the returned result K_ret\n            satisfies ``abs(K_true - K_ret) < atol + rtol * K_ret``\n            The default is zero (i.e. machine precision) for both.\n        breadth_first : boolean (default = False)\n            if True, use a breadth-first search.  If False (default) use a\n            depth-first search.  Breadth-first is generally faster for\n            compact kernels and/or high tolerances.\n        return_log : boolean (default = False)\n            return the logarithm of the result.  This can be more accurate\n            than returning the result itself for narrow kernels.\n\n        Returns\n        -------\n        density : ndarray\n            The array of (log)-density evaluations, shape = X.shape[:-1]\n        "
        pass
    
    @property
    def node_bounds(self):
        pass
    
    @property
    def node_data(self):
        pass
    
    def query(self):
        '\n        query(X, k=1, return_distance=True,\n              dualtree=False, breadth_first=False)\n\n        query the tree for the k nearest neighbors\n\n        Parameters\n        ----------\n        X : array-like, shape = [n_samples, n_features]\n            An array of points to query\n        k : integer  (default = 1)\n            The number of nearest neighbors to return\n        return_distance : boolean (default = True)\n            if True, return a tuple (d, i) of distances and indices\n            if False, return array i\n        dualtree : boolean (default = False)\n            if True, use the dual tree formalism for the query: a tree is\n            built for the query points, and the pair of trees is used to\n            efficiently search this space.  This can lead to better\n            performance as the number of points grows large.\n        breadth_first : boolean (default = False)\n            if True, then query the nodes in a breadth-first manner.\n            Otherwise, query the nodes in a depth-first manner.\n        sort_results : boolean (default = True)\n            if True, then distances and indices of each point are sorted\n            on return, so that the first column contains the closest points.\n            Otherwise, neighbors are returned in an arbitrary order.\n\n        Returns\n        -------\n        i    : if return_distance == False\n        (d,i) : if return_distance == True\n\n        d : array of doubles - shape: x.shape[:-1] + (k,)\n            each entry gives the list of distances to the\n            neighbors of the corresponding point\n\n        i : array of integers - shape: x.shape[:-1] + (k,)\n            each entry gives the list of indices of\n            neighbors of the corresponding point\n        '
        pass
    
    def query_radius(self):
        '\n        query_radius(self, X, r, count_only = False):\n\n        query the tree for neighbors within a radius r\n\n        Parameters\n        ----------\n        X : array-like, shape = [n_samples, n_features]\n            An array of points to query\n        r : distance within which neighbors are returned\n            r can be a single value, or an array of values of shape\n            x.shape[:-1] if different radii are desired for each point.\n        return_distance : boolean (default = False)\n            if True,  return distances to neighbors of each point\n            if False, return only neighbors\n            Note that unlike the query() method, setting return_distance=True\n            here adds to the computation time.  Not all distances need to be\n            calculated explicitly for return_distance=False.  Results are\n            not sorted by default: see ``sort_results`` keyword.\n        count_only : boolean (default = False)\n            if True,  return only the count of points within distance r\n            if False, return the indices of all points within distance r\n            If return_distance==True, setting count_only=True will\n            result in an error.\n        sort_results : boolean (default = False)\n            if True, the distances and indices will be sorted before being\n            returned.  If False, the results will not be sorted.  If\n            return_distance == False, setting sort_results = True will\n            result in an error.\n\n        Returns\n        -------\n        count       : if count_only == True\n        ind         : if count_only == False and return_distance == False\n        (ind, dist) : if count_only == False and return_distance == True\n\n        count : array of integers, shape = X.shape[:-1]\n            each entry gives the number of neighbors within\n            a distance r of the corresponding point.\n\n        ind : array of objects, shape = X.shape[:-1]\n            each element is a numpy integer array listing the indices of\n            neighbors of the corresponding point.  Note that unlike\n            the results of a k-neighbors query, the returned neighbors\n            are not sorted by distance by default.\n\n        dist : array of objects, shape = X.shape[:-1]\n            each element is a numpy double array\n            listing the distances corresponding to indices in i.\n        '
        pass
    
    def reset_n_calls(self):
        pass
    
    @property
    def sample_weight(self):
        pass
    
    @property
    def sum_weight(self):
        pass
    
    def two_point_correlation(self):
        'Compute the two-point correlation function\n\n        Parameters\n        ----------\n        X : array-like, shape = [n_samples, n_features]\n            An array of points to query.  Last dimension should match dimension\n            of training data.\n        r : array_like\n            A one-dimensional array of distances\n        dualtree : boolean (default = False)\n            If true, use a dualtree algorithm.  Otherwise, use a single-tree\n            algorithm.  Dual tree algorithms can have better scaling for\n            large N.\n\n        Returns\n        -------\n        counts : ndarray\n            counts[i] contains the number of pairs of points with distance\n            less than or equal to r[i]\n        '
        pass
    
    valid_metrics = _mod_builtins.list()

CLASS_DOC = "{BinaryTree} for fast generalized N-point problems\n\n{BinaryTree}(X, leaf_size=40, metric='minkowski', \\**kwargs)\n\nParameters\n----------\nX : array-like, shape = [n_samples, n_features]\n    n_samples is the number of points in the data set, and\n    n_features is the dimension of the parameter space.\n    Note: if X is a C-contiguous array of doubles then data will\n    not be copied. Otherwise, an internal copy will be made.\n\nleaf_size : positive integer (default = 40)\n    Number of points at which to switch to brute-force. Changing\n    leaf_size will not affect the results of a query, but can\n    significantly impact the speed of a query and the memory required\n    to store the constructed tree.  The amount of memory needed to\n    store the tree scales as approximately n_samples / leaf_size.\n    For a specified ``leaf_size``, a leaf node is guaranteed to\n    satisfy ``leaf_size <= n_points <= 2 * leaf_size``, except in\n    the case that ``n_samples < leaf_size``.\n\nmetric : string or DistanceMetric object\n    the distance metric to use for the tree.  Default='minkowski'\n    with p=2 (that is, a euclidean metric). See the documentation\n    of the DistanceMetric class for a list of available metrics.\n    {binary_tree}.valid_metrics gives a list of the metrics which\n    are valid for {BinaryTree}.\n\nAdditional keywords are passed to the distance metric class.\n\nAttributes\n----------\ndata : memory view\n    The training data\n\nExamples\n--------\nQuery for k-nearest neighbors\n\n    >>> import numpy as np\n    >>> rng = np.random.RandomState(0)\n    >>> X = rng.random_sample((10, 3))  # 10 points in 3 dimensions\n    >>> tree = {BinaryTree}(X, leaf_size=2)              # doctest: +SKIP\n    >>> dist, ind = tree.query(X[:1], k=3)                # doctest: +SKIP\n    >>> print(ind)  # indices of 3 closest neighbors\n    [0 3 1]\n    >>> print(dist)  # distances to 3 closest neighbors\n    [ 0.          0.19662693  0.29473397]\n\nPickle and Unpickle a tree.  Note that the state of the tree is saved in the\npickle operation: the tree needs not be rebuilt upon unpickling.\n\n    >>> import numpy as np\n    >>> import pickle\n    >>> rng = np.random.RandomState(0)\n    >>> X = rng.random_sample((10, 3))  # 10 points in 3 dimensions\n    >>> tree = {BinaryTree}(X, leaf_size=2)        # doctest: +SKIP\n    >>> s = pickle.dumps(tree)                     # doctest: +SKIP\n    >>> tree_copy = pickle.loads(s)                # doctest: +SKIP\n    >>> dist, ind = tree_copy.query(X[:1], k=3)     # doctest: +SKIP\n    >>> print(ind)  # indices of 3 closest neighbors\n    [0 3 1]\n    >>> print(dist)  # distances to 3 closest neighbors\n    [ 0.          0.19662693  0.29473397]\n\nQuery for neighbors within a given radius\n\n    >>> import numpy as np\n    >>> rng = np.random.RandomState(0)\n    >>> X = rng.random_sample((10, 3))  # 10 points in 3 dimensions\n    >>> tree = {BinaryTree}(X, leaf_size=2)     # doctest: +SKIP\n    >>> print(tree.query_radius(X[:1], r=0.3, count_only=True))\n    3\n    >>> ind = tree.query_radius(X[:1], r=0.3)  # doctest: +SKIP\n    >>> print(ind)  # indices of neighbors within distance 0.3\n    [3 0 1]\n\n\nCompute a gaussian kernel density estimate:\n\n    >>> import numpy as np\n    >>> rng = np.random.RandomState(42)\n    >>> X = rng.random_sample((100, 3))\n    >>> tree = {BinaryTree}(X)                # doctest: +SKIP\n    >>> tree.kernel_density(X[:3], h=0.1, kernel='gaussian')\n    array([ 6.94114649,  7.83281226,  7.2071716 ])\n\nCompute a two-point auto-correlation function\n\n    >>> import numpy as np\n    >>> rng = np.random.RandomState(0)\n    >>> X = rng.random_sample((30, 3))\n    >>> r = np.linspace(0, 1, 5)\n    >>> tree = {BinaryTree}(X)                # doctest: +SKIP\n    >>> tree.two_point_correlation(X, r)\n    array([ 30,  62, 278, 580, 820])\n\n"
DOC_DICT = _mod_builtins.dict()
DTYPE = _mod_numpy.float64
ITYPE = _mod_numpy.int64
class KDTree(BinaryTree):
    "KDTree for fast generalized N-point problems\n\nKDTree(X, leaf_size=40, metric='minkowski', \\**kwargs)\n\nParameters\n----------\nX : array-like, shape = [n_samples, n_features]\n    n_samples is the number of points in the data set, and\n    n_features is the dimension of the parameter space.\n    Note: if X is a C-contiguous array of doubles then data will\n    not be copied. Otherwise, an internal copy will be made.\n\nleaf_size : positive integer (default = 40)\n    Number of points at which to switch to brute-force. Changing\n    leaf_size will not affect the results of a query, but can\n    significantly impact the speed of a query and the memory required\n    to store the constructed tree.  The amount of memory needed to\n    store the tree scales as approximately n_samples / leaf_size.\n    For a specified ``leaf_size``, a leaf node is guaranteed to\n    satisfy ``leaf_size <= n_points <= 2 * leaf_size``, except in\n    the case that ``n_samples < leaf_size``.\n\nmetric : string or DistanceMetric object\n    the distance metric to use for the tree.  Default='minkowski'\n    with p=2 (that is, a euclidean metric). See the documentation\n    of the DistanceMetric class for a list of available metrics.\n    kd_tree.valid_metrics gives a list of the metrics which\n    are valid for KDTree.\n\nAdditional keywords are passed to the distance metric class.\n\nAttributes\n----------\ndata : memory view\n    The training data\n\nExamples\n--------\nQuery for k-nearest neighbors\n\n    >>> import numpy as np\n    >>> rng = np.random.RandomState(0)\n    >>> X = rng.random_sample((10, 3))  # 10 points in 3 dimensions\n    >>> tree = KDTree(X, leaf_size=2)              # doctest: +SKIP\n    >>> dist, ind = tree.query(X[:1], k=3)                # doctest: +SKIP\n    >>> print(ind)  # indices of 3 closest neighbors\n    [0 3 1]\n    >>> print(dist)  # distances to 3 closest neighbors\n    [ 0.          0.19662693  0.29473397]\n\nPickle and Unpickle a tree.  Note that the state of the tree is saved in the\npickle operation: the tree needs not be rebuilt upon unpickling.\n\n    >>> import numpy as np\n    >>> import pickle\n    >>> rng = np.random.RandomState(0)\n    >>> X = rng.random_sample((10, 3))  # 10 points in 3 dimensions\n    >>> tree = KDTree(X, leaf_size=2)        # doctest: +SKIP\n    >>> s = pickle.dumps(tree)                     # doctest: +SKIP\n    >>> tree_copy = pickle.loads(s)                # doctest: +SKIP\n    >>> dist, ind = tree_copy.query(X[:1], k=3)     # doctest: +SKIP\n    >>> print(ind)  # indices of 3 closest neighbors\n    [0 3 1]\n    >>> print(dist)  # distances to 3 closest neighbors\n    [ 0.          0.19662693  0.29473397]\n\nQuery for neighbors within a given radius\n\n    >>> import numpy as np\n    >>> rng = np.random.RandomState(0)\n    >>> X = rng.random_sample((10, 3))  # 10 points in 3 dimensions\n    >>> tree = KDTree(X, leaf_size=2)     # doctest: +SKIP\n    >>> print(tree.query_radius(X[:1], r=0.3, count_only=True))\n    3\n    >>> ind = tree.query_radius(X[:1], r=0.3)  # doctest: +SKIP\n    >>> print(ind)  # indices of neighbors within distance 0.3\n    [3 0 1]\n\n\nCompute a gaussian kernel density estimate:\n\n    >>> import numpy as np\n    >>> rng = np.random.RandomState(42)\n    >>> X = rng.random_sample((100, 3))\n    >>> tree = KDTree(X)                # doctest: +SKIP\n    >>> tree.kernel_density(X[:3], h=0.1, kernel='gaussian')\n    array([ 6.94114649,  7.83281226,  7.2071716 ])\n\nCompute a two-point auto-correlation function\n\n    >>> import numpy as np\n    >>> rng = np.random.RandomState(0)\n    >>> X = rng.random_sample((30, 3))\n    >>> r = np.linspace(0, 1, 5)\n    >>> tree = KDTree(X)                # doctest: +SKIP\n    >>> tree.two_point_correlation(X, r)\n    array([ 30,  62, 278, 580, 820])\n\n"
    __class__ = KDTree
    def __init__(self, *args, **kwargs):
        "KDTree for fast generalized N-point problems\n\nKDTree(X, leaf_size=40, metric='minkowski', \\**kwargs)\n\nParameters\n----------\nX : array-like, shape = [n_samples, n_features]\n    n_samples is the number of points in the data set, and\n    n_features is the dimension of the parameter space.\n    Note: if X is a C-contiguous array of doubles then data will\n    not be copied. Otherwise, an internal copy will be made.\n\nleaf_size : positive integer (default = 40)\n    Number of points at which to switch to brute-force. Changing\n    leaf_size will not affect the results of a query, but can\n    significantly impact the speed of a query and the memory required\n    to store the constructed tree.  The amount of memory needed to\n    store the tree scales as approximately n_samples / leaf_size.\n    For a specified ``leaf_size``, a leaf node is guaranteed to\n    satisfy ``leaf_size <= n_points <= 2 * leaf_size``, except in\n    the case that ``n_samples < leaf_size``.\n\nmetric : string or DistanceMetric object\n    the distance metric to use for the tree.  Default='minkowski'\n    with p=2 (that is, a euclidean metric). See the documentation\n    of the DistanceMetric class for a list of available metrics.\n    kd_tree.valid_metrics gives a list of the metrics which\n    are valid for KDTree.\n\nAdditional keywords are passed to the distance metric class.\n\nAttributes\n----------\ndata : memory view\n    The training data\n\nExamples\n--------\nQuery for k-nearest neighbors\n\n    >>> import numpy as np\n    >>> rng = np.random.RandomState(0)\n    >>> X = rng.random_sample((10, 3))  # 10 points in 3 dimensions\n    >>> tree = KDTree(X, leaf_size=2)              # doctest: +SKIP\n    >>> dist, ind = tree.query(X[:1], k=3)                # doctest: +SKIP\n    >>> print(ind)  # indices of 3 closest neighbors\n    [0 3 1]\n    >>> print(dist)  # distances to 3 closest neighbors\n    [ 0.          0.19662693  0.29473397]\n\nPickle and Unpickle a tree.  Note that the state of the tree is saved in the\npickle operation: the tree needs not be rebuilt upon unpickling.\n\n    >>> import numpy as np\n    >>> import pickle\n    >>> rng = np.random.RandomState(0)\n    >>> X = rng.random_sample((10, 3))  # 10 points in 3 dimensions\n    >>> tree = KDTree(X, leaf_size=2)        # doctest: +SKIP\n    >>> s = pickle.dumps(tree)                     # doctest: +SKIP\n    >>> tree_copy = pickle.loads(s)                # doctest: +SKIP\n    >>> dist, ind = tree_copy.query(X[:1], k=3)     # doctest: +SKIP\n    >>> print(ind)  # indices of 3 closest neighbors\n    [0 3 1]\n    >>> print(dist)  # distances to 3 closest neighbors\n    [ 0.          0.19662693  0.29473397]\n\nQuery for neighbors within a given radius\n\n    >>> import numpy as np\n    >>> rng = np.random.RandomState(0)\n    >>> X = rng.random_sample((10, 3))  # 10 points in 3 dimensions\n    >>> tree = KDTree(X, leaf_size=2)     # doctest: +SKIP\n    >>> print(tree.query_radius(X[:1], r=0.3, count_only=True))\n    3\n    >>> ind = tree.query_radius(X[:1], r=0.3)  # doctest: +SKIP\n    >>> print(ind)  # indices of neighbors within distance 0.3\n    [3 0 1]\n\n\nCompute a gaussian kernel density estimate:\n\n    >>> import numpy as np\n    >>> rng = np.random.RandomState(42)\n    >>> X = rng.random_sample((100, 3))\n    >>> tree = KDTree(X)                # doctest: +SKIP\n    >>> tree.kernel_density(X[:3], h=0.1, kernel='gaussian')\n    array([ 6.94114649,  7.83281226,  7.2071716 ])\n\nCompute a two-point auto-correlation function\n\n    >>> import numpy as np\n    >>> rng = np.random.RandomState(0)\n    >>> X = rng.random_sample((30, 3))\n    >>> r = np.linspace(0, 1, 5)\n    >>> tree = KDTree(X)                # doctest: +SKIP\n    >>> tree.two_point_correlation(X, r)\n    array([ 30,  62, 278, 580, 820])\n\n"
        pass
    
    @classmethod
    def __init_subclass__(cls):
        'This method is called when a class is subclassed.\n\nThe default implementation does nothing. It may be\noverridden to extend subclasses.\n'
        return None
    
    __pyx_vtable__ = _mod_builtins.PyCapsule()
    @classmethod
    def __subclasshook__(cls, subclass):
        'Abstract classes can override this to customize issubclass().\n\nThis is invoked early on by abc.ABCMeta.__subclasscheck__().\nIt should return True, False or NotImplemented.  If it returns\nNotImplemented, the normal algorithm is used.  Otherwise, it\noverrides the normal algorithm (and the outcome is cached).\n'
        return False
    

class NeighborsHeap(_mod_builtins.object):
    'A max-heap structure to keep track of distances/indices of neighbors\n\n    This implements an efficient pre-allocated set of fixed-size heaps\n    for chasing neighbors, holding both an index and a distance.\n    When any row of the heap is full, adding an additional point will push\n    the furthest point off the heap.\n\n    Parameters\n    ----------\n    n_pts : int\n        the number of heaps to use\n    n_nbrs : int\n        the size of each heap.\n    '
    __class__ = NeighborsHeap
    def __init__(self, *args, **kwargs):
        'A max-heap structure to keep track of distances/indices of neighbors\n\n    This implements an efficient pre-allocated set of fixed-size heaps\n    for chasing neighbors, holding both an index and a distance.\n    When any row of the heap is full, adding an additional point will push\n    the furthest point off the heap.\n\n    Parameters\n    ----------\n    n_pts : int\n        the number of heaps to use\n    n_nbrs : int\n        the size of each heap.\n    '
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
    
    def get_arrays(self):
        'Get the arrays of distances and indices within the heap.\n\n        If sort=True, then simultaneously sort the indices and distances,\n        so the closer points are listed first.\n        '
        pass
    
    def push(self):
        pass
    

NodeData = dtype()
class NodeHeap(_mod_builtins.object):
    'NodeHeap\n\n    This is a min-heap implementation for keeping track of nodes\n    during a breadth-first search.  Unlike the NeighborsHeap above,\n    the NodeHeap does not have a fixed size and must be able to grow\n    as elements are added.\n\n    Internally, the data is stored in a simple binary heap which meets\n    the min heap condition:\n\n        heap[i].val < min(heap[2 * i + 1].val, heap[2 * i + 2].val)\n    '
    __class__ = NodeHeap
    def __init__(self, *args, **kwargs):
        'NodeHeap\n\n    This is a min-heap implementation for keeping track of nodes\n    during a breadth-first search.  Unlike the NeighborsHeap above,\n    the NodeHeap does not have a fixed size and must be able to grow\n    as elements are added.\n\n    Internally, the data is stored in a simple binary heap which meets\n    the min heap condition:\n\n        heap[i].val < min(heap[2 * i + 1].val, heap[2 * i + 2].val)\n    '
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
    

NodeHeapData = dtype()
VALID_METRICS = _mod_builtins.list()
VALID_METRIC_IDS = _mod_builtins.list()
__all__ = _mod_builtins.list()
__builtins__ = {}
__doc__ = None
__file__ = '/home/claudinei/.local/lib/python3.6/site-packages/sklearn/neighbors/kd_tree.cpython-36m-x86_64-linux-gnu.so'
__name__ = 'sklearn.neighbors.kd_tree'
__package__ = 'sklearn.neighbors'
def __pyx_unpickle_Enum():
    pass

__test__ = _mod_builtins.dict()
def check_array(array, accept_sparse, accept_large_sparse, dtype, order, copy, force_all_finite, ensure_2d, allow_nd, ensure_min_samples, ensure_min_features, warn_on_dtype, estimator):
    'Input validation on an array, list, sparse matrix or similar.\n\n    By default, the input is checked to be a non-empty 2D array containing\n    only finite values. If the dtype of the array is object, attempt\n    converting to float, raising on failure.\n\n    Parameters\n    ----------\n    array : object\n        Input object to check / convert.\n\n    accept_sparse : string, boolean or list/tuple of strings (default=False)\n        String[s] representing allowed sparse matrix formats, such as \'csc\',\n        \'csr\', etc. If the input is sparse but not in the allowed format,\n        it will be converted to the first listed format. True allows the input\n        to be any format. False means that a sparse matrix input will\n        raise an error.\n\n    accept_large_sparse : bool (default=True)\n        If a CSR, CSC, COO or BSR sparse matrix is supplied and accepted by\n        accept_sparse, accept_large_sparse=False will cause it to be accepted\n        only if its indices are stored with a 32-bit dtype.\n\n        .. versionadded:: 0.20\n\n    dtype : string, type, list of types or None (default="numeric")\n        Data type of result. If None, the dtype of the input is preserved.\n        If "numeric", dtype is preserved unless array.dtype is object.\n        If dtype is a list of types, conversion on the first type is only\n        performed if the dtype of the input is not in the list.\n\n    order : \'F\', \'C\' or None (default=None)\n        Whether an array will be forced to be fortran or c-style.\n        When order is None (default), then if copy=False, nothing is ensured\n        about the memory layout of the output array; otherwise (copy=True)\n        the memory layout of the returned array is kept as close as possible\n        to the original array.\n\n    copy : boolean (default=False)\n        Whether a forced copy will be triggered. If copy=False, a copy might\n        be triggered by a conversion.\n\n    force_all_finite : boolean or \'allow-nan\', (default=True)\n        Whether to raise an error on np.inf and np.nan in array. The\n        possibilities are:\n\n        - True: Force all values of array to be finite.\n        - False: accept both np.inf and np.nan in array.\n        - \'allow-nan\': accept only np.nan values in array. Values cannot\n          be infinite.\n\n        For object dtyped data, only np.nan is checked and not np.inf.\n\n        .. versionadded:: 0.20\n           ``force_all_finite`` accepts the string ``\'allow-nan\'``.\n\n    ensure_2d : boolean (default=True)\n        Whether to raise a value error if array is not 2D.\n\n    allow_nd : boolean (default=False)\n        Whether to allow array.ndim > 2.\n\n    ensure_min_samples : int (default=1)\n        Make sure that the array has a minimum number of samples in its first\n        axis (rows for a 2D array). Setting to 0 disables this check.\n\n    ensure_min_features : int (default=1)\n        Make sure that the 2D array has some minimum number of features\n        (columns). The default value of 1 rejects empty datasets.\n        This check is only enforced when the input data has effectively 2\n        dimensions or is originally 1D and ``ensure_2d`` is True. Setting to 0\n        disables this check.\n\n    warn_on_dtype : boolean or None, optional (default=None)\n        Raise DataConversionWarning if the dtype of the input data structure\n        does not match the requested dtype, causing a memory copy.\n\n        .. deprecated:: 0.21\n            ``warn_on_dtype`` is deprecated in version 0.21 and will be\n            removed in 0.23.\n\n    estimator : str or estimator instance (default=None)\n        If passed, include the name of the estimator in warning messages.\n\n    Returns\n    -------\n    array_converted : object\n        The converted and validated array.\n    '
    pass

def get_valid_metric_ids():
    "Given an iterable of metric class names or class identifiers,\n    return a list of metric IDs which map to those classes.\n\n    Example:\n    >>> L = get_valid_metric_ids([EuclideanDistance, 'ManhattanDistance'])\n    >>> sorted(L)\n    ['cityblock', 'euclidean', 'l1', 'l2', 'manhattan']\n    "
    pass

def kernel_norm():
    "Given a string specification of a kernel, compute the normalization.\n\n    Parameters\n    ----------\n    h : float\n        the bandwidth of the kernel\n    d : int\n        the dimension of the space in which the kernel norm is computed\n    kernel : string\n        The kernel identifier.  Must be one of\n        ['gaussian'|'tophat'|'epanechnikov'|\n         'exponential'|'linear'|'cosine']\n    return_log : boolean\n        if True, return the log of the kernel norm.  Otherwise, return the\n        kernel norm.\n    Returns\n    -------\n    knorm or log_knorm : float\n        the kernel norm or logarithm of the kernel norm.\n    "
    pass

def load_heap():
    'test fully loading the heap'
    pass

def newObj():
    pass

def nodeheap_sort():
    'In-place reverse sort of vals using NodeHeap'
    pass

offsets = _mod_builtins.list()
def simultaneous_sort():
    'In-place simultaneous sort the given row of the arrays\n\n    This python wrapper exists primarily to enable unit testing\n    of the _simultaneous_sort C routine.\n    '
    pass

