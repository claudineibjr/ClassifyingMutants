import builtins as _mod_builtins
import gi as _mod_gi
import gobject as _mod_gobject

ArgInfo = _mod_gi.ArgInfo
class ArrayType(_mod_builtins.type):
    ARRAY = 1
    BYTE_ARRAY = 3
    C = 0
    PTR_ARRAY = 2
    __base__ = _mod_builtins.type
    __bases__ = ()
    __basicsize__ = 864
    __class__ = ArrayType
    __dict__ = {}
    __dictoffset__ = 264
    __flags__ = 2148292097
    def __init__(self, *args, **kwargs):
        pass
    
    @classmethod
    def __init_subclass__(cls):
        'This method is called when a class is subclassed.\n\nThe default implementation does nothing. It may be\noverridden to extend subclasses.\n'
        return None
    
    __module__ = 'gi._gi'
    __mro__ = ()
    __name__ = 'ArrayType'
    @classmethod
    def __prepare__(cls, name, bases, **kwds):
        '__prepare__() -> dict\nused to create the namespace for the class statement'
        return dict()
    
    __qualname__ = 'ArrayType'
    @classmethod
    def __subclasshook__(cls, subclass):
        'Abstract classes can override this to customize issubclass().\n\nThis is invoked early on by abc.ABCMeta.__subclasscheck__().\nIt should return True, False or NotImplemented.  If it returns\nNotImplemented, the normal algorithm is used.  Otherwise, it\noverrides the normal algorithm (and the outcome is cached).\n'
        return False
    
    __weakrefoffset__ = 368

BaseInfo = _mod_gi.BaseInfo
Boxed = _mod_gi.Boxed
CCallback = _mod_gi.CCallback
CallableInfo = _mod_gi.CallableInfo
CallbackInfo = _mod_gi.CallbackInfo
ConstantInfo = _mod_gi.ConstantInfo
class Direction(_mod_builtins.type):
    IN = 0
    INOUT = 2
    OUT = 1
    __base__ = _mod_builtins.type
    __bases__ = ()
    __basicsize__ = 864
    __class__ = Direction
    __dict__ = {}
    __dictoffset__ = 264
    __flags__ = 2148292097
    def __init__(self, *args, **kwargs):
        pass
    
    @classmethod
    def __init_subclass__(cls):
        'This method is called when a class is subclassed.\n\nThe default implementation does nothing. It may be\noverridden to extend subclasses.\n'
        return None
    
    __module__ = 'gi._gi'
    __mro__ = ()
    __name__ = 'Direction'
    @classmethod
    def __prepare__(cls, name, bases, **kwds):
        '__prepare__() -> dict\nused to create the namespace for the class statement'
        return dict()
    
    __qualname__ = 'Direction'
    @classmethod
    def __subclasshook__(cls, subclass):
        'Abstract classes can override this to customize issubclass().\n\nThis is invoked early on by abc.ABCMeta.__subclasscheck__().\nIt should return True, False or NotImplemented.  If it returns\nNotImplemented, the normal algorithm is used.  Otherwise, it\noverrides the normal algorithm (and the outcome is cached).\n'
        return False
    
    __weakrefoffset__ = 368

EnumInfo = _mod_gi.EnumInfo
ErrorDomainInfo = _mod_gi.ErrorDomainInfo
FieldInfo = _mod_gi.FieldInfo
class FieldInfoFlags(_mod_builtins.type):
    IS_READABLE = 1
    IS_WRITABLE = 2
    __base__ = _mod_builtins.type
    __bases__ = ()
    __basicsize__ = 864
    __class__ = FieldInfoFlags
    __dict__ = {}
    __dictoffset__ = 264
    __flags__ = 2148292097
    def __init__(self, *args, **kwargs):
        pass
    
    @classmethod
    def __init_subclass__(cls):
        'This method is called when a class is subclassed.\n\nThe default implementation does nothing. It may be\noverridden to extend subclasses.\n'
        return None
    
    __module__ = 'gi._gi'
    __mro__ = ()
    __name__ = 'FieldInfoFlags'
    @classmethod
    def __prepare__(cls, name, bases, **kwds):
        '__prepare__() -> dict\nused to create the namespace for the class statement'
        return dict()
    
    __qualname__ = 'FieldInfoFlags'
    @classmethod
    def __subclasshook__(cls, subclass):
        'Abstract classes can override this to customize issubclass().\n\nThis is invoked early on by abc.ABCMeta.__subclasscheck__().\nIt should return True, False or NotImplemented.  If it returns\nNotImplemented, the normal algorithm is used.  Otherwise, it\noverrides the normal algorithm (and the outcome is cached).\n'
        return False
    
    __weakrefoffset__ = 368

FunctionInfo = _mod_gi.FunctionInfo
class FunctionInfoFlags(_mod_builtins.type):
    IS_CONSTRUCTOR = 2
    IS_GETTER = 4
    IS_METHOD = 1
    IS_SETTER = 8
    THROWS = 32
    WRAPS_VFUNC = 16
    __base__ = _mod_builtins.type
    __bases__ = ()
    __basicsize__ = 864
    __class__ = FunctionInfoFlags
    __dict__ = {}
    __dictoffset__ = 264
    __flags__ = 2148292097
    def __init__(self, *args, **kwargs):
        pass
    
    @classmethod
    def __init_subclass__(cls):
        'This method is called when a class is subclassed.\n\nThe default implementation does nothing. It may be\noverridden to extend subclasses.\n'
        return None
    
    __module__ = 'gi._gi'
    __mro__ = ()
    __name__ = 'FunctionInfoFlags'
    @classmethod
    def __prepare__(cls, name, bases, **kwds):
        '__prepare__() -> dict\nused to create the namespace for the class statement'
        return dict()
    
    __qualname__ = 'FunctionInfoFlags'
    @classmethod
    def __subclasshook__(cls, subclass):
        'Abstract classes can override this to customize issubclass().\n\nThis is invoked early on by abc.ABCMeta.__subclasscheck__().\nIt should return True, False or NotImplemented.  If it returns\nNotImplemented, the normal algorithm is used.  Otherwise, it\noverrides the normal algorithm (and the outcome is cached).\n'
        return False
    
    __weakrefoffset__ = 368

GBoxed = _mod_gobject.GBoxed
GEnum = _mod_gobject.GEnum
GFlags = _mod_gobject.GFlags
GInterface = _mod_gobject.GInterface
class GObject(_mod_builtins.object):
    'Object GObject\n\nSignals from GObject:\n  notify (GParam)\n\n'
    __class__ = GObject
    def __copy__(self):
        pass
    
    def __deepcopy__(self):
        pass
    
    def __delattr__(self, name):
        'Implement delattr(self, name).'
        return None
    
    __dict__ = {}
    def __eq__(self, value):
        'Return self==value.'
        return False
    
    __gdoc__ = 'Object GObject\n\nSignals from GObject:\n  notify (GParam)\n\n'
    def __ge__(self, value):
        'Return self>=value.'
        return False
    
    @property
    def __gpointer__(self):
        pass
    
    @property
    def __grefcount__(self):
        pass
    
    def __gt__(self, value):
        'Return self>value.'
        return False
    
    __gtype__ = _mod_gobject.GType()
    def __hash__(self):
        'Return hash(self).'
        return 0
    
    def __init__(self, *args, **kwargs):
        'Object GObject\n\nSignals from GObject:\n  notify (GParam)\n\n'
        pass
    
    @classmethod
    def __init_subclass__(cls):
        'This method is called when a class is subclassed.\n\nThe default implementation does nothing. It may be\noverridden to extend subclasses.\n'
        return None
    
    def __le__(self, value):
        'Return self<=value.'
        return False
    
    def __lt__(self, value):
        'Return self<value.'
        return False
    
    __module__ = 'gi._gi'
    def __ne__(self, value):
        'Return self!=value.'
        return False
    
    def __repr__(self):
        'Return repr(self).'
        return ''
    
    def __setattr__(self, name, value):
        'Implement setattr(self, name, value).'
        return None
    
    @classmethod
    def __subclasshook__(cls, subclass):
        'Abstract classes can override this to customize issubclass().\n\nThis is invoked early on by abc.ABCMeta.__subclasscheck__().\nIt should return True, False or NotImplemented.  If it returns\nNotImplemented, the normal algorithm is used.  Otherwise, it\noverrides the normal algorithm (and the outcome is cached).\n'
        return False
    
    def bind_property(self):
        pass
    
    def chain(self):
        pass
    
    def connect(self):
        pass
    
    def connect_after(self):
        pass
    
    def connect_object(self):
        pass
    
    def connect_object_after(self):
        pass
    
    def disconnect_by_func(self):
        pass
    
    def emit(self):
        pass
    
    def get_properties(self):
        pass
    
    def get_property(self):
        pass
    
    def handler_block_by_func(self):
        pass
    
    def handler_unblock_by_func(self):
        pass
    
    props = GProps()
    def set_properties(self):
        pass
    
    def set_property(self):
        pass
    
    def weak_ref(self):
        pass
    

class GObjectWeakRef(_mod_builtins.object):
    'A GObject weak reference'
    def __call__(self, *args, **kwargs):
        'Call self as a function.'
        pass
    
    __class__ = GObjectWeakRef
    def __init__(self, *args, **kwargs):
        'A GObject weak reference'
        pass
    
    @classmethod
    def __init_subclass__(cls):
        'This method is called when a class is subclassed.\n\nThe default implementation does nothing. It may be\noverridden to extend subclasses.\n'
        return None
    
    @classmethod
    def __subclasshook__(cls, subclass):
        'Abstract classes can override this to customize issubclass().\n\nThis is invoked early on by abc.ABCMeta.__subclasscheck__().\nIt should return True, False or NotImplemented.  If it returns\nNotImplemented, the normal algorithm is used.  Otherwise, it\noverrides the normal algorithm (and the outcome is cached).\n'
        return False
    
    def unref(self):
        pass
    

GParamSpec = _mod_gobject.GParamSpec
GPointer = _mod_gobject.GPointer
GType = _mod_gobject.GType
G_MAXDOUBLE = 1.7976931348623157e+308
G_MAXFLOAT = 3.4028234663852886e+38
G_MAXINT = 2147483647
G_MAXLONG = 9223372036854775807
G_MAXOFFSET = 9223372036854775807
G_MAXSHORT = 32767
G_MAXSIZE = 18446744073709551615
G_MAXSSIZE = 9223372036854775807
G_MAXUINT = 4294967295
G_MAXULONG = 18446744073709551615
G_MAXUSHORT = 65535
G_MINDOUBLE = 2.2250738585072014e-308
G_MINFLOAT = 1.1754943508222875e-38
G_MININT = -2147483648
G_MINLONG = -9223372036854775808
G_MINOFFSET = -9223372036854775808
G_MINSHORT = -32768
G_MINSSIZE = -9223372036854775808
class InfoType(_mod_builtins.type):
    ARG = 17
    BOXED = 4
    CALLBACK = 2
    CONSTANT = 9
    ENUM = 5
    FIELD = 16
    FLAGS = 6
    FUNCTION = 1
    INTERFACE = 8
    INVALID = 0
    INVALID_0 = 10
    OBJECT = 7
    PROPERTY = 15
    SIGNAL = 13
    STRUCT = 3
    TYPE = 18
    UNION = 11
    UNRESOLVED = 19
    VALUE = 12
    VFUNC = 14
    __base__ = _mod_builtins.type
    __bases__ = ()
    __basicsize__ = 864
    __class__ = InfoType
    __dict__ = {}
    __dictoffset__ = 264
    __flags__ = 2148292097
    def __init__(self, *args, **kwargs):
        pass
    
    @classmethod
    def __init_subclass__(cls):
        'This method is called when a class is subclassed.\n\nThe default implementation does nothing. It may be\noverridden to extend subclasses.\n'
        return None
    
    __module__ = 'gi._gi'
    __mro__ = ()
    __name__ = 'InfoType'
    @classmethod
    def __prepare__(cls, name, bases, **kwds):
        '__prepare__() -> dict\nused to create the namespace for the class statement'
        return dict()
    
    __qualname__ = 'InfoType'
    @classmethod
    def __subclasshook__(cls, subclass):
        'Abstract classes can override this to customize issubclass().\n\nThis is invoked early on by abc.ABCMeta.__subclasscheck__().\nIt should return True, False or NotImplemented.  If it returns\nNotImplemented, the normal algorithm is used.  Otherwise, it\noverrides the normal algorithm (and the outcome is cached).\n'
        return False
    
    __weakrefoffset__ = 368

InterfaceInfo = _mod_builtins.InterfaceInfo
ObjectInfo = _mod_builtins.ObjectInfo
class OptionContext(_mod_builtins.object):
    __class__ = OptionContext
    def __eq__(self, value):
        'Return self==value.'
        return False
    
    def __ge__(self, value):
        'Return self>=value.'
        return False
    
    def __gt__(self, value):
        'Return self>value.'
        return False
    
    __hash__ = None
    def __init__(self, *args, **kwargs):
        pass
    
    @classmethod
    def __init_subclass__(cls):
        'This method is called when a class is subclassed.\n\nThe default implementation does nothing. It may be\noverridden to extend subclasses.\n'
        return None
    
    def __le__(self, value):
        'Return self<=value.'
        return False
    
    def __lt__(self, value):
        'Return self<value.'
        return False
    
    def __ne__(self, value):
        'Return self!=value.'
        return False
    
    @classmethod
    def __subclasshook__(cls, subclass):
        'Abstract classes can override this to customize issubclass().\n\nThis is invoked early on by abc.ABCMeta.__subclasscheck__().\nIt should return True, False or NotImplemented.  If it returns\nNotImplemented, the normal algorithm is used.  Otherwise, it\noverrides the normal algorithm (and the outcome is cached).\n'
        return False
    
    def _get_context(self):
        pass
    
    def add_group(self):
        pass
    
    def get_help_enabled(self):
        pass
    
    def get_ignore_unknown_options(self):
        pass
    
    def get_main_group(self):
        pass
    
    def parse(self):
        pass
    
    def set_help_enabled(self):
        pass
    
    def set_ignore_unknown_options(self):
        pass
    
    def set_main_group(self):
        pass
    

class OptionGroup(_mod_builtins.object):
    __class__ = OptionGroup
    def __eq__(self, value):
        'Return self==value.'
        return False
    
    def __ge__(self, value):
        'Return self>=value.'
        return False
    
    def __gt__(self, value):
        'Return self>value.'
        return False
    
    __hash__ = None
    def __init__(self, *args, **kwargs):
        pass
    
    @classmethod
    def __init_subclass__(cls):
        'This method is called when a class is subclassed.\n\nThe default implementation does nothing. It may be\noverridden to extend subclasses.\n'
        return None
    
    def __le__(self, value):
        'Return self<=value.'
        return False
    
    def __lt__(self, value):
        'Return self<value.'
        return False
    
    def __ne__(self, value):
        'Return self!=value.'
        return False
    
    @classmethod
    def __subclasshook__(cls, subclass):
        'Abstract classes can override this to customize issubclass().\n\nThis is invoked early on by abc.ABCMeta.__subclasscheck__().\nIt should return True, False or NotImplemented.  If it returns\nNotImplemented, the normal algorithm is used.  Otherwise, it\noverrides the normal algorithm (and the outcome is cached).\n'
        return False
    
    def add_entries(self):
        pass
    
    def set_translation_domain(self):
        pass
    

PARAM_READWRITE = 3
class Pid(_mod_builtins.int):
    __class__ = Pid
    def __init__(self, *args, **kwargs):
        pass
    
    @classmethod
    def __init_subclass__(cls):
        'This method is called when a class is subclassed.\n\nThe default implementation does nothing. It may be\noverridden to extend subclasses.\n'
        return None
    
    @classmethod
    def __subclasshook__(cls, subclass):
        'Abstract classes can override this to customize issubclass().\n\nThis is invoked early on by abc.ABCMeta.__subclasscheck__().\nIt should return True, False or NotImplemented.  If it returns\nNotImplemented, the normal algorithm is used.  Otherwise, it\noverrides the normal algorithm (and the outcome is cached).\n'
        return False
    
    def close(self):
        pass
    
    @classmethod
    def from_bytes(cls):
        "int.from_bytes(bytes, byteorder, *, signed=False) -> int\n\nReturn the integer represented by the given array of bytes.\n\nThe bytes argument must be a bytes-like object (e.g. bytes or bytearray).\n\nThe byteorder argument determines the byte order used to represent the\ninteger.  If byteorder is 'big', the most significant byte is at the\nbeginning of the byte array.  If byteorder is 'little', the most\nsignificant byte is at the end of the byte array.  To request the native\nbyte order of the host system, use `sys.byteorder' as the byte order value.\n\nThe signed keyword-only argument indicates whether two's complement is\nused to represent the integer."
        return 1
    

PropertyInfo = _mod_gi.PropertyInfo
PyGIDeprecationWarning = _mod_gi.PyGIDeprecationWarning
PyGIWarning = _mod_gi.PyGIWarning
RegisteredTypeInfo = _mod_gi.RegisteredTypeInfo
Repository = _mod_gi.Repository
RepositoryError = _mod_gi.RepositoryError
class ResultTuple(_mod_builtins.tuple):
    __class__ = ResultTuple
    def __dir__(self):
        return ['']
    
    def __getattribute__(self, name):
        'Return getattr(self, name).'
        pass
    
    def __init__(self, *args, **kwargs):
        pass
    
    @classmethod
    def __init_subclass__(cls):
        'This method is called when a class is subclassed.\n\nThe default implementation does nothing. It may be\noverridden to extend subclasses.\n'
        return None
    
    def __reduce__(self):
        return ''; return ()
    
    def __repr__(self):
        'Return repr(self).'
        return ''
    
    @classmethod
    def __subclasshook__(cls, subclass):
        'Abstract classes can override this to customize issubclass().\n\nThis is invoked early on by abc.ABCMeta.__subclasscheck__().\nIt should return True, False or NotImplemented.  If it returns\nNotImplemented, the normal algorithm is used.  Otherwise, it\noverrides the normal algorithm (and the outcome is cached).\n'
        return False
    
    @classmethod
    def _new_type(cls):
        pass
    

SIGNAL_RUN_FIRST = 1
class ScopeType(_mod_builtins.type):
    ASYNC = 2
    CALL = 1
    INVALID = 0
    NOTIFIED = 3
    __base__ = _mod_builtins.type
    __bases__ = ()
    __basicsize__ = 864
    __class__ = ScopeType
    __dict__ = {}
    __dictoffset__ = 264
    __flags__ = 2148292097
    def __init__(self, *args, **kwargs):
        pass
    
    @classmethod
    def __init_subclass__(cls):
        'This method is called when a class is subclassed.\n\nThe default implementation does nothing. It may be\noverridden to extend subclasses.\n'
        return None
    
    __module__ = 'gi._gi'
    __mro__ = ()
    __name__ = 'ScopeType'
    @classmethod
    def __prepare__(cls, name, bases, **kwds):
        '__prepare__() -> dict\nused to create the namespace for the class statement'
        return dict()
    
    __qualname__ = 'ScopeType'
    @classmethod
    def __subclasshook__(cls, subclass):
        'Abstract classes can override this to customize issubclass().\n\nThis is invoked early on by abc.ABCMeta.__subclasscheck__().\nIt should return True, False or NotImplemented.  If it returns\nNotImplemented, the normal algorithm is used.  Otherwise, it\noverrides the normal algorithm (and the outcome is cached).\n'
        return False
    
    __weakrefoffset__ = 368

SignalInfo = _mod_gi.SignalInfo
Struct = _mod_gi.Struct
StructInfo = _mod_builtins.StructInfo
TYPE_GSTRING = _mod_gobject.GType()
TYPE_INVALID = _mod_gobject.GType()
class Transfer(_mod_builtins.type):
    CONTAINER = 1
    EVERYTHING = 2
    NOTHING = 0
    __base__ = _mod_builtins.type
    __bases__ = ()
    __basicsize__ = 864
    __class__ = Transfer
    __dict__ = {}
    __dictoffset__ = 264
    __flags__ = 2148292097
    def __init__(self, *args, **kwargs):
        pass
    
    @classmethod
    def __init_subclass__(cls):
        'This method is called when a class is subclassed.\n\nThe default implementation does nothing. It may be\noverridden to extend subclasses.\n'
        return None
    
    __module__ = 'gi._gi'
    __mro__ = ()
    __name__ = 'Transfer'
    @classmethod
    def __prepare__(cls, name, bases, **kwds):
        '__prepare__() -> dict\nused to create the namespace for the class statement'
        return dict()
    
    __qualname__ = 'Transfer'
    @classmethod
    def __subclasshook__(cls, subclass):
        'Abstract classes can override this to customize issubclass().\n\nThis is invoked early on by abc.ABCMeta.__subclasscheck__().\nIt should return True, False or NotImplemented.  If it returns\nNotImplemented, the normal algorithm is used.  Otherwise, it\noverrides the normal algorithm (and the outcome is cached).\n'
        return False
    
    __weakrefoffset__ = 368

TypeInfo = _mod_gi.TypeInfo
class TypeTag(_mod_builtins.type):
    ARRAY = 15
    BOOLEAN = 1
    DOUBLE = 11
    ERROR = 20
    FILENAME = 14
    FLOAT = 10
    GHASH = 19
    GLIST = 17
    GSLIST = 18
    GTYPE = 12
    INT16 = 4
    INT32 = 6
    INT64 = 8
    INT8 = 2
    INTERFACE = 16
    UINT16 = 5
    UINT32 = 7
    UINT64 = 9
    UINT8 = 3
    UNICHAR = 21
    UTF8 = 13
    VOID = 0
    __base__ = _mod_builtins.type
    __bases__ = ()
    __basicsize__ = 864
    __class__ = TypeTag
    __dict__ = {}
    __dictoffset__ = 264
    __flags__ = 2148292097
    def __init__(self, *args, **kwargs):
        pass
    
    @classmethod
    def __init_subclass__(cls):
        'This method is called when a class is subclassed.\n\nThe default implementation does nothing. It may be\noverridden to extend subclasses.\n'
        return None
    
    __module__ = 'gi._gi'
    __mro__ = ()
    __name__ = 'TypeTag'
    @classmethod
    def __prepare__(cls, name, bases, **kwds):
        '__prepare__() -> dict\nused to create the namespace for the class statement'
        return dict()
    
    __qualname__ = 'TypeTag'
    @classmethod
    def __subclasshook__(cls, subclass):
        'Abstract classes can override this to customize issubclass().\n\nThis is invoked early on by abc.ABCMeta.__subclasscheck__().\nIt should return True, False or NotImplemented.  If it returns\nNotImplemented, the normal algorithm is used.  Otherwise, it\noverrides the normal algorithm (and the outcome is cached).\n'
        return False
    
    __weakrefoffset__ = 368

UnionInfo = _mod_gi.UnionInfo
UnresolvedInfo = _mod_gi.UnresolvedInfo
VFuncInfo = _mod_gi.VFuncInfo
class VFuncInfoFlags(_mod_builtins.type):
    CHAIN_UP = 1
    NOT_OVERRIDE = 4
    OVERRIDE = 2
    __base__ = _mod_builtins.type
    __bases__ = ()
    __basicsize__ = 864
    __class__ = VFuncInfoFlags
    __dict__ = {}
    __dictoffset__ = 264
    __flags__ = 2148292097
    def __init__(self, *args, **kwargs):
        pass
    
    @classmethod
    def __init_subclass__(cls):
        'This method is called when a class is subclassed.\n\nThe default implementation does nothing. It may be\noverridden to extend subclasses.\n'
        return None
    
    __module__ = 'gi._gi'
    __mro__ = ()
    __name__ = 'VFuncInfoFlags'
    @classmethod
    def __prepare__(cls, name, bases, **kwds):
        '__prepare__() -> dict\nused to create the namespace for the class statement'
        return dict()
    
    __qualname__ = 'VFuncInfoFlags'
    @classmethod
    def __subclasshook__(cls, subclass):
        'Abstract classes can override this to customize issubclass().\n\nThis is invoked early on by abc.ABCMeta.__subclasscheck__().\nIt should return True, False or NotImplemented.  If it returns\nNotImplemented, the normal algorithm is used.  Otherwise, it\noverrides the normal algorithm (and the outcome is cached).\n'
        return False
    
    __weakrefoffset__ = 368

ValueInfo = _mod_gi.ValueInfo
Warning = _mod_gobject.Warning
_API = _mod_builtins.PyCapsule()
_PyGObject_API = _mod_builtins.PyCapsule()
__doc__ = None
__file__ = '/usr/lib/python3/dist-packages/gi/_gi.cpython-36m-x86_64-linux-gnu.so'
__name__ = 'gi._gi'
__package__ = 'gi._gi'
def _gvalue_get():
    pass

def _gvalue_set():
    pass

def _install_metaclass():
    pass

def add_emission_hook():
    pass

def enum_add():
    pass

def enum_register_new_gtype_and_add():
    pass

features = _mod_builtins.dict()
def flags_add():
    pass

def flags_register_new_gtype_and_add():
    pass

def hook_up_vfunc_implementation():
    pass

def io_channel_read():
    pass

def list_properties():
    pass

def new():
    pass

pygobject_version = _mod_builtins.tuple()
def register_interface_info():
    pass

def require_foreign():
    pass

def signal_accumulator_true_handled():
    pass

def signal_new():
    pass

def source_new():
    pass

def source_set_callback():
    pass

def spawn_async():
    'spawn_async(argv, envp=None, working_directory=None,\n            flags=0, child_setup=None, user_data=None,\n            standard_input=None, standard_output=None,\n            standard_error=None) -> (pid, stdin, stdout, stderr)\n\nExecute a child program asynchronously within a glib.MainLoop()\nSee the reference manual for a complete reference.\n'
    pass

def type_from_name():
    pass

def type_is_a():
    pass

def type_name():
    pass

def type_register():
    pass

def variant_type_from_string():
    pass

