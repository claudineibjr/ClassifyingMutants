import __builtin__ as _mod___builtin__

__doc__ = 'Access to the Unix group database.\n\nGroup entries are reported as 4-tuples containing the following fields\nfrom the group database, in order:\n\n  gr_name   - name of the group\n  gr_passwd - group password (encrypted); often empty\n  gr_gid    - numeric ID of the group\n  gr_mem    - list of members\n\nThe gid is an integer, name and password are strings.  (Note that most\nusers are not explicitly listed as members of the groups they are in\naccording to the password database.  Check both databases to get\ncomplete membership information.)'
__name__ = 'grp'
__package__ = None
def getgrall():
    "getgrall() -> list of tuples\nReturn a list of all available group entries, in arbitrary order.\nAn entry whose name starts with '+' or '-' represents an instruction\nto use YP/NIS and may not be accessible via getgrnam or getgrgid."
    return list()

def getgrgid(id):
    'getgrgid(id) -> (gr_name,gr_passwd,gr_gid,gr_mem)\nReturn the group database entry for the given numeric group ID.  If\nid is not valid, raise KeyError.'
    return tuple()

def getgrnam(name):
    'getgrnam(name) -> (gr_name,gr_passwd,gr_gid,gr_mem)\nReturn the group database entry for the given group name.  If\nname is not valid, raise KeyError.'
    return tuple()

class struct_group(_mod___builtin__.object):
    'grp.struct_group: Results from getgr*() routines.\n\nThis object may be accessed either as a tuple of\n  (gr_name,gr_passwd,gr_gid,gr_mem)\nor via the object attributes as named in the above tuple.\n'
    def __add__(self):
        'x.__add__(y) <==> x+y'
        return struct_group()
    
    __class__ = struct_group
    def __contains__(self, value):
        'x.__contains__(y) <==> y in x'
        return False
    
    def __eq__(self):
        'x.__eq__(y) <==> x==y'
        return False
    
    def __ge__(self):
        'x.__ge__(y) <==> x>=y'
        return False
    
    def __getitem__(self, index):
        'x.__getitem__(y) <==> x[y]'
        pass
    
    def __getslice__(self):
        'x.__getslice__(i, j) <==> x[i:j]\n           \n           Use of negative indices is not supported.'
        return struct_group()
    
    def __gt__(self):
        'x.__gt__(y) <==> x>y'
        return False
    
    def __hash__(self):
        'x.__hash__() <==> hash(x)'
        return 0
    
    def __init__(self):
        'x.__init__(...) initializes x; see help(type(x)) for signature'
        pass
    
    def __le__(self):
        'x.__le__(y) <==> x<=y'
        return False
    
    def __len__(self):
        'x.__len__() <==> len(x)'
        return 0
    
    def __lt__(self):
        'x.__lt__(y) <==> x<y'
        return False
    
    def __mul__(self):
        'x.__mul__(n) <==> x*n'
        return struct_group()
    
    def __ne__(self):
        'x.__ne__(y) <==> x!=y'
        return False
    
    def __reduce__(self):
        return ''; return ()
    
    def __repr__(self):
        'x.__repr__() <==> repr(x)'
        return ''
    
    def __rmul__(self):
        'x.__rmul__(n) <==> n*x'
        return struct_group()
    
    @classmethod
    def __subclasshook__(cls, subclass):
        'Abstract classes can override this to customize issubclass().\n\nThis is invoked early on by abc.ABCMeta.__subclasscheck__().\nIt should return True, False or NotImplemented.  If it returns\nNotImplemented, the normal algorithm is used.  Otherwise, it\noverrides the normal algorithm (and the outcome is cached).\n'
        return False
    
    @property
    def gr_gid(self):
        'group id'
        pass
    
    @property
    def gr_mem(self):
        'group members'
        pass
    
    @property
    def gr_name(self):
        'group name'
        pass
    
    @property
    def gr_passwd(self):
        'password'
        pass
    
    n_fields = 4
    n_sequence_fields = 4
    n_unnamed_fields = 0

