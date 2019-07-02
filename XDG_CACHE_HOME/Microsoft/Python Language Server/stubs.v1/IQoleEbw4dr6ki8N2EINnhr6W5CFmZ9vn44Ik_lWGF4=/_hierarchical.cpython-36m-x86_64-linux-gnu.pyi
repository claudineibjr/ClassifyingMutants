import builtins as _mod_builtins
import numpy as _mod_numpy

DTYPE = _mod_numpy.float64
ITYPE = _mod_numpy.int64
class UnionFind(_mod_builtins.object):
    __class__ = UnionFind
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
    

class WeightedEdge(_mod_builtins.object):
    __class__ = WeightedEdge
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
    
    def __reduce__(self):
        return ''; return ()
    
    def __repr__(self):
        'Return repr(self).'
        return ''
    
    def __setstate__(self, state):
        return None
    
    @classmethod
    def __subclasshook__(cls, subclass):
        'Abstract classes can override this to customize issubclass().\n\nThis is invoked early on by abc.ABCMeta.__subclasscheck__().\nIt should return True, False or NotImplemented.  If it returns\nNotImplemented, the normal algorithm is used.  Otherwise, it\noverrides the normal algorithm (and the outcome is cached).\n'
        return False
    
    @property
    def a(self):
        pass
    
    @property
    def b(self):
        pass
    
    @property
    def weight(self):
        pass
    

__builtins__ = {}
__doc__ = None
__file__ = '/home/claudinei/.local/lib/python3.6/site-packages/sklearn/cluster/_hierarchical.cpython-36m-x86_64-linux-gnu.so'
__name__ = 'sklearn.cluster._hierarchical'
__package__ = 'sklearn.cluster'
def __pyx_unpickle_Enum():
    pass

def __pyx_unpickle_UnionFind():
    pass

def __pyx_unpickle_WeightedEdge():
    pass

__test__ = _mod_builtins.dict()
def _get_parents():
    "Returns the heads of the given nodes, as defined by parents.\n\n    Modifies 'heads' and 'not_visited' in-place.\n\n    Parameters\n    ----------\n    nodes : list of integers\n        The nodes to start from\n    heads : list of integers\n        A list to hold the results (modified inplace)\n    parents : array of integers\n        The parent structure defining the tree\n    not_visited\n        The tree nodes to consider (modified inplace)\n\n    "
    pass

def _hc_get_descendent():
    '\n    Function returning all the descendent leaves of a set of nodes in the tree.\n\n    Parameters\n    ----------\n    node : integer\n        The node for which we want the descendents.\n\n    children : list of pairs, length n_nodes\n        The children of each non-leaf node. Values less than `n_samples` refer\n        to leaves of the tree. A greater value `i` indicates a node with\n        children `children[i - n_samples]`.\n\n    n_leaves : integer\n        Number of leaves.\n\n    Returns\n    -------\n    descendent : list of int\n    '
    pass

def _single_linkage_label():
    '\n    Convert an linkage array or MST to a tree by labelling clusters at merges.\n    This is done by using a Union find structure to keep track of merges\n    efficiently. This is the private version of the function that assumes that\n    ``L`` has been properly validated. See ``single_linkage_label`` for the\n    user facing version of this function.\n\n    Parameters\n    ----------\n    L: array of shape (n_samples - 1, 3)\n        The linkage array or MST where each row specifies two samples\n        to be merged and a distance or weight at which the merge occurs. This\n         array is assumed to be sorted by the distance/weight.\n\n    Returns\n    -------\n    A tree in the format used by scipy.cluster.hierarchy.\n    '
    pass

def average_merge():
    'Merge two IntFloatDicts with the average strategy: when the \n    same key is present in the two dicts, the weighted average of the two \n    values is used.\n\n    Parameters\n    ==========\n    a, b : IntFloatDict object\n        The IntFloatDicts to merge\n    mask : ndarray array of dtype integer and of dimension 1\n        a mask for keys to ignore: if not mask[key] the corresponding key\n        is skipped in the output dictionary\n    n_a, n_b : float\n        n_a and n_b are weights for a and b for the merge strategy.\n        They are used for a weighted mean.\n\n    Returns\n    =======\n    out : IntFloatDict object\n        The IntFloatDict resulting from the merge\n    '
    pass

def compute_ward_dist():
    pass

def hc_get_heads():
    "Returns the heads of the forest, as defined by parents.\n\n    Parameters\n    ----------\n    parents : array of integers\n        The parent structure defining the forest (ensemble of trees)\n    copy : boolean\n        If copy is False, the input 'parents' array is modified inplace\n\n    Returns\n    -------\n    heads : array of integers of same shape as parents\n        The indices in the 'parents' of the tree heads\n\n    "
    pass

def max_merge():
    'Merge two IntFloatDicts with the max strategy: when the same key is\n    present in the two dicts, the max of the two values is used.\n\n    Parameters\n    ==========\n    a, b : IntFloatDict object\n        The IntFloatDicts to merge\n    mask : ndarray array of dtype integer and of dimension 1\n        a mask for keys to ignore: if not mask[key] the corresponding key\n        is skipped in the output dictionary\n    n_a, n_b : float\n        n_a and n_b are weights for a and b for the merge strategy.\n        They are not used in the case of a max merge.\n\n    Returns\n    =======\n    out : IntFloatDict object\n        The IntFloatDict resulting from the merge\n    '
    pass

def single_linkage_label():
    '\n    Convert an linkage array or MST to a tree by labelling clusters at merges.\n    This is done by using a Union find structure to keep track of merges\n    efficiently.\n\n    Parameters\n    ----------\n    L: array of shape (n_samples - 1, 3)\n        The linkage array or MST where each row specifies two samples\n        to be merged and a distance or weight at which the merge occurs. This\n         array is assumed to be sorted by the distance/weight.\n\n    Returns\n    -------\n    A tree in the format used by scipy.cluster.hierarchy.\n    '
    pass

