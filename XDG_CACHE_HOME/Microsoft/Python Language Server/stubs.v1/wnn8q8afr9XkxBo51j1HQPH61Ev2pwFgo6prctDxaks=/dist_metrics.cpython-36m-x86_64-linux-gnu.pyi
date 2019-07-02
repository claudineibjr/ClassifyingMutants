import builtins as _mod_builtins
import numpy as _mod_numpy

class BrayCurtisDistance(DistanceMetric):
    'Bray-Curtis Distance\n\n    Bray-Curtis distance is meant for discrete-valued vectors, though it is\n    a valid metric for real-valued vectors.\n\n    .. math::\n       D(x, y) = \\frac{\\sum_i |x_i - y_i|}{\\sum_i(|x_i| + |y_i|)}\n    '
    __class__ = BrayCurtisDistance
    def __init__(self, *args, **kwargs):
        'Bray-Curtis Distance\n\n    Bray-Curtis distance is meant for discrete-valued vectors, though it is\n    a valid metric for real-valued vectors.\n\n    .. math::\n       D(x, y) = \\frac{\\sum_i |x_i - y_i|}{\\sum_i(|x_i| + |y_i|)}\n    '
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
    
    @classmethod
    def get_metric(cls):
        'Get the given distance metric from the string identifier.\n\n        See the docstring of DistanceMetric for a list of available metrics.\n\n        Parameters\n        ----------\n        metric : string or class name\n            The distance metric to use\n        **kwargs\n            additional arguments will be passed to the requested metric\n        '
        pass
    

class CanberraDistance(DistanceMetric):
    'Canberra Distance\n\n    Canberra distance is meant for discrete-valued vectors, though it is\n    a valid metric for real-valued vectors.\n\n    .. math::\n       D(x, y) = \\sum_i \\frac{|x_i - y_i|}{|x_i| + |y_i|}\n    '
    __class__ = CanberraDistance
    def __init__(self, *args, **kwargs):
        'Canberra Distance\n\n    Canberra distance is meant for discrete-valued vectors, though it is\n    a valid metric for real-valued vectors.\n\n    .. math::\n       D(x, y) = \\sum_i \\frac{|x_i - y_i|}{|x_i| + |y_i|}\n    '
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
    
    @classmethod
    def get_metric(cls):
        'Get the given distance metric from the string identifier.\n\n        See the docstring of DistanceMetric for a list of available metrics.\n\n        Parameters\n        ----------\n        metric : string or class name\n            The distance metric to use\n        **kwargs\n            additional arguments will be passed to the requested metric\n        '
        pass
    

class ChebyshevDistance(DistanceMetric):
    'Chebyshev/Infinity Distance\n\n    .. math::\n       D(x, y) = max_i (|x_i - y_i|)\n    '
    __class__ = ChebyshevDistance
    def __init__(self, *args, **kwargs):
        'Chebyshev/Infinity Distance\n\n    .. math::\n       D(x, y) = max_i (|x_i - y_i|)\n    '
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
    
    @classmethod
    def get_metric(cls):
        'Get the given distance metric from the string identifier.\n\n        See the docstring of DistanceMetric for a list of available metrics.\n\n        Parameters\n        ----------\n        metric : string or class name\n            The distance metric to use\n        **kwargs\n            additional arguments will be passed to the requested metric\n        '
        pass
    

DTYPE = _mod_numpy.float64
class DiceDistance(DistanceMetric):
    'Dice Distance\n\n    Dice Distance is a dissimilarity measure for boolean-valued\n    vectors. All nonzero entries will be treated as True, zero entries will\n    be treated as False.\n\n    .. math::\n       D(x, y) = \\frac{N_{TF} + N_{FT}}{2 * N_{TT} + N_{TF} + N_{FT}}\n    '
    __class__ = DiceDistance
    def __init__(self, *args, **kwargs):
        'Dice Distance\n\n    Dice Distance is a dissimilarity measure for boolean-valued\n    vectors. All nonzero entries will be treated as True, zero entries will\n    be treated as False.\n\n    .. math::\n       D(x, y) = \\frac{N_{TF} + N_{FT}}{2 * N_{TT} + N_{TF} + N_{FT}}\n    '
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
    
    @classmethod
    def get_metric(cls):
        'Get the given distance metric from the string identifier.\n\n        See the docstring of DistanceMetric for a list of available metrics.\n\n        Parameters\n        ----------\n        metric : string or class name\n            The distance metric to use\n        **kwargs\n            additional arguments will be passed to the requested metric\n        '
        pass
    

class DistanceMetric(_mod_builtins.object):
    'DistanceMetric class\n\n    This class provides a uniform interface to fast distance metric\n    functions.  The various metrics can be accessed via the `get_metric`\n    class method and the metric string identifier (see below).\n    For example, to use the Euclidean distance:\n\n    >>> dist = DistanceMetric.get_metric(\'euclidean\')\n    >>> X = [[0, 1, 2],\n             [3, 4, 5]]\n    >>> dist.pairwise(X)\n    array([[ 0.        ,  5.19615242],\n           [ 5.19615242,  0.        ]])\n\n    Available Metrics\n\n    The following lists the string metric identifiers and the associated\n    distance metric classes:\n\n    **Metrics intended for real-valued vector spaces:**\n\n    ==============  ====================  ========  ===============================\n    identifier      class name            args      distance function\n    --------------  --------------------  --------  -------------------------------\n    "euclidean"     EuclideanDistance     -         ``sqrt(sum((x - y)^2))``\n    "manhattan"     ManhattanDistance     -         ``sum(|x - y|)``\n    "chebyshev"     ChebyshevDistance     -         ``max(|x - y|)``\n    "minkowski"     MinkowskiDistance     p         ``sum(|x - y|^p)^(1/p)``\n    "wminkowski"    WMinkowskiDistance    p, w      ``sum(|w * (x - y)|^p)^(1/p)``\n    "seuclidean"    SEuclideanDistance    V         ``sqrt(sum((x - y)^2 / V))``\n    "mahalanobis"   MahalanobisDistance   V or VI   ``sqrt((x - y)\' V^-1 (x - y))``\n    ==============  ====================  ========  ===============================\n\n    **Metrics intended for two-dimensional vector spaces:**  Note that the haversine\n    distance metric requires data in the form of [latitude, longitude] and both\n    inputs and outputs are in units of radians.\n\n    ============  ==================  ===============================================================\n    identifier    class name          distance function\n    ------------  ------------------  ---------------------------------------------------------------\n    "haversine"   HaversineDistance   ``2 arcsin(sqrt(sin^2(0.5*dx) + cos(x1)cos(x2)sin^2(0.5*dy)))``\n    ============  ==================  ===============================================================\n\n\n    **Metrics intended for integer-valued vector spaces:**  Though intended\n    for integer-valued vectors, these are also valid metrics in the case of\n    real-valued vectors.\n\n    =============  ====================  ========================================\n    identifier     class name            distance function\n    -------------  --------------------  ----------------------------------------\n    "hamming"      HammingDistance       ``N_unequal(x, y) / N_tot``\n    "canberra"     CanberraDistance      ``sum(|x - y| / (|x| + |y|))``\n    "braycurtis"   BrayCurtisDistance    ``sum(|x - y|) / (sum(|x|) + sum(|y|))``\n    =============  ====================  ========================================\n\n    **Metrics intended for boolean-valued vector spaces:**  Any nonzero entry\n    is evaluated to "True".  In the listings below, the following\n    abbreviations are used:\n\n     - N  : number of dimensions\n     - NTT : number of dims in which both values are True\n     - NTF : number of dims in which the first value is True, second is False\n     - NFT : number of dims in which the first value is False, second is True\n     - NFF : number of dims in which both values are False\n     - NNEQ : number of non-equal dimensions, NNEQ = NTF + NFT\n     - NNZ : number of nonzero dimensions, NNZ = NTF + NFT + NTT\n\n    =================  =======================  ===============================\n    identifier         class name               distance function\n    -----------------  -----------------------  -------------------------------\n    "jaccard"          JaccardDistance          NNEQ / NNZ\n    "matching"         MatchingDistance         NNEQ / N\n    "dice"             DiceDistance             NNEQ / (NTT + NNZ)\n    "kulsinski"        KulsinskiDistance        (NNEQ + N - NTT) / (NNEQ + N)\n    "rogerstanimoto"   RogersTanimotoDistance   2 * NNEQ / (N + NNEQ)\n    "russellrao"       RussellRaoDistance       NNZ / N\n    "sokalmichener"    SokalMichenerDistance    2 * NNEQ / (N + NNEQ)\n    "sokalsneath"      SokalSneathDistance      NNEQ / (NNEQ + 0.5 * NTT)\n    =================  =======================  ===============================\n\n    **User-defined distance:**\n\n    ===========    ===============    =======\n    identifier     class name         args\n    -----------    ---------------    -------\n    "pyfunc"       PyFuncDistance     func\n    ===========    ===============    =======\n\n    Here ``func`` is a function which takes two one-dimensional numpy\n    arrays, and returns a distance.  Note that in order to be used within\n    the BallTree, the distance must be a true metric:\n    i.e. it must satisfy the following properties\n\n    1) Non-negativity: d(x, y) >= 0\n    2) Identity: d(x, y) = 0 if and only if x == y\n    3) Symmetry: d(x, y) = d(y, x)\n    4) Triangle Inequality: d(x, y) + d(y, z) >= d(x, z)\n\n    Because of the Python object overhead involved in calling the python\n    function, this will be fairly slow, but it will have the same\n    scaling as other distances.\n    '
    __class__ = DistanceMetric
    def __getstate__(self):
        '\n        get state for pickling\n        '
        pass
    
    def __init__(self, *args, **kwargs):
        'DistanceMetric class\n\n    This class provides a uniform interface to fast distance metric\n    functions.  The various metrics can be accessed via the `get_metric`\n    class method and the metric string identifier (see below).\n    For example, to use the Euclidean distance:\n\n    >>> dist = DistanceMetric.get_metric(\'euclidean\')\n    >>> X = [[0, 1, 2],\n             [3, 4, 5]]\n    >>> dist.pairwise(X)\n    array([[ 0.        ,  5.19615242],\n           [ 5.19615242,  0.        ]])\n\n    Available Metrics\n\n    The following lists the string metric identifiers and the associated\n    distance metric classes:\n\n    **Metrics intended for real-valued vector spaces:**\n\n    ==============  ====================  ========  ===============================\n    identifier      class name            args      distance function\n    --------------  --------------------  --------  -------------------------------\n    "euclidean"     EuclideanDistance     -         ``sqrt(sum((x - y)^2))``\n    "manhattan"     ManhattanDistance     -         ``sum(|x - y|)``\n    "chebyshev"     ChebyshevDistance     -         ``max(|x - y|)``\n    "minkowski"     MinkowskiDistance     p         ``sum(|x - y|^p)^(1/p)``\n    "wminkowski"    WMinkowskiDistance    p, w      ``sum(|w * (x - y)|^p)^(1/p)``\n    "seuclidean"    SEuclideanDistance    V         ``sqrt(sum((x - y)^2 / V))``\n    "mahalanobis"   MahalanobisDistance   V or VI   ``sqrt((x - y)\' V^-1 (x - y))``\n    ==============  ====================  ========  ===============================\n\n    **Metrics intended for two-dimensional vector spaces:**  Note that the haversine\n    distance metric requires data in the form of [latitude, longitude] and both\n    inputs and outputs are in units of radians.\n\n    ============  ==================  ===============================================================\n    identifier    class name          distance function\n    ------------  ------------------  ---------------------------------------------------------------\n    "haversine"   HaversineDistance   ``2 arcsin(sqrt(sin^2(0.5*dx) + cos(x1)cos(x2)sin^2(0.5*dy)))``\n    ============  ==================  ===============================================================\n\n\n    **Metrics intended for integer-valued vector spaces:**  Though intended\n    for integer-valued vectors, these are also valid metrics in the case of\n    real-valued vectors.\n\n    =============  ====================  ========================================\n    identifier     class name            distance function\n    -------------  --------------------  ----------------------------------------\n    "hamming"      HammingDistance       ``N_unequal(x, y) / N_tot``\n    "canberra"     CanberraDistance      ``sum(|x - y| / (|x| + |y|))``\n    "braycurtis"   BrayCurtisDistance    ``sum(|x - y|) / (sum(|x|) + sum(|y|))``\n    =============  ====================  ========================================\n\n    **Metrics intended for boolean-valued vector spaces:**  Any nonzero entry\n    is evaluated to "True".  In the listings below, the following\n    abbreviations are used:\n\n     - N  : number of dimensions\n     - NTT : number of dims in which both values are True\n     - NTF : number of dims in which the first value is True, second is False\n     - NFT : number of dims in which the first value is False, second is True\n     - NFF : number of dims in which both values are False\n     - NNEQ : number of non-equal dimensions, NNEQ = NTF + NFT\n     - NNZ : number of nonzero dimensions, NNZ = NTF + NFT + NTT\n\n    =================  =======================  ===============================\n    identifier         class name               distance function\n    -----------------  -----------------------  -------------------------------\n    "jaccard"          JaccardDistance          NNEQ / NNZ\n    "matching"         MatchingDistance         NNEQ / N\n    "dice"             DiceDistance             NNEQ / (NTT + NNZ)\n    "kulsinski"        KulsinskiDistance        (NNEQ + N - NTT) / (NNEQ + N)\n    "rogerstanimoto"   RogersTanimotoDistance   2 * NNEQ / (N + NNEQ)\n    "russellrao"       RussellRaoDistance       NNZ / N\n    "sokalmichener"    SokalMichenerDistance    2 * NNEQ / (N + NNEQ)\n    "sokalsneath"      SokalSneathDistance      NNEQ / (NNEQ + 0.5 * NTT)\n    =================  =======================  ===============================\n\n    **User-defined distance:**\n\n    ===========    ===============    =======\n    identifier     class name         args\n    -----------    ---------------    -------\n    "pyfunc"       PyFuncDistance     func\n    ===========    ===============    =======\n\n    Here ``func`` is a function which takes two one-dimensional numpy\n    arrays, and returns a distance.  Note that in order to be used within\n    the BallTree, the distance must be a true metric:\n    i.e. it must satisfy the following properties\n\n    1) Non-negativity: d(x, y) >= 0\n    2) Identity: d(x, y) = 0 if and only if x == y\n    3) Symmetry: d(x, y) = d(y, x)\n    4) Triangle Inequality: d(x, y) + d(y, z) >= d(x, z)\n\n    Because of the Python object overhead involved in calling the python\n    function, this will be fairly slow, but it will have the same\n    scaling as other distances.\n    '
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
    
    def dist_to_rdist(self):
        'Convert the true distance to the reduced distance.\n\n        The reduced distance, defined for some metrics, is a computationally\n        more efficient measure which preserves the rank of the true distance.\n        For example, in the Euclidean distance metric, the reduced distance\n        is the squared-euclidean distance.\n        '
        pass
    
    @classmethod
    def get_metric(cls):
        'Get the given distance metric from the string identifier.\n\n        See the docstring of DistanceMetric for a list of available metrics.\n\n        Parameters\n        ----------\n        metric : string or class name\n            The distance metric to use\n        **kwargs\n            additional arguments will be passed to the requested metric\n        '
        pass
    
    def pairwise(self):
        'Compute the pairwise distances between X and Y\n\n        This is a convenience routine for the sake of testing.  For many\n        metrics, the utilities in scipy.spatial.distance.cdist and\n        scipy.spatial.distance.pdist will be faster.\n\n        Parameters\n        ----------\n        X : array_like\n            Array of shape (Nx, D), representing Nx points in D dimensions.\n        Y : array_like (optional)\n            Array of shape (Ny, D), representing Ny points in D dimensions.\n            If not specified, then Y=X.\n        Returns\n        -------\n        dist : ndarray\n            The shape (Nx, Ny) array of pairwise distances between points in\n            X and Y.\n        '
        pass
    
    def rdist_to_dist(self):
        'Convert the Reduced distance to the true distance.\n\n        The reduced distance, defined for some metrics, is a computationally\n        more efficient measure which preserves the rank of the true distance.\n        For example, in the Euclidean distance metric, the reduced distance\n        is the squared-euclidean distance.\n        '
        pass
    

class EuclideanDistance(DistanceMetric):
    'Euclidean Distance metric\n\n    .. math::\n       D(x, y) = \\sqrt{ \\sum_i (x_i - y_i) ^ 2 }\n    '
    __class__ = EuclideanDistance
    def __init__(self, *args, **kwargs):
        'Euclidean Distance metric\n\n    .. math::\n       D(x, y) = \\sqrt{ \\sum_i (x_i - y_i) ^ 2 }\n    '
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
    
    def dist_to_rdist(self):
        pass
    
    @classmethod
    def get_metric(cls):
        'Get the given distance metric from the string identifier.\n\n        See the docstring of DistanceMetric for a list of available metrics.\n\n        Parameters\n        ----------\n        metric : string or class name\n            The distance metric to use\n        **kwargs\n            additional arguments will be passed to the requested metric\n        '
        pass
    
    def rdist_to_dist(self):
        pass
    

class HammingDistance(DistanceMetric):
    'Hamming Distance\n\n    Hamming distance is meant for discrete-valued vectors, though it is\n    a valid metric for real-valued vectors.\n\n    .. math::\n       D(x, y) = \\frac{1}{N} \\sum_i \\delta_{x_i, y_i}\n    '
    __class__ = HammingDistance
    def __init__(self, *args, **kwargs):
        'Hamming Distance\n\n    Hamming distance is meant for discrete-valued vectors, though it is\n    a valid metric for real-valued vectors.\n\n    .. math::\n       D(x, y) = \\frac{1}{N} \\sum_i \\delta_{x_i, y_i}\n    '
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
    
    @classmethod
    def get_metric(cls):
        'Get the given distance metric from the string identifier.\n\n        See the docstring of DistanceMetric for a list of available metrics.\n\n        Parameters\n        ----------\n        metric : string or class name\n            The distance metric to use\n        **kwargs\n            additional arguments will be passed to the requested metric\n        '
        pass
    

class HaversineDistance(DistanceMetric):
    'Haversine (Spherical) Distance\n\n    The Haversine distance is the angular distance between two points on\n    the surface of a sphere.  The first distance of each point is assumed\n    to be the latitude, the second is the longitude, given in radians.\n    The dimension of the points must be 2:\n\n    .. math::\n       D(x, y) = 2\\arcsin[\\sqrt{\\sin^2((x1 - y1) / 2)\n                                + \\cos(x1)\\cos(y1)\\sin^2((x2 - y2) / 2)}]\n    '
    __class__ = HaversineDistance
    def __init__(self, Spherical):
        'Haversine (Spherical) Distance\n\n    The Haversine distance is the angular distance between two points on\n    the surface of a sphere.  The first distance of each point is assumed\n    to be the latitude, the second is the longitude, given in radians.\n    The dimension of the points must be 2:\n\n    .. math::\n       D(x, y) = 2\\arcsin[\\sqrt{\\sin^2((x1 - y1) / 2)\n                                + \\cos(x1)\\cos(y1)\\sin^2((x2 - y2) / 2)}]\n    '
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
    
    def dist_to_rdist(self):
        pass
    
    @classmethod
    def get_metric(cls):
        'Get the given distance metric from the string identifier.\n\n        See the docstring of DistanceMetric for a list of available metrics.\n\n        Parameters\n        ----------\n        metric : string or class name\n            The distance metric to use\n        **kwargs\n            additional arguments will be passed to the requested metric\n        '
        pass
    
    def rdist_to_dist(self):
        pass
    

ITYPE = _mod_numpy.int64
class JaccardDistance(DistanceMetric):
    'Jaccard Distance\n\n    Jaccard Distance is a dissimilarity measure for boolean-valued\n    vectors. All nonzero entries will be treated as True, zero entries will\n    be treated as False.\n\n    .. math::\n       D(x, y) = \\frac{N_{TF} + N_{FT}}{N_{TT} + N_{TF} + N_{FT}}\n    '
    __class__ = JaccardDistance
    def __init__(self, *args, **kwargs):
        'Jaccard Distance\n\n    Jaccard Distance is a dissimilarity measure for boolean-valued\n    vectors. All nonzero entries will be treated as True, zero entries will\n    be treated as False.\n\n    .. math::\n       D(x, y) = \\frac{N_{TF} + N_{FT}}{N_{TT} + N_{TF} + N_{FT}}\n    '
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
    
    @classmethod
    def get_metric(cls):
        'Get the given distance metric from the string identifier.\n\n        See the docstring of DistanceMetric for a list of available metrics.\n\n        Parameters\n        ----------\n        metric : string or class name\n            The distance metric to use\n        **kwargs\n            additional arguments will be passed to the requested metric\n        '
        pass
    

class KulsinskiDistance(DistanceMetric):
    'Kulsinski Distance\n\n    Kulsinski Distance is a dissimilarity measure for boolean-valued\n    vectors. All nonzero entries will be treated as True, zero entries will\n    be treated as False.\n\n    .. math::\n       D(x, y) = 1 - \\frac{N_{TT}}{N + N_{TF} + N_{FT}}\n    '
    __class__ = KulsinskiDistance
    def __init__(self, *args, **kwargs):
        'Kulsinski Distance\n\n    Kulsinski Distance is a dissimilarity measure for boolean-valued\n    vectors. All nonzero entries will be treated as True, zero entries will\n    be treated as False.\n\n    .. math::\n       D(x, y) = 1 - \\frac{N_{TT}}{N + N_{TF} + N_{FT}}\n    '
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
    
    @classmethod
    def get_metric(cls):
        'Get the given distance metric from the string identifier.\n\n        See the docstring of DistanceMetric for a list of available metrics.\n\n        Parameters\n        ----------\n        metric : string or class name\n            The distance metric to use\n        **kwargs\n            additional arguments will be passed to the requested metric\n        '
        pass
    

METRIC_MAPPING = _mod_builtins.dict()
class MahalanobisDistance(DistanceMetric):
    'Mahalanobis Distance\n\n    .. math::\n       D(x, y) = \\sqrt{ (x - y)^T V^{-1} (x - y) }\n\n    Parameters\n    ----------\n    V : array_like\n        Symmetric positive-definite covariance matrix.\n        The inverse of this matrix will be explicitly computed.\n    VI : array_like\n        optionally specify the inverse directly.  If VI is passed,\n        then V is not referenced.\n    '
    __class__ = MahalanobisDistance
    def __init__(self, *args, **kwargs):
        'Mahalanobis Distance\n\n    .. math::\n       D(x, y) = \\sqrt{ (x - y)^T V^{-1} (x - y) }\n\n    Parameters\n    ----------\n    V : array_like\n        Symmetric positive-definite covariance matrix.\n        The inverse of this matrix will be explicitly computed.\n    VI : array_like\n        optionally specify the inverse directly.  If VI is passed,\n        then V is not referenced.\n    '
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
    
    def dist_to_rdist(self):
        pass
    
    @classmethod
    def get_metric(cls):
        'Get the given distance metric from the string identifier.\n\n        See the docstring of DistanceMetric for a list of available metrics.\n\n        Parameters\n        ----------\n        metric : string or class name\n            The distance metric to use\n        **kwargs\n            additional arguments will be passed to the requested metric\n        '
        pass
    
    def rdist_to_dist(self):
        pass
    

class ManhattanDistance(DistanceMetric):
    'Manhattan/City-block Distance metric\n\n    .. math::\n       D(x, y) = \\sum_i |x_i - y_i|\n    '
    __class__ = ManhattanDistance
    def __init__(self, *args, **kwargs):
        'Manhattan/City-block Distance metric\n\n    .. math::\n       D(x, y) = \\sum_i |x_i - y_i|\n    '
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
    
    @classmethod
    def get_metric(cls):
        'Get the given distance metric from the string identifier.\n\n        See the docstring of DistanceMetric for a list of available metrics.\n\n        Parameters\n        ----------\n        metric : string or class name\n            The distance metric to use\n        **kwargs\n            additional arguments will be passed to the requested metric\n        '
        pass
    

class MatchingDistance(DistanceMetric):
    'Matching Distance\n\n    Matching Distance is a dissimilarity measure for boolean-valued\n    vectors. All nonzero entries will be treated as True, zero entries will\n    be treated as False.\n\n    .. math::\n       D(x, y) = \\frac{N_{TF} + N_{FT}}{N}\n    '
    __class__ = MatchingDistance
    def __init__(self, *args, **kwargs):
        'Matching Distance\n\n    Matching Distance is a dissimilarity measure for boolean-valued\n    vectors. All nonzero entries will be treated as True, zero entries will\n    be treated as False.\n\n    .. math::\n       D(x, y) = \\frac{N_{TF} + N_{FT}}{N}\n    '
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
    
    @classmethod
    def get_metric(cls):
        'Get the given distance metric from the string identifier.\n\n        See the docstring of DistanceMetric for a list of available metrics.\n\n        Parameters\n        ----------\n        metric : string or class name\n            The distance metric to use\n        **kwargs\n            additional arguments will be passed to the requested metric\n        '
        pass
    

class MinkowskiDistance(DistanceMetric):
    'Minkowski Distance\n\n    .. math::\n       D(x, y) = [\\sum_i (x_i - y_i)^p] ^ (1/p)\n\n    Minkowski Distance requires p >= 1 and finite. For p = infinity,\n    use ChebyshevDistance.\n    Note that for p=1, ManhattanDistance is more efficient, and for\n    p=2, EuclideanDistance is more efficient.\n    '
    __class__ = MinkowskiDistance
    def __init__(self, *args, **kwargs):
        'Minkowski Distance\n\n    .. math::\n       D(x, y) = [\\sum_i (x_i - y_i)^p] ^ (1/p)\n\n    Minkowski Distance requires p >= 1 and finite. For p = infinity,\n    use ChebyshevDistance.\n    Note that for p=1, ManhattanDistance is more efficient, and for\n    p=2, EuclideanDistance is more efficient.\n    '
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
    
    def dist_to_rdist(self):
        pass
    
    @classmethod
    def get_metric(cls):
        'Get the given distance metric from the string identifier.\n\n        See the docstring of DistanceMetric for a list of available metrics.\n\n        Parameters\n        ----------\n        metric : string or class name\n            The distance metric to use\n        **kwargs\n            additional arguments will be passed to the requested metric\n        '
        pass
    
    def rdist_to_dist(self):
        pass
    

class PyFuncDistance(DistanceMetric):
    'PyFunc Distance\n\n    A user-defined distance\n\n    Parameters\n    ----------\n    func : function\n        func should take two numpy arrays as input, and return a distance.\n    '
    __class__ = PyFuncDistance
    def __init__(self, *args, **kwargs):
        'PyFunc Distance\n\n    A user-defined distance\n\n    Parameters\n    ----------\n    func : function\n        func should take two numpy arrays as input, and return a distance.\n    '
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
    
    @classmethod
    def get_metric(cls):
        'Get the given distance metric from the string identifier.\n\n        See the docstring of DistanceMetric for a list of available metrics.\n\n        Parameters\n        ----------\n        metric : string or class name\n            The distance metric to use\n        **kwargs\n            additional arguments will be passed to the requested metric\n        '
        pass
    

class RogersTanimotoDistance(DistanceMetric):
    'Rogers-Tanimoto Distance\n\n    Rogers-Tanimoto Distance is a dissimilarity measure for boolean-valued\n    vectors. All nonzero entries will be treated as True, zero entries will\n    be treated as False.\n\n    .. math::\n       D(x, y) = \\frac{2 (N_{TF} + N_{FT})}{N + N_{TF} + N_{FT}}\n    '
    __class__ = RogersTanimotoDistance
    def __init__(self, *args, **kwargs):
        'Rogers-Tanimoto Distance\n\n    Rogers-Tanimoto Distance is a dissimilarity measure for boolean-valued\n    vectors. All nonzero entries will be treated as True, zero entries will\n    be treated as False.\n\n    .. math::\n       D(x, y) = \\frac{2 (N_{TF} + N_{FT})}{N + N_{TF} + N_{FT}}\n    '
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
    
    @classmethod
    def get_metric(cls):
        'Get the given distance metric from the string identifier.\n\n        See the docstring of DistanceMetric for a list of available metrics.\n\n        Parameters\n        ----------\n        metric : string or class name\n            The distance metric to use\n        **kwargs\n            additional arguments will be passed to the requested metric\n        '
        pass
    

class RussellRaoDistance(DistanceMetric):
    'Russell-Rao Distance\n\n    Russell-Rao Distance is a dissimilarity measure for boolean-valued\n    vectors. All nonzero entries will be treated as True, zero entries will\n    be treated as False.\n\n    .. math::\n       D(x, y) = \\frac{N - N_{TT}}{N}\n    '
    __class__ = RussellRaoDistance
    def __init__(self, *args, **kwargs):
        'Russell-Rao Distance\n\n    Russell-Rao Distance is a dissimilarity measure for boolean-valued\n    vectors. All nonzero entries will be treated as True, zero entries will\n    be treated as False.\n\n    .. math::\n       D(x, y) = \\frac{N - N_{TT}}{N}\n    '
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
    
    @classmethod
    def get_metric(cls):
        'Get the given distance metric from the string identifier.\n\n        See the docstring of DistanceMetric for a list of available metrics.\n\n        Parameters\n        ----------\n        metric : string or class name\n            The distance metric to use\n        **kwargs\n            additional arguments will be passed to the requested metric\n        '
        pass
    

class SEuclideanDistance(DistanceMetric):
    'Standardized Euclidean Distance metric\n\n    .. math::\n       D(x, y) = \\sqrt{ \\sum_i \\frac{ (x_i - y_i) ^ 2}{V_i} }\n    '
    __class__ = SEuclideanDistance
    def __init__(self, *args, **kwargs):
        'Standardized Euclidean Distance metric\n\n    .. math::\n       D(x, y) = \\sqrt{ \\sum_i \\frac{ (x_i - y_i) ^ 2}{V_i} }\n    '
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
    
    def dist_to_rdist(self):
        pass
    
    @classmethod
    def get_metric(cls):
        'Get the given distance metric from the string identifier.\n\n        See the docstring of DistanceMetric for a list of available metrics.\n\n        Parameters\n        ----------\n        metric : string or class name\n            The distance metric to use\n        **kwargs\n            additional arguments will be passed to the requested metric\n        '
        pass
    
    def rdist_to_dist(self):
        pass
    

class SokalMichenerDistance(DistanceMetric):
    'Sokal-Michener Distance\n\n    Sokal-Michener Distance is a dissimilarity measure for boolean-valued\n    vectors. All nonzero entries will be treated as True, zero entries will\n    be treated as False.\n\n    .. math::\n       D(x, y) = \\frac{2 (N_{TF} + N_{FT})}{N + N_{TF} + N_{FT}}\n    '
    __class__ = SokalMichenerDistance
    def __init__(self, *args, **kwargs):
        'Sokal-Michener Distance\n\n    Sokal-Michener Distance is a dissimilarity measure for boolean-valued\n    vectors. All nonzero entries will be treated as True, zero entries will\n    be treated as False.\n\n    .. math::\n       D(x, y) = \\frac{2 (N_{TF} + N_{FT})}{N + N_{TF} + N_{FT}}\n    '
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
    
    @classmethod
    def get_metric(cls):
        'Get the given distance metric from the string identifier.\n\n        See the docstring of DistanceMetric for a list of available metrics.\n\n        Parameters\n        ----------\n        metric : string or class name\n            The distance metric to use\n        **kwargs\n            additional arguments will be passed to the requested metric\n        '
        pass
    

class SokalSneathDistance(DistanceMetric):
    'Sokal-Sneath Distance\n\n    Sokal-Sneath Distance is a dissimilarity measure for boolean-valued\n    vectors. All nonzero entries will be treated as True, zero entries will\n    be treated as False.\n\n    .. math::\n       D(x, y) = \\frac{N_{TF} + N_{FT}}{N_{TT} / 2 + N_{TF} + N_{FT}}\n    '
    __class__ = SokalSneathDistance
    def __init__(self, *args, **kwargs):
        'Sokal-Sneath Distance\n\n    Sokal-Sneath Distance is a dissimilarity measure for boolean-valued\n    vectors. All nonzero entries will be treated as True, zero entries will\n    be treated as False.\n\n    .. math::\n       D(x, y) = \\frac{N_{TF} + N_{FT}}{N_{TT} / 2 + N_{TF} + N_{FT}}\n    '
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
    
    @classmethod
    def get_metric(cls):
        'Get the given distance metric from the string identifier.\n\n        See the docstring of DistanceMetric for a list of available metrics.\n\n        Parameters\n        ----------\n        metric : string or class name\n            The distance metric to use\n        **kwargs\n            additional arguments will be passed to the requested metric\n        '
        pass
    

class WMinkowskiDistance(DistanceMetric):
    'Weighted Minkowski Distance\n\n    .. math::\n       D(x, y) = [\\sum_i |w_i * (x_i - y_i)|^p] ^ (1/p)\n\n    Weighted Minkowski Distance requires p >= 1 and finite.\n\n    Parameters\n    ----------\n    p : int\n        The order of the norm of the difference :math:`{||u-v||}_p`.\n    w : (N,) array_like\n        The weight vector.\n\n    '
    __class__ = WMinkowskiDistance
    def __init__(self, *args, **kwargs):
        'Weighted Minkowski Distance\n\n    .. math::\n       D(x, y) = [\\sum_i |w_i * (x_i - y_i)|^p] ^ (1/p)\n\n    Weighted Minkowski Distance requires p >= 1 and finite.\n\n    Parameters\n    ----------\n    p : int\n        The order of the norm of the difference :math:`{||u-v||}_p`.\n    w : (N,) array_like\n        The weight vector.\n\n    '
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
    
    def dist_to_rdist(self):
        pass
    
    @classmethod
    def get_metric(cls):
        'Get the given distance metric from the string identifier.\n\n        See the docstring of DistanceMetric for a list of available metrics.\n\n        Parameters\n        ----------\n        metric : string or class name\n            The distance metric to use\n        **kwargs\n            additional arguments will be passed to the requested metric\n        '
        pass
    
    def rdist_to_dist(self):
        pass
    

__builtins__ = {}
__doc__ = None
__file__ = '/home/claudinei/.local/lib/python3.6/site-packages/sklearn/neighbors/dist_metrics.cpython-36m-x86_64-linux-gnu.so'
__name__ = 'sklearn.neighbors.dist_metrics'
__package__ = 'sklearn.neighbors'
def __pyx_unpickle_Enum():
    pass

__test__ = _mod_builtins.dict()
def get_valid_metric_ids():
    "Given an iterable of metric class names or class identifiers,\n    return a list of metric IDs which map to those classes.\n\n    Example:\n    >>> L = get_valid_metric_ids([EuclideanDistance, 'ManhattanDistance'])\n    >>> sorted(L)\n    ['cityblock', 'euclidean', 'l1', 'l2', 'manhattan']\n    "
    pass

def newObj():
    pass

