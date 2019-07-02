import builtins as _mod_builtins
import numpy as _mod_numpy
import scipy.sparse.csr as _mod_scipy_sparse_csr

DTYPE = _mod_numpy.float64
ITYPE = _mod_numpy.int32
__builtins__ = {}
__doc__ = "\nRoutines for performing shortest-path graph searches\n\nThe main interface is in the function `graph_shortest_path`.  This\ncalls cython routines that compute the shortest path using either\nthe Floyd-Warshall algorithm, or Dykstra's algorithm with Fibonacci Heaps.\n"
__file__ = '/home/claudinei/.local/lib/python3.6/site-packages/sklearn/utils/graph_shortest_path.cpython-36m-x86_64-linux-gnu.so'
__name__ = 'sklearn.utils.graph_shortest_path'
__package__ = 'sklearn.utils'
__test__ = _mod_builtins.dict()
csr_matrix = _mod_scipy_sparse_csr.csr_matrix
def graph_shortest_path():
    "\n    Perform a shortest-path graph search on a positive directed or\n    undirected graph.\n\n    Parameters\n    ----------\n    dist_matrix : arraylike or sparse matrix, shape = (N,N)\n        Array of positive distances.\n        If vertex i is connected to vertex j, then dist_matrix[i,j] gives\n        the distance between the vertices.\n        If vertex i is not connected to vertex j, then dist_matrix[i,j] = 0\n    directed : boolean\n        if True, then find the shortest path on a directed graph: only\n        progress from a point to its neighbors, not the other way around.\n        if False, then find the shortest path on an undirected graph: the\n        algorithm can progress from a point to its neighbors and vice versa.\n    method : string ['auto'|'FW'|'D']\n        method to use.  Options are\n        'auto' : attempt to choose the best method for the current problem\n        'FW' : Floyd-Warshall algorithm.  O[N^3]\n        'D' : Dijkstra's algorithm with Fibonacci stacks.  O[(k+log(N))N^2]\n\n    Returns\n    -------\n    G : np.ndarray, float, shape = [N,N]\n        G[i,j] gives the shortest distance from point i to point j\n        along the graph.\n\n    Notes\n    -----\n    As currently implemented, Dijkstra's algorithm does not work for\n    graphs with direction-dependent distances when directed == False.\n    i.e., if dist_matrix[i,j] and dist_matrix[j,i] are not equal and\n    both are nonzero, method='D' will not necessarily yield the correct\n    result.\n\n    Also, these routines have not been tested for graphs with negative\n    distances.  Negative distances can lead to infinite cycles that must\n    be handled by specialized algorithms.\n    "
    pass

def isspmatrix(x):
    'Is x of a sparse matrix type?\n\n    Parameters\n    ----------\n    x\n        object to check for being a sparse matrix\n\n    Returns\n    -------\n    bool\n        True if x is a sparse matrix, False otherwise\n\n    Notes\n    -----\n    issparse and isspmatrix are aliases for the same function.\n\n    Examples\n    --------\n    >>> from scipy.sparse import csr_matrix, isspmatrix\n    >>> isspmatrix(csr_matrix([[5]]))\n    True\n\n    >>> from scipy.sparse import isspmatrix\n    >>> isspmatrix(5)\n    False\n    '
    pass

def isspmatrix_csr(x):
    'Is x of csr_matrix type?\n\n    Parameters\n    ----------\n    x\n        object to check for being a csr matrix\n\n    Returns\n    -------\n    bool\n        True if x is a csr matrix, False otherwise\n\n    Examples\n    --------\n    >>> from scipy.sparse import csr_matrix, isspmatrix_csr\n    >>> isspmatrix_csr(csr_matrix([[5]]))\n    True\n\n    >>> from scipy.sparse import csc_matrix, csr_matrix, isspmatrix_csc\n    >>> isspmatrix_csr(csc_matrix([[5]]))\n    False\n    '
    pass

