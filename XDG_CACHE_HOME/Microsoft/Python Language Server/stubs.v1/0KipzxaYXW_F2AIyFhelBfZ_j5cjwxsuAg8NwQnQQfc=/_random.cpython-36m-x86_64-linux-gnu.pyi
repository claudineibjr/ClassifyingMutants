import builtins as _mod_builtins

__builtins__ = {}
__doc__ = '\nRandom utility function\n=======================\nThis module complements missing features of ``numpy.random``.\n\nThe module contains:\n    * Several algorithms to sample integers without replacement.\n    * Fast rand_r alternative based on xor shifts\n'
__file__ = '/home/claudinei/.local/lib/python3.6/site-packages/sklearn/utils/_random.cpython-36m-x86_64-linux-gnu.so'
__name__ = 'sklearn.utils._random'
__package__ = 'sklearn.utils'
__pyx_capi__ = _mod_builtins.dict()
__test__ = _mod_builtins.dict()
def _our_rand_r_py():
    'Python utils to test the our_rand_r function'
    pass

def _sample_without_replacement_check_input():
    ' Check that input are consistent for sample_without_replacement'
    pass

def _sample_without_replacement_with_pool():
    'Sample integers without replacement.\n\n    Select n_samples integers from the set [0, n_population) without\n    replacement.\n\n    Time complexity: O(n_population +  O(np.random.randint) * n_samples)\n\n    Space complexity of O(n_population + n_samples).\n\n\n    Parameters\n    ----------\n    n_population : int,\n        The size of the set to sample from.\n\n    n_samples : int,\n        The number of integer to sample.\n\n    random_state : int, RandomState instance or None, optional (default=None)\n        If int, random_state is the seed used by the random number generator;\n        If RandomState instance, random_state is the random number generator;\n        If None, the random number generator is the RandomState instance used\n        by `np.random`.\n\n    Returns\n    -------\n    out : array of size (n_samples, )\n        The sampled subsets of integer.\n    '
    pass

def _sample_without_replacement_with_reservoir_sampling():
    'Sample integers without replacement.\n\n    Select n_samples integers from the set [0, n_population) without\n    replacement.\n\n    Time complexity of\n        O((n_population - n_samples) * O(np.random.randint) + n_samples)\n    Space complexity of O(n_samples)\n\n\n    Parameters\n    ----------\n    n_population : int,\n        The size of the set to sample from.\n\n    n_samples : int,\n         The number of integer to sample.\n\n    random_state : int, RandomState instance or None, optional (default=None)\n        If int, random_state is the seed used by the random number generator;\n        If RandomState instance, random_state is the random number generator;\n        If None, the random number generator is the RandomState instance used\n        by `np.random`.\n\n    Returns\n    -------\n    out : array of size (n_samples, )\n        The sampled subsets of integer. The order of the items is not\n        necessarily random. Use a random permutation of the array if the order\n        of the items has to be randomized.\n    '
    pass

def _sample_without_replacement_with_tracking_selection():
    'Sample integers without replacement.\n\n    Select n_samples integers from the set [0, n_population) without\n    replacement.\n\n    Time complexity:\n        - Worst-case: unbounded\n        - Average-case:\n            O(O(np.random.randint) * \\sum_{i=1}^n_samples 1 /\n                                              (1 - i / n_population)))\n            <= O(O(np.random.randint) *\n                   n_population * ln((n_population - 2)\n                                     /(n_population - 1 - n_samples)))\n            <= O(O(np.random.randint) *\n                 n_population * 1 / (1 - n_samples / n_population))\n\n    Space complexity of O(n_samples) in a python set.\n\n\n    Parameters\n    ----------\n    n_population : int,\n        The size of the set to sample from.\n\n    n_samples : int,\n        The number of integer to sample.\n\n    random_state : int, RandomState instance or None, optional (default=None)\n        If int, random_state is the seed used by the random number generator;\n        If RandomState instance, random_state is the random number generator;\n        If None, the random number generator is the RandomState instance used\n        by `np.random`.\n\n    Returns\n    -------\n    out : array of size (n_samples, )\n        The sampled subsets of integer.\n    '
    pass

def check_random_state(seed):
    'Turn seed into a np.random.RandomState instance\n\n    Parameters\n    ----------\n    seed : None | int | instance of RandomState\n        If seed is None, return the RandomState singleton used by np.random.\n        If seed is an int, return a new RandomState instance seeded with seed.\n        If seed is already a RandomState instance, return it.\n        Otherwise raise ValueError.\n    '
    pass

def sample_without_replacement():
    'Sample integers without replacement.\n\n    Select n_samples integers from the set [0, n_population) without\n    replacement.\n\n\n    Parameters\n    ----------\n    n_population : int,\n        The size of the set to sample from.\n\n    n_samples : int,\n        The number of integer to sample.\n\n    random_state : int, RandomState instance or None, optional (default=None)\n        If int, random_state is the seed used by the random number generator;\n        If RandomState instance, random_state is the random number generator;\n        If None, the random number generator is the RandomState instance used\n        by `np.random`.\n\n    method : "auto", "tracking_selection", "reservoir_sampling" or "pool"\n        If method == "auto", the ratio of n_samples / n_population is used\n        to determine which algorithm to use:\n        If ratio is between 0 and 0.01, tracking selection is used.\n        If ratio is between 0.01 and 0.99, numpy.random.permutation is used.\n        If ratio is greater than 0.99, reservoir sampling is used.\n        The order of the selected integers is undefined. If a random order is\n        desired, the selected subset should be shuffled.\n\n        If method =="tracking_selection", a set based implementation is used\n        which is suitable for `n_samples` <<< `n_population`.\n\n        If method == "reservoir_sampling", a reservoir sampling algorithm is\n        used which is suitable for high memory constraint or when\n        O(`n_samples`) ~ O(`n_population`).\n        The order of the selected integers is undefined. If a random order is\n        desired, the selected subset should be shuffled.\n\n        If method == "pool", a pool based algorithm is particularly fast, even\n        faster than the tracking selection method. Hovewer, a vector containing\n        the entire population has to be initialized.\n        If n_samples ~ n_population, the reservoir sampling method is faster.\n\n    Returns\n    -------\n    out : array of size (n_samples, )\n        The sampled subsets of integer. The subset of selected integer might\n        not be randomized, see the method argument.\n    '
    pass

