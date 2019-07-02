import builtins as _mod_builtins

__builtins__ = {}
__doc__ = 'Cython wrapper for MurmurHash3 non-cryptographic hash function\n\nMurmurHash is an extensively tested and very fast hash function that has\ngood distribution properties suitable for machine learning use cases\nsuch as feature hashing and random projections.\n\nThe original C++ code by Austin Appleby is released the public domain\nand can be found here:\n\n  https://code.google.com/p/smhasher/\n\n'
__file__ = '/home/claudinei/.local/lib/python3.6/site-packages/sklearn/utils/murmurhash.cpython-36m-x86_64-linux-gnu.so'
__name__ = 'sklearn.utils.murmurhash'
__package__ = 'sklearn.utils'
__pyx_capi__ = _mod_builtins.dict()
__test__ = _mod_builtins.dict()
def murmurhash3_32():
    'Compute the 32bit murmurhash3 of key at seed.\n\n    The underlying implementation is MurmurHash3_x86_32 generating low\n    latency 32bits hash suitable for implementing lookup tables, Bloom\n    filters, count min sketch or feature hashing.\n\n    Parameters\n    ----------\n    key : int32, bytes, unicode or ndarray with dtype int32\n        the physical object to hash\n\n    seed : int, optional default is 0\n        integer seed for the hashing algorithm.\n\n    positive : boolean, optional default is False\n        True: the results is casted to an unsigned int\n          from 0 to 2 ** 32 - 1\n        False: the results is casted to a signed int\n          from -(2 ** 31) to 2 ** 31 - 1\n\n    '
    pass

def murmurhash3_bytes_array_s32():
    'Compute 32bit murmurhash3 hashes of a key int array at seed.'
    pass

def murmurhash3_bytes_array_u32():
    'Compute 32bit murmurhash3 hashes of a key int array at seed.'
    pass

def murmurhash3_bytes_s32():
    'Compute the 32bit murmurhash3 of a bytes key at seed.'
    pass

def murmurhash3_bytes_u32():
    'Compute the 32bit murmurhash3 of a bytes key at seed.'
    pass

def murmurhash3_int_s32():
    'Compute the 32bit murmurhash3 of a int key at seed.'
    pass

def murmurhash3_int_u32():
    'Compute the 32bit murmurhash3 of a int key at seed.'
    pass

