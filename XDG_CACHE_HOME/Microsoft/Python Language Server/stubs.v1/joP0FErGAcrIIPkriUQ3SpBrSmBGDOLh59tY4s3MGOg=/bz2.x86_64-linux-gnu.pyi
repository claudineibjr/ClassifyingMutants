import __builtin__ as _mod___builtin__

class BZ2Compressor(_mod___builtin__.object):
    'BZ2Compressor([compresslevel=9]) -> compressor object\n\nCreate a new compressor object. This object may be used to compress\ndata sequentially. If you want to compress data in one shot, use the\ncompress() function instead. The compresslevel parameter, if given,\nmust be a number between 1 and 9.\n'
    __class__ = BZ2Compressor
    def __delattr__(self):
        "x.__delattr__('name') <==> del x.name"
        return None
    
    def __getattribute__(self):
        "x.__getattribute__('name') <==> x.name"
        pass
    
    def __init__(self, compresslevel=9):
        'x.__init__(...) initializes x; see help(type(x)) for signature'
        pass
    
    def __setattr__(self):
        "x.__setattr__('name', value) <==> x.name = value"
        return None
    
    @classmethod
    def __subclasshook__(cls, subclass):
        'Abstract classes can override this to customize issubclass().\n\nThis is invoked early on by abc.ABCMeta.__subclasscheck__().\nIt should return True, False or NotImplemented.  If it returns\nNotImplemented, the normal algorithm is used.  Otherwise, it\noverrides the normal algorithm (and the outcome is cached).\n'
        return False
    
    def compress(self, data):
        "compress(data) -> string\n\nProvide more data to the compressor object. It will return chunks of\ncompressed data whenever possible. When you've finished providing data\nto compress, call the flush() method to finish the compression process,\nand return what is left in the internal buffers.\n"
        return ''
    
    def flush(self):
        'flush() -> string\n\nFinish the compression process and return what is left in internal buffers.\nYou must not use the compressor object after calling this method.\n'
        return ''
    

class BZ2Decompressor(_mod___builtin__.object):
    'BZ2Decompressor() -> decompressor object\n\nCreate a new decompressor object. This object may be used to decompress\ndata sequentially. If you want to decompress data in one shot, use the\ndecompress() function instead.\n'
    __class__ = BZ2Decompressor
    def __delattr__(self):
        "x.__delattr__('name') <==> del x.name"
        return None
    
    def __getattribute__(self):
        "x.__getattribute__('name') <==> x.name"
        pass
    
    def __init__(self):
        'x.__init__(...) initializes x; see help(type(x)) for signature'
        pass
    
    def __setattr__(self):
        "x.__setattr__('name', value) <==> x.name = value"
        return None
    
    @classmethod
    def __subclasshook__(cls, subclass):
        'Abstract classes can override this to customize issubclass().\n\nThis is invoked early on by abc.ABCMeta.__subclasscheck__().\nIt should return True, False or NotImplemented.  If it returns\nNotImplemented, the normal algorithm is used.  Otherwise, it\noverrides the normal algorithm (and the outcome is cached).\n'
        return False
    
    def decompress(self, data):
        "decompress(data) -> string\n\nProvide more data to the decompressor object. It will return chunks\nof decompressed data whenever possible. If you try to decompress data\nafter the end of stream is found, EOFError will be raised. If any data\nwas found after the end of stream, it'll be ignored and saved in\nunused_data attribute.\n"
        return ''
    
    @property
    def unused_data(self):
        pass
    

class BZ2File(_mod___builtin__.object):
    "BZ2File(name [, mode='r', buffering=0, compresslevel=9]) -> file object\n\nOpen a bz2 file. The mode can be 'r' or 'w', for reading (default) or\nwriting. When opened for writing, the file will be created if it doesn't\nexist, and truncated otherwise. If the buffering argument is given, 0 means\nunbuffered, and larger numbers specify the buffer size. If compresslevel\nis given, must be a number between 1 and 9.\n\nAdd a 'U' to mode to open the file for input with universal newline\nsupport. Any line ending in the input file will be seen as a '\\n' in\nPython. Also, a file so opened gains the attribute 'newlines'; the value\nfor this attribute is one of None (no newline read yet), '\\r', '\\n',\n'\\r\\n' or a tuple containing all the newline types seen. Universal\nnewlines are available only when reading.\n"
    __class__ = BZ2File
    def __delattr__(self):
        "x.__delattr__('name') <==> del x.name"
        return None
    
    def __enter__(self):
        '__enter__() -> self.'
        return self
    
    def __exit__(self, *excinfo):
        '__exit__(*excinfo) -> None.  Closes the file.'
        pass
    
    def __getattribute__(self):
        "x.__getattribute__('name') <==> x.name"
        pass
    
    def __init__(self, name, mode='r', buffering=0, compresslevel=9):
        'x.__init__(...) initializes x; see help(type(x)) for signature'
        pass
    
    def __iter__(self):
        'x.__iter__() <==> iter(x)'
        return BZ2File()
    
    def __setattr__(self):
        "x.__setattr__('name', value) <==> x.name = value"
        return None
    
    @classmethod
    def __subclasshook__(cls, subclass):
        'Abstract classes can override this to customize issubclass().\n\nThis is invoked early on by abc.ABCMeta.__subclasscheck__().\nIt should return True, False or NotImplemented.  If it returns\nNotImplemented, the normal algorithm is used.  Otherwise, it\noverrides the normal algorithm (and the outcome is cached).\n'
        return False
    
    def close(self):
        'close() -> None or (perhaps) an integer\n\nClose the file. Sets data attribute .closed to true. A closed file\ncannot be used for further I/O operations. close() may be called more\nthan once without error.\n'
        pass
    
    @property
    def closed(self):
        'True if the file is closed'
        pass
    
    @property
    def mode(self):
        "file mode ('r', 'w', or 'U')"
        pass
    
    @property
    def name(self):
        'file name'
        pass
    
    @property
    def newlines(self):
        'end-of-line convention used in this file'
        pass
    
    def next(self):
        'x.next() -> the next value, or raise StopIteration'
        pass
    
    def read(self, size=None):
        'read([size]) -> string\n\nRead at most size uncompressed bytes, returned as a string. If the size\nargument is negative or omitted, read until EOF is reached.\n'
        return ''
    
    def readline(self, size=None):
        'readline([size]) -> string\n\nReturn the next line from the file, as a string, retaining newline.\nA non-negative size argument will limit the maximum number of bytes to\nreturn (an incomplete line may be returned then). Return an empty\nstring at EOF.\n'
        return ''
    
    def readlines(self, size=None):
        'readlines([size]) -> list\n\nCall readline() repeatedly and return a list of lines read.\nThe optional size argument, if given, is an approximate bound on the\ntotal number of bytes in the lines returned.\n'
        return list()
    
    def seek(self, offset, whence=None):
        'seek(offset [, whence]) -> None\n\nMove to new file position. Argument offset is a byte count. Optional\nargument whence defaults to 0 (offset from start of file, offset\nshould be >= 0); other values are 1 (move relative to current position,\npositive or negative), and 2 (move relative to end of file, usually\nnegative, although many platforms allow seeking beyond the end of a file).\n\nNote that seeking of bz2 files is emulated, and depending on the parameters\nthe operation may be extremely slow.\n'
        pass
    
    @property
    def softspace(self):
        'flag indicating that a space needs to be printed; used by print'
        pass
    
    def tell(self):
        'tell() -> int\n\nReturn the current file position, an integer (may be a long integer).\n'
        return 1
    
    def write(self, data):
        "write(data) -> None\n\nWrite the 'data' string to file. Note that due to buffering, close() may\nbe needed before the file on disk reflects the data written.\n"
        pass
    
    def writelines(self, sequence_of_strings):
        'writelines(sequence_of_strings) -> None\n\nWrite the sequence of strings to the file. Note that newlines are not\nadded. The sequence can be any iterable object producing strings. This is\nequivalent to calling write() for each string.\n'
        pass
    
    def xreadlines(self):
        'xreadlines() -> self\n\nFor backward compatibility. BZ2File objects now include the performance\noptimizations previously implemented in the xreadlines module.\n'
        pass
    

__author__ = 'The bz2 python module was written by:\n\n    Gustavo Niemeyer <niemeyer@conectiva.com>\n'
__doc__ = 'The python bz2 module provides a comprehensive interface for\nthe bz2 compression library. It implements a complete file\ninterface, one shot (de)compression functions, and types for\nsequential (de)compression.\n'
__file__ = '/usr/lib/python2.7/lib-dynload/bz2.x86_64-linux-gnu.so'
__name__ = 'bz2'
__package__ = None
def compress(data, compresslevel=9):
    'compress(data [, compresslevel=9]) -> string\n\nCompress data in one shot. If you want to compress data sequentially,\nuse an instance of BZ2Compressor instead. The compresslevel parameter, if\ngiven, must be a number between 1 and 9.\n'
    return ''

def decompress(data):
    'decompress(data) -> decompressed data\n\nDecompress data in one shot. If you want to decompress data sequentially,\nuse an instance of BZ2Decompressor instead.\n'
    pass

