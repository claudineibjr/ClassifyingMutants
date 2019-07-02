import builtins as _mod_builtins
import zmq.error as _mod_zmq_error

InterruptedSystemCall = _mod_zmq_error.InterruptedSystemCall
ROUTER = 6
ZMQError = _mod_zmq_error.ZMQError
__all__ = _mod_builtins.list()
__builtins__ = {}
__doc__ = 'MonitoredQueue classes and functions.\n\nAuthors\n-------\n* MinRK\n* Brian Granger\n'
__file__ = '/usr/local/lib/python3.6/dist-packages/zmq/devices/monitoredqueue.cpython-36m-x86_64-linux-gnu.so'
__name__ = 'zmq.devices.monitoredqueue'
__package__ = 'zmq.devices'
__test__ = _mod_builtins.dict()
def monitored_queue(in_socket, out_socket, mon_socket, in_prefix, out_prefix):
    "monitored_queue(in_socket, out_socket, mon_socket,\n                       in_prefix=b'in', out_prefix=b'out')\n    \n    Start a monitored queue device.\n    \n    A monitored queue is very similar to the zmq.proxy device (monitored queue came first).\n    \n    Differences from zmq.proxy:\n    \n    - monitored_queue supports both in and out being ROUTER sockets\n      (via swapping IDENTITY prefixes).\n    - monitor messages are prefixed, making in and out messages distinguishable.\n    \n    Parameters\n    ----------\n    in_socket : Socket\n        One of the sockets to the Queue. Its messages will be prefixed with\n        'in'.\n    out_socket : Socket\n        One of the sockets to the Queue. Its messages will be prefixed with\n        'out'. The only difference between in/out socket is this prefix.\n    mon_socket : Socket\n        This socket sends out every message received by each of the others\n        with an in/out prefix specifying which one it was.\n    in_prefix : str\n        Prefix added to broadcast messages from in_socket.\n    out_prefix : str\n        Prefix added to broadcast messages from out_socket.\n    "
    pass

