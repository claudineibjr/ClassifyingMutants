import builtins as _mod_builtins

AF_APPLETALK = 5
AF_ASH = 18
AF_ATMPVC = 8
AF_ATMSVC = 20
AF_AX25 = 3
AF_BLUETOOTH = 31
AF_BRIDGE = 7
AF_DECnet = 12
AF_ECONET = 19
AF_FILE = 1
AF_INET = 2
AF_INET6 = 10
AF_IPX = 4
AF_IRDA = 23
AF_ISDN = 34
AF_KEY = 15
AF_LINK = 17
AF_NETBEUI = 13
AF_NETLINK = 16
AF_NETROM = 6
AF_PACKET = 17
AF_PPPOX = 24
AF_ROSE = 11
AF_ROUTE = 16
AF_SECURITY = 14
AF_SNA = 22
AF_UNIX = 1
AF_UNSPEC = 0
AF_WANPIPE = 25
AF_X25 = 9
__doc__ = None
__file__ = '/usr/lib/python3/dist-packages/netifaces.cpython-36m-x86_64-linux-gnu.so'
__name__ = 'netifaces'
__package__ = ''
address_families = _mod_builtins.dict()
def gateways():
    "Obtain a list of the gateways on this machine.\n\nReturns a dict whose keys are equal to the address family constants,\ne.g. netifaces.AF_INET, and whose values are a list of tuples of the\nformat (<address>, <interface>, <is_default>).\n\nThere is also a special entry with the key 'default', which you can use\nto quickly obtain the default gateway for a particular address family.\n\nThere may in general be multiple gateways; different address\nfamilies may have different gateway settings (e.g. AF_INET vs AF_INET6)\nand on some systems it's also possible to have interface-specific\ndefault gateways.\n"
    pass

def ifaddresses():
    'Obtain information about the specified network interface.\n\nReturns a dict whose keys are equal to the address family constants,\ne.g. netifaces.AF_INET, and whose values are a list of addresses in\nthat family that are attached to the network interface.'
    pass

def interfaces():
    'Obtain a list of the interfaces available on this machine.'
    pass

version = '0.10.4'
