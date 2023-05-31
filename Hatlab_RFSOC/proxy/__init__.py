import Pyro4
from qick import QickConfig
import socket

LocalIP_Identifier = "192.168" #Assume the local network IP always starts with 192.168

def getLocalIPv4():
    """
    get IPv4 addresses of the local PC, and find the address of the ethernet card that is connected
    to lab local network.
    Assume the local network IP always starts with LocalIP_Identifier
    """
    addrInfo = socket.getaddrinfo(socket.gethostname(), None, family=socket.AF_INET)
    ipv4Addres = [i[4][0] for i in addrInfo]
    localIPv4 = None
    for ip in ipv4Addres:
        if LocalIP_Identifier in ip:
            if localIPv4 is None:
                localIPv4 = ip
            else:
                raise ValueError("can't determine which ip is used for nameserver, "
                                 "check nameserver window and find ip manually")
    return localIPv4


def getSocProxy(server_name: str, ns_host: str = None):
    if ns_host is None:
        ns_host = getLocalIPv4()  # socket.gethostname()

    Pyro4.config.SERIALIZER = "pickle"
    Pyro4.config.PICKLE_PROTOCOL_VERSION = 4
    ns_port = 8888
    ns = Pyro4.locateNS(host=ns_host, port=ns_port)

    soc = Pyro4.Proxy(ns.lookup(server_name))
    soccfg = QickConfig(soc.get_cfg())
    return soc, soccfg


if __name__ == "__main__":
    soc, soccfg = getSocProxy("myqick111_01")
    print(soccfg)
