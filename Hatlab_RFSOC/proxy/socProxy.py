import Pyro4
from qick import QickConfig
import socket

# get IPv4 addresses of the local PC and find the one that doesn't start with 136.142. (the local, non-Pitt one).
PittIP = "136.142"
addrInfo = socket.getaddrinfo(socket.gethostname(),None, family=socket.AF_INET)
ipv4Addres = [i[4][0] for i in addrInfo]
localIPv4 = None
for ip in ipv4Addres:
    if PittIP not in ip:
        if localIPv4 is None:
            localIPv4 = ip
        else:
            raise ValueError("can't determine which ip is used for nameserver, "
                             "check nameserver window and find ip manually")

Pyro4.config.SERIALIZER = "pickle"
Pyro4.config.PICKLE_PROTOCOL_VERSION=4

ns_host = localIPv4 # socket.gethostname() #"192.168.137.1"
ns_port = 8888
server_name = "myqick"

ns = Pyro4.locateNS(host=ns_host, port=ns_port)

for k,v in ns.list().items():
    print(k,v)

soc = Pyro4.Proxy(ns.lookup(server_name))
soccfg = QickConfig(soc.get_cfg())
# print(soccfg)