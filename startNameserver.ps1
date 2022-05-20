# Start a Pyro4 nameserver.

# call C:\ProgramData\Anaconda3\Scripts\activate
SET PYRO_SERIALIZERS_ACCEPTED=pickle
SET PYRO_PICKLE_PROTOCOL_VERSION=4
$hn = hostname

# pass all arguments to pyro4-ns
pyro4-ns -n "$hn" -p 8888

pause