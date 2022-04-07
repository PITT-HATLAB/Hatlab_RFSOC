:: Start a Pyro4 nameserver.
call C:\ProgramData\Anaconda3\Scripts\activate
SET PYRO_SERIALIZERS_ACCEPTED=pickle
SET PYRO_PICKLE_PROTOCOL_VERSION=4

:: pass all arguments to pyro4-ns
:: -n ip_address -p port
pyro4-ns -n 192.168.2.3 -p 8888
