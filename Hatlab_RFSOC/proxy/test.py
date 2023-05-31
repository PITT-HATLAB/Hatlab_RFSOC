from Hatlab_RFSOC.proxy import getSocProxy

# For test, change proxy_name name to the name you set on the Pyro server (running on the board)
soc, soccfg = getSocProxy(proxy_name="myqick111_01")
print(soccfg)