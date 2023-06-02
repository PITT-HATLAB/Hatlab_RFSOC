:: try to find conda path and activate it 
SET condapath=C:\Users\%username%\anaconda3\Scripts
IF exist C:\ProgramData\Anaconda3 (call C:\ProgramData\Anaconda3\Scripts\activate ) ELSE (echo trying to find conda in user dir & IF exist %condapath% (call %condapath%\activate) ELSE (echo can't find conda))

:: Start a Pyro4 nameserver with pc hostname
SET PYRO_SERIALIZERS_ACCEPTED=pickle
SET PYRO_PICKLE_PROTOCOL_VERSION=4

:: find IPV4 address of the ethernet card on local network,

:: find the local network IP by excluding the pitt one that starts with 136.142
REM for /f "tokens=1-2 delims=:" %%a in ('ipconfig^|find "IPv4"^|find /v ": 136.142"') do set LOCALIPV4=%%b

:: find the local network IP by searching the one that starts with 192.168
for /f "tokens=1-2 delims=:" %%a in ('ipconfig^|find "IPv4"^|find ": 192.168"') do set LOCALIPV4=%%b

pyro4-ns -n %LOCALIPV4% -p 8888

:: using hostname doesn't guarantee that the IPv4 of local network will be used.
:: for /f "delims=" %%i in ('hostname') do pyro4-ns -n "%%i" -p 8888 

PAUSE