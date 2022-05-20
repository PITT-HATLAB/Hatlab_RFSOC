:: try to find conda path and activate it 
SET condapath=C:\Users\%username%\anaconda3\Scripts
IF exist C:\ProgramData\Anaconda3 (call C:\ProgramData\Anaconda3\Scripts\activate ) ELSE (echo trying to find conda in user dir & IF exist %condapath% (call %condapath%\activate) ELSE (echo can't find conda))

:: Start a Pyro4 nameserver with pc hostname
SET PYRO_SERIALIZERS_ACCEPTED=pickle
SET PYRO_PICKLE_PROTOCOL_VERSION=4
for /f "delims=" %%i in ('hostname') do pyro4-ns -n "%%i" -p 8888

PAUSE