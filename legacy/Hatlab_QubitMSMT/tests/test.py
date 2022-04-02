
# import subprocess
# commandFinal = "echo xilinx | sudo -S su"
# subprocess.Popen(commandFinal, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
# subprocess.Popen("sudo su", shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

# Import the QICK drivers and auxiliary libraries
from qick import *
from qick.helpers import gauss

print(0)

# Load bitstream with custom overlay
soc = QickSoc()
# Set the loopback DAC channel to be in 1st Nyquist zone mode
soc.set_nyquist(ch=7,nqz=1)

a=1

print(1)