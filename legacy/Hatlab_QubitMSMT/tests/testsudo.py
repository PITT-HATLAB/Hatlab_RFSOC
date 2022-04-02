import subprocess
sudoPassword = "xilinx"
command      = "su"

commandFinal = "echo " + sudoPassword + " | sudo -S " + command
output = subprocess.Popen(commandFinal, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
out = (output.communicate())
out = (out[0].strip())

subprocess.Popen("sudo su", shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

print("done")