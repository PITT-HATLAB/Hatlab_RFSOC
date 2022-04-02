import subprocess
command = "scp test.txt hatlab@192.168.2.2:/C:/Users/hatla/Downloads"
subprocess.Popen(command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)