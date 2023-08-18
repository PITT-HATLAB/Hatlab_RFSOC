import time

from Hatlab_RFSOC.proxy import getSocProxy
from qick import *
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
soc, soccfg = getSocProxy("myqick216-01")
from T004C_phaseReset_test import LoopbackProgram





config = {"res_ch": 0,  # --Fixed
          "ro_chs": [0],  # --Fixed
          "reps": 20000,  # --Fixed
          "relax_delay": 1.0,  # --us
          "res_phase": 0,  # --degrees
          "pulse_style": "const",  # --Fixed

          "length": 1000,  # [Clock ticks]
          # Try varying length from 10-100 clock ticks

          "readout_length": 1000,  # [Clock ticks]
          # Try varying readout_length from 50-1000 clock ticks

          "pulse_gain": 30000,  # [DAC units]
          # Try varying pulse_gain from 500 to 30000 DAC units

          # "pulse_freq": 2457.60 * 3 +10 ,  # [MHz]
          # "readout_freq": 10,  # [MHz]

          "pulse_freq": 4933.3,  # [MHz]
          "readout_freq": 18.1,  # [MHz]

          "adc_trig_offset": 120,  # [Clock ticks]
          # Try varying adc_trig_offset from 100 to 220 clock ticks

          "soft_avgs": 1
          # Try varying soft_avgs from 1 to 200 averages

          }

###################
# Try it yourself !
###################

pulse_len_list = np.arange(100, 200, 1)
bufi_list = np.zeros((len(pulse_len_list), config["reps"]))
bufq_list = np.zeros((len(pulse_len_list), config["reps"]))
i = 0
while i < len(pulse_len_list):
    try:
        config["length"] = pulse_len_list[i]
        config["readout_length"] = pulse_len_list[i]
        prog = LoopbackProgram(soccfg, config)
        avg_i, avg_q = prog.acquire(soc, load_pulses=True, progress=False, debug=False)
        bufi_list[i] = prog.di_buf[0]
        bufq_list[i] = prog.dq_buf[0]
        i+=1
        print(i)
    except RuntimeError:
        print(i, "!!!!!!!!")



avgi = np.average(bufi_list, axis=1)
avgq = np.average(bufq_list, axis=1)
stdr_i = np.std(bufi_list, axis=1)
stdr_q = np.std(bufq_list, axis=1)



plt.figure()
plt.title("avg")
plt.plot(pulse_len_list, avgi)
plt.plot(pulse_len_list, avgq)

plt.figure()
plt.title("std err")
plt.plot(pulse_len_list, stdr_i)
plt.plot(pulse_len_list, stdr_q)
plt.xlabel("pulse (integration) length, clock cycles")

plt.figure()
plt.title("avg/stderr")
plt.plot(pulse_len_list, abs(avgi/stdr_i))
plt.plot(pulse_len_list, abs(avgq/stdr_q))
plt.xlabel("pulse (integration) length, clock cycles")


