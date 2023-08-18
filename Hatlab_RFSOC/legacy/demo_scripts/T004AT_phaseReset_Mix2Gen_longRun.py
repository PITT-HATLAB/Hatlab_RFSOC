from Hatlab_RFSOC.proxy import getSocProxy
from qick import *
import matplotlib.pyplot as plt
import numpy as np
from tqdm import  tqdm
import time
from datetime import  datetime
soc, soccfg = getSocProxy("myqick216-01")
from T004A_phaseReset_Mix2Gen import LoopbackProgram
from Hatlab_RFSOC.data.data_transfer import saveData

config = {"res_ch_0": 1,  # 1-4 GHz
          "res_ch_1": 3,  # > 6 GHz
          "ro_chs": [0],  #
          "reps": 1000,  #
          "relax_delay": 1.0,  # --us

          "length": 150,  # [Clock ticks]

          "readout_length": 250,  # [Clock ticks]

          "pulse_gain": 30000,  # [DAC units]
          # Try varying pulse_gain from 500 to 30000 DAC units

          "pulse_freq_0": 4000,  # [MHz]
          "pulse_freq_1": 6000,  # [MHz]

          "adc_trig_offset": 100,  # [Clock ticks]
          # Try varying adc_trig_offset from 100 to 220 clock ticks
          "phrst": 1,
          "soft_avgs": 1
          # Try varying soft_avgs from 1 to 200 averages
          }

###################
# Try it yourself !
###################

n_runs = 720*3
t_sleep = 5 #s
t_list = np.zeros(n_runs)
i_list =  np.zeros(n_runs)
q_list =  np.zeros(n_runs)


saveDir = rf'L:\Data\SNAIL_Pump_Limitation\RT_test\ZCU216_PhaseRest_Mix2Gens\2022-11-02\\'
for run in tqdm(range(n_runs)):
    prog = LoopbackProgram(soccfg, config)
    t_list[run] = time.time()
    avg_i, avg_q = prog.acquire(soc, load_pulses=True, progress=False, debug=False)
    time.sleep(t_sleep)
    i_list[run] = avg_i[0]
    q_list[run] = avg_q[0]
    data = {"t_list": t_list, "i_list": i_list, "q_list": q_list}
    saveData({**data, **config}, "test", saveDir)

datetime_list = [datetime.fromtimestamp(t_) for t_ in t_list]

plt.figure()
plt.plot(datetime_list, i_list)
plt.plot(datetime_list, q_list)

fig, ax1 = plt.subplots(figsize=(10, 7))
fig.suptitle("demodulated signal")
ax2 = ax1.twinx()
ax2.plot(datetime_list, np.unwrap(np.angle(i_list + 1j* q_list))/np.pi*180, "C0")
ax1.plot(datetime_list, np.abs(i_list + 1j* q_list), 'C1')
ax2.set_xlabel('time')
ax2.set_ylabel('IQ phase (deg)', color='C0', size=18)
ax1.set_ylabel('IQ amp (relative to average)', color='C1', size=18)
plt.gcf().autofmt_xdate()



