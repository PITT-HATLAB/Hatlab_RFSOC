import json
import matplotlib.pyplot as plt
import numpy as np
from tqdm import  tqdm
import time
from datetime import  datetime

saveDir = rf'L:\Data\SNAIL_Pump_Limitation\RT_test\ZCU216_PhaseRest_Mix2Gens\2022-11-02\\'
data = json.load(open(saveDir + "test"))

t_list = np.array(data["t_list"])
i_list = np.array(data["i_list"])
q_list = np.array(data["q_list"])


data = {"t_list": t_list, "i_list": i_list, "q_list": q_list}

datetime_list = [datetime.fromtimestamp(t_) for t_ in t_list]
mean_abs = np.average(np.abs(i_list + 1j* q_list))

plt.figure()
plt.plot(datetime_list, i_list)
plt.plot(datetime_list, q_list)

fig, ax1 = plt.subplots(figsize=(10, 7))
fig.suptitle("demodulated signal")
ax2 = ax1.twinx()
ax2.plot(datetime_list, np.unwrap(np.angle(i_list + 1j* q_list))/np.pi*180, "C0")
ax1.plot(datetime_list, (np.abs(i_list + 1j* q_list) - mean_abs)/mean_abs, 'C1')
ax2.set_xlabel('time')
ax2.set_ylabel('IQ phase (deg)', color='C0', size=18)
ax1.set_ylabel('IQ amp (relative to average)', color='C1', size=18)
plt.gcf().autofmt_xdate()



