import matplotlib.pyplot as plt
from Hatlab_RFSOC.helpers import DataFromQDDH5
from Hatlab_DataProcessing.post_selection import simpleSelection_1Qge

qdd = DataFromQDDH5(r"L:\Data\SNAIL_Pump_Limitation\test\2022-08-25\lengthRabi_sweepFreq-7.ddh5")
qdd.reorder_data(flatten_sweep=True)
freqList = qdd.axes["freq"]["values"]
lengthList = qdd.axes["length"]["values"]

g_pct, I_vld, Q_vld, selData = simpleSelection_1Qge(qdd.buf_iq["ro_1"].real, qdd.buf_iq["ro_1"].imag)


plt.figure()

plt.pcolormesh(freqList, lengthList, g_pct.reshape(len(freqList), len(lengthList)).T, shading="auto")
plt.colorbar()






