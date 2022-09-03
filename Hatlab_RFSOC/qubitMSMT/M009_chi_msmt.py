# todo: write this as a function or class

from Hatlab_RFSOC.proxy import getSocProxy
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

from Hatlab_RFSOC.qubitMSMT.exampleConfig import config, PyroServer
from M003_amplituderabi import AmplitudeRabiProgram
from Hatlab_DataProcessing.fitter import cavity_functions_hanger as cfr

from instrumentserver.client import Client

cli = Client()
SC_C = cli.get_instrument("SC_C")

soc, soccfg = getSocProxy(PyroServer)

expt_cfg={
    "start":0,
    "step":config["pi_gain"],
    "expts":2,
    "reps": 500,
    "relax_delay":300
    }

config = {**config, **expt_cfg}


freqList = np.linspace(-1e6, 1e6, 201) + 7.26204058e9
avgi_array = np.zeros((2, len(freqList)))
avgq_array = np.zeros((2, len(freqList)))

for i, freq in enumerate(tqdm(freqList)):
    SC_C.frequency(freq)
    prog = AmplitudeRabiProgram(soccfg, config)
    x_pts, avgi, avgq = prog.acquire(soc, load_pulses=True, progress=True, debug=False)
    avgi_array[:,i], avgq_array[:, i] = avgi[0][0], avgq[0][0]


g_trace = avgi_array[0] - 1j*avgq_array[0]
e_trace = avgi_array[1] - 1j*avgq_array[1]

# --------- processing shits-------------------------
plt.figure()
plt.subplot(111, title= f"drive ge pi pulse, vary f cavity", xlabel="freq", ylabel="I Q" )
plt.plot(freqList, avgi_array[0],'o-', markersize = 1)
plt.plot(freqList, avgq_array[0],'o-', markersize = 1)
plt.plot(freqList, avgi_array[1],'*-', markersize = 1)
plt.plot(freqList, avgq_array[1],'*-', markersize = 1)
plt.figure(figsize=(13,5))
plt.subplot(121, title= f"drive ge pi pulse, vary f cavity", xlabel="freq", ylabel="amp" )
plt.plot(freqList, np.abs(g_trace),'o-', markersize = 1)
plt.plot(freqList, np.abs(e_trace),'o-', markersize = 1)
plt.subplot(122, title= f"drive ge pi pulse, vary f cavity", xlabel="freq", ylabel="phase" )
plt.plot(freqList, np.unwrap(np.angle(g_trace)),'o-', markersize = 1)
plt.plot(freqList, np.unwrap(np.angle(e_trace)),'o-', markersize = 1)
chi_fromMag = np.abs(freqList[np.argmin(np.abs(g_trace))]-freqList[np.argmin(np.abs(e_trace))])/1e6
print(f"chi from magnitude: {chi_fromMag} MHz")


from importlib import reload

reload(cfr)


cavRef_g = cfr.CavHangerPhaseOnly(freqList, g_trace, conjugate=True)
futCavRef_g = cavRef_g.run()
cavRef_e = cfr.CavHangerPhaseOnly(freqList, e_trace, conjugate=True)
futCavRef_e = cavRef_e.run()#params={"f0":lmfit.Parameter("f0", value=6.709e9), "eDelay":lmfit.Parameter("eDelay", value=-2.365e-11)})


plt.figure()
plt.plot(freqList, futCavRef_g.lmfit_result.data, ".")
plt.plot(freqList, futCavRef_g.lmfit_result.best_fit)
plt.plot(freqList, futCavRef_e.lmfit_result.data, ".")
plt.plot(freqList, futCavRef_e.lmfit_result.best_fit)

print("chi from fitting:", np.abs(futCavRef_g.f0-futCavRef_e.f0)/1e6, "MHz")
print("best res freq:", (futCavRef_g.f0+futCavRef_e.f0)/2/1e9, "GHz")

print(cavRef_e.guess(cavRef_e.coordinates, cavRef_e.data))

# ----- save data to pc --------
# data_temp = {"x_data":freqList, "i_data": avgi_array[0], "q_data": avgq_array[0]}
# saveData(data_temp, sampleName+"_chi_g", dataPath)
# data_temp = {"x_data":freqList, "i_data": avgi_array[1], "q_data": avgq_array[1]}
# saveData(data_temp, sampleName+"_chi_e", dataPath)

