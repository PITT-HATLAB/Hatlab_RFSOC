import json
import numpy as np
from matplotlib import pyplot as plt
from Hatlab_DataProcessing.analyzer import qubit_functions_rot as qfr

from Hatlab_RFSOC.qubitMSMT.config import config, rotResult, dataPath, sampleName

def loadData(fileName, filePath):
    with open(filePath+fileName, 'r') as infile:
        dataDict = json.load(infile)
        for k, v in dataDict.items():
            dataDict[k] = np.array(v)
    return dataDict


dataPath = r"L:\Data\WISPE\LL_WISPE\s6\cooldown_20220401\\"
sampleName = "LL_Wispe_0401_candle2"

pi_ef_rabi = loadData(sampleName + "_ef_rabi", dataPath)
pi_ge_amplitude = loadData(sampleName + "_ge_amplitude", dataPath)

piPulef_e = qfr.PiPulseTuneUp(pi_ef_rabi["x_data"], pi_ef_rabi["i_data"] + 1j * pi_ef_rabi["q_data"])
piResultef_e = piPulef_e.run()

piPulef_g = qfr.PiPulseTuneUp(pi_ge_amplitude["x_data"], pi_ge_amplitude["i_data"] + 1j * pi_ge_amplitude["q_data"])
piResultef_g = piPulef_g.run()

piResultef_e.plot()
piResultef_g.plot()

pi_e = np.abs(piResultef_e.lmfit_result.params["A"].value)
pi_g = np.abs(piResultef_g.lmfit_result.params["A"].value)
delta_pi_e = np.abs(piResultef_e.lmfit_result.params["A"].stderr)
delta_pi_g = np.abs(piResultef_g.lmfit_result.params["A"].stderr)

print("e_pct:", pi_g / (pi_g + pi_e))
print("delta_e_pct:", (delta_pi_g/pi_g) + np.sqrt(((delta_pi_g/pi_g))**2+(delta_pi_e/pi_e)**2))


