import matplotlib.pyplot as plt
import numpy as np

from Hatlab_DataProcessing.analyzer import qubit_functions_rot as qfr

from Hatlab_RFSOC.proxy import getSocProxy
from Hatlab_RFSOC.data import quick_save
from Hatlab_RFSOC.helpers import get_sweep_vals, plotData

import programs as msmt

from M000_ConfigSel import get_cfg_info
config, info = get_cfg_info()


if __name__ == "__main__":
    soc, soccfg = getSocProxy(info["PyroServer"])
    ADC_idx = info.get("ADC_idx",0)

    expt_cfg={
        "f_start": 4660, # MHz
        "f_stop": 4710,
        "f_expts": 201,

        "prob_length": 3,
        "prob_gain": 100,

        "reps": 200
        }

    config.update(expt_cfg)

    prog=msmt.EfPulseSpecProgram(soccfg, config)
    expt_pts, avgi, avgq = prog.acquire(soc, load_pulses=True,progress=True, debug=False)
    sweepFreq = get_sweep_vals(config, "f") + config.get("qubit_mixer_freq", 0)

    # plot IQ result
    plotData.plotAvgIQresults(sweepFreq, avgi, avgq, title="ef Pulse Spectroscopy",
                              xlabel="Drive Frequency (MHz)", ylabel="Qubit IQ", ro_chs=[ADC_idx])

    # fit result
    specFit = qfr.PulseSpec(sweepFreq, avgi[ADC_idx][0] + 1j * avgq[ADC_idx][0])
    specResult = specFit.run()
    specResult.plot()

    # save data to ddh5
    quick_save(info["dataPath"], f"{info['sampleName']}_EfPulseSpec", avgi, avgq, config=config, sweepFreq=sweepFreq)