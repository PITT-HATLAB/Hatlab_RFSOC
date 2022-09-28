import matplotlib.pyplot as plt
import numpy as np

from Hatlab_DataProcessing.analyzer import qubit_functions_rot as qfr
from Hatlab_DataProcessing.fitter import qubit_functions as qf
from Hatlab_DataProcessing.post_selection import simpleSelection_1Qge

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

        "reps": 500,
        "sel_msmt": False
        }

    config.update(expt_cfg)

    prog=msmt.EfPulseSpecProgram(soccfg, config)
    expt_pts, avgi, avgq = prog.acquire(soc, load_pulses=True,progress=True, debug=False)
    sweepFreq = get_sweep_vals(config, "f") + config.get("qubit_mixer_freq", 0)

    # save data to ddh5
    quick_save(info["dataPath"], f"{info['sampleName']}_EfPulseSpec", avgi, avgq, prog.di_buf_p, prog.dq_buf_p, config=config, sweepFreq=sweepFreq)

    if not config["sel_msmt"]:
        # plot IQ result
        plotData.plotAvgIQresults(sweepFreq, avgi, avgq, title="ef Pulse Spectroscopy",
                                  xlabel="Drive Frequency (MHz)", ylabel="Qubit IQ")
        IQ_data = avgi[ADC_idx][0] + 1j * avgq[ADC_idx][0]
    else:
        # post select
        bufi, bufq = prog.di_buf_p[ADC_idx], prog.dq_buf_p[ADC_idx]
        g_pct, I_vld, Q_vld, selData = simpleSelection_1Qge(bufi, bufq) # note that we can't use g_pct here
        IQ_data = np.array([np.average(v) for v in I_vld]) + 1j * np.array([np.average(v) for v in Q_vld])

    # fit result
    fit = qfr.PulseSpec(sweepFreq, IQ_data)
    fitResult = fit.run()
    fitResult.plot()