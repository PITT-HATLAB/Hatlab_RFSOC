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
        "g_start": -30000, # MHz
        "g_stop": 30000,
        "g_expts": 201,

        "reps": 600,
        "sel_msmt": False,
        "prepare_g": False, # start qubit at g state, should only be True for temperature msmt,
        "flip_back_g": True # when true, add an extra pipulse after ef_rabi. Usually the cavity is driven at best g/e
                            # separation frequency, so setting this to True gives a better final msmt resolution.
        }

    config.update(expt_cfg)

    prog=msmt.EfRabiProgram(soccfg, config)
    expt_pts, avgi, avgq = prog.acquire(soc, load_pulses=True,progress=True, debug=False)
    sweepGain = get_sweep_vals(config, "g")

    # save data to ddh5
    quick_save(info["dataPath"], f"{info['sampleName']}_EfRabi", avgi, avgq, prog.di_buf_p, prog.dq_buf_p, config=config, sweepGain=sweepGain)

    if not config["sel_msmt"]:
        # plot IQ result
        plotData.plotAvgIQresults(sweepGain, avgi, avgq, title="ef Rabi",
                                  xlabel="Drive Gain (DAC)", ylabel="Qubit IQ")
        # fit IQ result
        fit = qfr.PiPulseTuneUp(sweepGain, avgi[ADC_idx][0] + 1j * avgq[ADC_idx][0])
        fitResult = fit.run()
        fitResult.plot()

    else:
        # post select
        pass
        bufi, bufq = prog.di_buf_p[ADC_idx], prog.dq_buf_p[ADC_idx]
        g_pct, I_vld, Q_vld, selData = simpleSelection_1Qge(bufi, bufq) # note that we can't use g_pct here
        fit = qf.PiPulseTuneUp(sweepGain, g_pct)
        fitResult = fit.run()
        fitResult.plot()

