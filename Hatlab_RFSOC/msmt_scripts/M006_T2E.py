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

    expt_cfg = {
        "t_start": 0.05,
        "t_stop": 40.05,
        "t_expts": 101,

        "reps": 500,

        "sel_msmt": False
    }
    config.update(expt_cfg)  # combine configs

    prog = msmt.T2EProgram(soccfg, config)
    _, avgi, avgq = prog.acquire(soc, load_pulses=True, progress=True, debug=False)
    sweepTime = get_sweep_vals(config, "t")

    # save data to ddh5
    quick_save(info["dataPath"], f"{info['sampleName']}_T2E", avgi, avgq, prog.di_buf_p, prog.dq_buf_p, config=config, sweepTime=sweepTime)

    # data process
    if not config["sel_msmt"]:
        # plot IQ result
        plotData.plotAvgIQresults(sweepTime, avgi, avgq, title="T2E",xlabel="time (us)", ylabel="Qubit IQ")

        # fit result
        fit = qfr.T1Decay(sweepTime, avgi[ADC_idx][0] + 1j * avgq[ADC_idx][0])
        fitReresult = fit.run(info["rotResult"])
        fitReresult.plot()

    else:
        # post select and fit g_pct
        bufi, bufq = prog.di_buf_p[ADC_idx], prog.dq_buf_p[ADC_idx]
        g_pct, I_vld, Q_vld, selData = simpleSelection_1Qge(bufi, bufq)
        fit = qf.T1Decay(sweepTime, g_pct)
        fitReresult = fit.run()
        fitReresult.plot(xlabel="Gain (DAC)", ylabel="g pct")