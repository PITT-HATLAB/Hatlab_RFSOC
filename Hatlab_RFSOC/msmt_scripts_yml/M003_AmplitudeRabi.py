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
        "g_start": -30000,
        "g_stop": 30000,
        "g_expts": 101,

        "reps": 500,

        "sel_msmt": True
    }
    config.update(expt_cfg)  # combine configs

    prog = msmt.AmplitudeRabiProgram(soccfg, config)
    x_pts, avgi, avgq = prog.acquire(soc, load_pulses=True, progress=True, debug=False)
    sweepGain = get_sweep_vals(config, "g")

    # save data to ddh5
    quick_save(info["dataPath"], f"{info['sampleName']}_ampRabi", avgi, avgq, prog.di_buf_p, prog.dq_buf_p, config=config, sweepGain=sweepGain)

    # data process
    if not config["sel_msmt"]:
        # plot IQ result
        plotData.plotAvgIQresults(sweepGain, avgi, avgq, title="Amplitude Rabi",
                                  xlabel="Gain (DAC)", ylabel="Qubit IQ")

        # fit IQ result
        fit = qfr.PiPulseTuneUp(sweepGain, avgi[ADC_idx][0] + 1j * avgq[ADC_idx][0])
        fitResult = fit.run()
        fitResult.plot()
        fitResult.print_ge_rotation()

    else:
        # post select and fit g_pct
        bufi, bufq = prog.di_buf_p[ADC_idx], prog.dq_buf_p[ADC_idx]
        g_pct, I_vld, Q_vld, selData = simpleSelection_1Qge(bufi, bufq)
        fit = qf.PiPulseTuneUp(sweepGain, g_pct)
        fitResult = fit.run()
        fitResult.plot(xlabel="Gain (DAC)", ylabel="g pct")


    # # ----- slider hist2d ----------
    # from Hatlab_DataProcessing.slider_plot.sliderPlot import sliderHist2d
    # sld = sliderHist2d(bufi.T, bufq.T, {"gain":sweepGain}, bins=101)