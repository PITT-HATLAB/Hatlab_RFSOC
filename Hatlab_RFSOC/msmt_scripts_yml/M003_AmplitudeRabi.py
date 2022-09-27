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

    expt_cfg = {
        "g_start": -30000,
        "g_stop": 30000,
        "g_expts": 101,

        "reps": 500,
        "rounds": 1,

        "sel_msmt": False
    }
    config.update(expt_cfg)  # combine configs

    prog = msmt.AmplitudeRabiProgram(soccfg, config)
    x_pts, avgi, avgq = prog.acquire(soc, load_pulses=True, progress=True, debug=False)
    sweepGain = get_sweep_vals(config, "g")

    # plot IQ result
    plotData.plotAvgIQresults(sweepGain, avgi, avgq, title="Amplitude Rabi",
                              xlabel="Gain (DAC)", ylabel="Qubit IQ", ro_chs=[ADC_idx])

    # fit result
    piPul = qfr.PiPulseTuneUp(sweepGain, avgi[ADC_idx][0] + 1j * avgq[ADC_idx][0])
    piResult = piPul.run()
    piResult.plot()
    piResult.print_ge_rotation()

    # histogram
    fig, ax = plt.subplots()
    hist = ax.hist2d(prog.di_buf[ADC_idx], prog.dq_buf[ADC_idx], bins=101)#, range=[[-400, 400], [-400, 400]])
    ax.set_aspect(1)
    fig.colorbar(hist[3])
    plt.show()

    # # ----- slider hist2d ----------
    # from Hatlab_DataProcessing.slider_plot.sliderPlot import sliderHist2d
    # sld = sliderHist2d(prog.di_buf_p[ADC_idx].T, prog.dq_buf_p[ADC_idx].T, {"amp":x_pts}, bins=101)

    # save data to ddh5
    quick_save(info["dataPath"], f"{info['sampleName']}_ampRabi", avgi, avgq, config=config, sweepGain=sweepGain)

