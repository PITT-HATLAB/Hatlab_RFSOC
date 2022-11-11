import matplotlib.pyplot as plt
import numpy as np

import programs as msmt
from Hatlab_RFSOC.proxy import getSocProxy
from Hatlab_RFSOC.data import quick_save
from Hatlab_RFSOC.helpers import get_sweep_vals, plotData
from Hatlab_DataProcessing.analyzer import qubit_functions_rot as qfr
from Hatlab_DataProcessing.fitter import qubit_functions as qf
from Hatlab_DataProcessing.post_selection import simpleSelection_1Qge

from M000_ConfigSel import get_cfg_info
config, info = get_cfg_info()

if __name__ == "__main__":
    soc, soccfg = getSocProxy(info["PyroServer"])
    ADC_idx = info.get("ADC_idx", 0)

    expt_cfg={"f_start": 4865, # MHz
              "f_stop": 4885,
              "f_expts": 201,

              "prob_length": 3,
              "prob_gain":30,

              "reps": 300,
              "sel_msmt": False,

              "relax_delay": 200 #[us]
             }


    config.update(expt_cfg)

    prog=msmt.PulseSpecProgram(soccfg, config)
    expt_pts, avgi, avgq = prog.acquire(soc, load_pulses=True,progress=True, debug=False)
    sweepFreq = get_sweep_vals(expt_cfg, "f")

    # save data to ddh5
    quick_save(info["dataPath"], f"{info['sampleName']}_pulseSpec", avgi, avgq, prog.di_buf_p, prog.dq_buf_p, config=config, sweepFreq=sweepFreq)

    if not config["sel_msmt"]:
        # plot IQ result
        plotData.plotAvgIQresults(sweepFreq, avgi, avgq, title="Pulse Spectroscopy",
                                  xlabel="Drive Frequency (MHz)", ylabel="Qubit IQ")

        # fit result
        fit = qfr.PulseSpec(sweepFreq, avgi[ADC_idx][0] + 1j * avgq[ADC_idx][0])
        fitResult = fit.run()
        fitResult.plot()
    else:
        # post select and fit g_pct
        bufi, bufq = prog.di_buf_p[ADC_idx], prog.dq_buf_p[ADC_idx]
        g_pct, I_vld, Q_vld, selData = simpleSelection_1Qge(bufi, bufq)
        fit = qf.PulseSpec(sweepFreq, g_pct)
        fitResult = fit.run()
        fitResult.plot(xlabel="Gain (DAC)", ylabel="g pct")




    # #-------------an example that shows how to load data from ddh5 file -----------------------
    # from Hatlab_RFSOC.data import DataFromQDDH5
    # data = DataFromQDDH5(r"L:\Data\SNAIL_Pump_Limitation\2022-09-22\\ChenWang_test.ddh5")
    # plt.figure(1)
    # plt.subplot(111,title="Qubit Spectroscopy", xlabel="Qubit Frequency (MHz)", ylabel="Qubit IQ")
    # plt.plot(data.axes["sweepFreq"]["values"], np.real(data.avg_iq["ro_0"])[:,0],'o-', markersize = 1)
    # plt.plot(data.axes["sweepFreq"]["values"], np.imag(data.avg_iq["ro_0"])[:,0],'o-', markersize = 1)
    # plt.show()

