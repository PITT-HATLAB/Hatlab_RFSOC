import matplotlib.pyplot as plt
import numpy as np

import programs as msmt
from Hatlab_RFSOC.proxy import getSocProxy
from Hatlab_RFSOC.data import quick_save
from Hatlab_RFSOC.helpers import get_sweep_vals, plotData
from Hatlab_DataProcessing.analyzer import qubit_functions_rot as qfr

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
              "rounds": 1,
              "relax_delay": 200 #[us]
             }


    config.update(expt_cfg)

    prog=msmt.PulseSpecProgram(soccfg, config)
    expt_pts, avgi, avgq = prog.acquire(soc, load_pulses=True,progress=True, debug=False)
    sweepFreq = get_sweep_vals(expt_cfg, "f")

    # plot IQ result
    plotData.plotAvgIQresults(sweepFreq, avgi, avgq, title="Pulse Spectroscopy",
                              xlabel="Drive Frequency (MHz)", ylabel="Qubit IQ", ro_chs=[ADC_idx])

    # fit result
    specFit = qfr.PulseSpec(sweepFreq, avgi[ADC_idx][0] + 1j * avgq[ADC_idx][0])
    specResult = specFit.run()
    specResult.plot()

    # save data to ddh5
    quick_save(info["dataPath"], f"{info['sampleName']}_pulseSpec", avgi, avgq, config=config, sweepFreq=sweepFreq)


    # #-------------an example that shows how to load data from ddh5 file -----------------------
    # from Hatlab_RFSOC.data import DataFromQDDH5
    # data = DataFromQDDH5(r"L:\Data\SNAIL_Pump_Limitation\2022-09-22\\ChenWang_test.ddh5")
    # plt.figure(1)
    # plt.subplot(111,title="Qubit Spectroscopy", xlabel="Qubit Frequency (MHz)", ylabel="Qubit IQ")
    # plt.plot(data.axes["sweepFreq"]["values"], np.real(data.avg_iq["ro_0"])[:,0],'o-', markersize = 1)
    # plt.plot(data.axes["sweepFreq"]["values"], np.imag(data.avg_iq["ro_0"])[:,0],'o-', markersize = 1)
    # plt.show()

