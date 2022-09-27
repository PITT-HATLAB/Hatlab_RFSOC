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
        "t_start": 0.05,
        "t_stop": 150.05,
        "t_expts": 101,

        "reps": 300,
        "rounds": 1,

        "sel_msmt":False
    }
    config.update(expt_cfg)  # combine configs

    prog = msmt.T1Program(soccfg, config)
    x_pts, avgi, avgq = prog.acquire(soc, load_pulses=True, progress=True, debug=False)
    sweepTime = get_sweep_vals(config, "t")

    # plot IQ result
    plotData.plotAvgIQresults(sweepTime, avgi, avgq, title="T1",xlabel="time (us)", ylabel="Qubit IQ", ro_chs=[ADC_idx])

    # fit result
    t1Decay = qfr.T1Decay(sweepTime, avgi[ADC_idx][0] + 1j * avgq[ADC_idx][0])
    t1Result = t1Decay.run(rot_result=info["rotResult"])
    t1Result.plot()

    # save data to ddh5
    quick_save(info["dataPath"], f"{info['sampleName']}_T1", avgi, avgq, config=config, sweepTime=sweepTime)



