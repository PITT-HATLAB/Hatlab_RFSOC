"""
To demonstrate how to run a 2D sweep of driving length and freq, in which the length sweep is done in qick, and freq
sweep is done in python. The experiment is managed with the Experiment class, which saves data to DataDict H5 after each
qick sweep.
"""

from importlib import reload
import M000_ConfigSel; reload(M000_ConfigSel) # just to make sure the data in config.py will update when running in same console

import numpy as np
from plottr.data import DataDict

from Hatlab_RFSOC.proxy import getSocProxy
from experiment import Experiment
from Hatlab_RFSOC.helpers import get_sweep_vals


from M000_ConfigSel import config, info

from M003B_LengthRabi import LengthRabiProgram

if __name__ == "__main__":
    soc, soccfg = getSocProxy(info["PyroServer"])
    ADC_idx = info["ADC_idx"]

    expt_cfg = {
        "l_start": 0.01,
        "l_stop": 1.01,
        "l_expts": 101,
        "gain": 3000,

        "reps": 200,
        "rounds": 1,

        "relax_delay": 200,  # [us]
        "sel_msmt": True
    }
    config.update(expt_cfg)  # combine configs

    pumpFreqList = np.linspace(4860, 4880, 21)

    inner_sweeps = DataDict(length={"unit": "us", "values": get_sweep_vals(expt_cfg, "l")})
    outer_sweeps = DataDict(freq={"unit": "MHz", "values": pumpFreqList})

    expt = Experiment(LengthRabiProgram, config, info, inner_sweeps, outer_sweeps,
                      data_base_dir=r"L:\Data\SNAIL_Pump_Limitation\test\\", data_file_name="lengthRabi_sweepFreq")

    for i, pf in enumerate(pumpFreqList):
        print(i)
        expt.cfg["q_pulse_cfg"]["ge_freq"] = pf
        x_pts, avgi, avgq, di_buf, dq_buf = expt.run(save_data=True, save_buf=True,
                                                     readouts_per_experiment=int(expt_cfg["sel_msmt"])+1, freq=pf)



