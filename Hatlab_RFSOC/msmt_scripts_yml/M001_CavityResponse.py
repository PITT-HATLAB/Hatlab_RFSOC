import matplotlib.pyplot as plt
import numpy as np

import programs as msmt
from Hatlab_RFSOC.proxy import getSocProxy
import Hatlab_RFSOC.helpers.plotData as plotdata

from M000_ConfigSel import get_cfg_info
config, info = get_cfg_info()

if __name__ == "__main__":
    soc, soccfg = getSocProxy(info["PyroServer"])
    config["soft_avgs"] = 1000
    config["reps"] = 1

    prog = msmt.CavityResponseProgram(soccfg, config)
    mux_iq_list = prog.acquire_decimated(soc, load_pulses=True, progress=True, debug=False)

    # Plot results.

    plotdata.plotIQTrace(mux_iq_list, [0])





