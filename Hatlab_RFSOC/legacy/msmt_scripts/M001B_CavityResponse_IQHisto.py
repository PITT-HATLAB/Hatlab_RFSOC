from importlib import reload
import M000_ConfigSel; reload(M000_ConfigSel) # just to make sure the data in config.py will update when running in same console

from M001_CavityResponse import CavityResponseProgram
from Hatlab_RFSOC.proxy import getSocProxy
import Hatlab_RFSOC.helpers.plotData as plotdata

from M000_ConfigSel import config, info

if __name__ == "__main__":
    soc, soccfg = getSocProxy(info["PyroServer"])

    readout_cfg = {
        "reps": 15000,
        "rounds": 1,
    }
    config.update(readout_cfg)

    prog = CavityResponseProgram(soccfg, config)
    mux_iq_points = prog.acquire(soc, load_pulses=True, progress=True, debug=False)
    di_buf, dq_buf = prog.di_buf, prog.dq_buf
    plotdata.plotIQHist2d(di_buf, dq_buf, ro_chs=[0, 1], bins=101)





