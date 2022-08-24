from importlib import reload
import M000_ConfigSel; reload(M000_ConfigSel) # just to make sure the data in config.py will update when running in same console

import matplotlib.pyplot as plt
import numpy as np

from Hatlab_RFSOC.proxy import getSocProxy
import Hatlab_RFSOC.helpers.plotData as plotdata

from M000_ConfigSel import config, info
from qick.averager_program import AveragerProgram


class CavityResponseProgram(AveragerProgram):
    def initialize(self):
        cfg = self.cfg
        # declare muxed generator and readout channels
        self.res_ch = self.cfg["gen_chs"]["muxed_res"]["ch"]
        self.declare_gen(**cfg["gen_chs"]["muxed_res"])
        for ro_cfg in cfg["ro_chs"].values():
            self.declare_readout(**ro_cfg)

        # set readout pulse registers
        self.set_pulse_registers(ch=self.res_ch, style="const", length=cfg["res_length"], mask=[0, 1, 2, 3])

        self.synci(200)  # give processor some time to configure pulses

    def body(self):
        self.measure(pulse_ch=self.res_ch,
                     adcs=self.ro_chs,
                     pins=[0],
                     adc_trig_offset=self.cfg["adc_trig_offset"],
                     wait=True,
                     syncdelay=self.us2cycles(self.cfg["relax_delay"]))



if __name__ == "__main__":
    soc, soccfg = getSocProxy(info["PyroServer"])
    config["soft_avgs"] = 3000
    config["reps"] = 1

    prog = CavityResponseProgram(soccfg, config)
    mux_iq_list = prog.acquire_decimated(soc, load_pulses=True, progress=True, debug=False)

    # Plot results.
    plotdata.plotIQTrace(mux_iq_list, [0, 1])


