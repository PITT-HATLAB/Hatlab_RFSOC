from collections import OrderedDict
from pprint import pprint

from proxy.socProxy import soccfg, soc
from qick import *
import matplotlib.pyplot as plt
import numpy as np

from Hatlab_RFSOC.helpers.pulseConfig import set_pulse_registers_IQ


class CWProgram(AveragerProgram):
    def initialize(self):
        cfg = self.cfg

        self.declare_gen(ch=cfg["res_ch_I"], nqz=cfg["res_nzq_I"])  # resonator drive I
        self.declare_gen(ch=cfg["res_ch_Q"], nqz=cfg["res_nzq_Q"])  # resonator drive Q
        self.declare_readout(ch=cfg["ro_ch"], length=cfg["readout_length"], freq=cfg["res_freq"],
                             gen_ch=cfg["res_ch_I"])

        res_freq = self.freq2reg(cfg["res_freq"], gen_ch=cfg["res_ch_I"], ro_ch=cfg[
            "ro_ch"])  # convert frequency to dac frequency (ensuring it is an available adc frequency)

        mode = ["oneshot", "periodic"][cfg["out_en"]]
        stdysel = ["zero", "last"][cfg["out_en"]]
        set_pulse_registers_IQ(self, cfg["res_ch_I"], cfg["res_ch_Q"], cfg["skewPhase"], cfg["IQScale"],
                               style="const", freq=res_freq, phase=cfg["res_phase"], gain=cfg["res_gain"],
                               length=cfg["res_length"], mode=mode, stdysel=stdysel)

        self.synci(200)  # give processor some time to configure pulses

    def body(self):
        cfg = self.cfg

        self.trigger([cfg["ro_ch"]], adc_trig_offset=cfg["adc_trig_offset"])
        self.pulse(ch=cfg["res_ch_I"], t=0)
        self.pulse(ch=cfg["res_ch_Q"], t=0)
        self.wait_all()
        # tProc should wait for the readout to complete.
        # This prevents loop counters from getting incremented before the data is available.
        self.sync_all(self.us2cycles(cfg["relax_delay"]))


expt_config = {
    "reps": 10,  # --Fixed
    "readout_length": 1000,  # [Clock ticks]
    "adc_trig_offset": 0,  # [Clock ticks]
    "soft_avgs": 1,
    "relax_delay": 10
}

hw_cfg = {
    "res_ch_I": 5,
    "res_ch_Q": 6,
    "ro_ch": 0,

    "res_nzq_I": 1,
    "res_nzq_Q": 1,
}

mixerConfig = OrderedDict({
    "res_freq": 90,  # [MHz]
    "res_gain": 15000,  # [DAC units]
    "res_length": 500,  # [clock ticks] (actually doesn't mean anything in CW mode)
    "res_phase": 0,  # [deg]
    "skewPhase": 83,  # [Degrees]
    "IQScale": 1.03
})


def manualTune(out_en=1, **kwargs):
    mixerConfig.update(kwargs)
    config = {**hw_cfg, **mixerConfig, **expt_config, "out_en": out_en}
    prog = CWProgram(soccfg, config)
    prog.acquire_round(soc, load_pulses=True, progress=True)
    if config["out_en"] == 0:
        pprint(dict(mixerConfig), sort_dicts=False)


if __name__ == "__main__":
    manualTune(out_en=1, skewPhase=90, IQScale=1)
