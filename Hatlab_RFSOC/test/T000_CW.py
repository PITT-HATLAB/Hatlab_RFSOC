from collections import OrderedDict

from Hatlab_RFSOC.proxy import getSocProxy
from qick import *

soc, soccfg = getSocProxy()
class CWProgram(AveragerProgram):
    def initialize(self):
        cfg = self.cfg

        self.declare_gen(ch=cfg["dac_ch"], nqz=cfg["dac_nzq"])  # resonator drive I

        self.declare_readout(ch=cfg["ro_ch"], length=cfg["readout_length"], freq=cfg["dac_freq"],
                             gen_ch=cfg["dac_ch"])

        dac_freq = self.freq2reg(cfg["dac_freq"], gen_ch=cfg["dac_ch"], ro_ch=cfg[
            "ro_ch"])  # convert frequency to dac frequency (ensuring it is an available adc frequency)

        mode = ["oneshot", "periodic"][cfg["out_en"]]
        stdysel = ["zero", "last"][cfg["out_en"]]
        self.set_pulse_registers(cfg["dac_ch"],
                               style="const", freq=dac_freq, phase=cfg["dac_phase"], gain=cfg["dac_gain"],
                               length=cfg["dac_length"], mode=mode, stdysel=stdysel)

        self.synci(200)  # give processor some time to configure pulses

    def body(self):
        cfg = self.cfg

        self.trigger([cfg["ro_ch"]], adc_trig_offset=cfg["adc_trig_offset"])
        self.pulse(ch=cfg["dac_ch"], t=0)
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
    "dac_ch": 6,
    "ro_ch": 0,

    "dac_nzq": 1
}

dacConfig = OrderedDict({
    "dac_freq": 90,  # [MHz]
    "dac_gain": 15000,  # [DAC units]
    "dac_length": 500,  # [clock ticks] (actually doesn't mean anything in CW mode)
    "dac_phase": 0  # [deg]
})


def manualTune(out_en=1, **kwargs):
    dacConfig.update(kwargs)
    config = {**hw_cfg, **dacConfig, **expt_config, "out_en": out_en}
    prog = CWProgram(soccfg, config)
    prog.acquire_round(soc, load_pulses=True, progress=True)



if __name__ == "__main__":
    manualTune(out_en=0, dac_freq=5500, dac_nzq=2, dac_gain=30000)
