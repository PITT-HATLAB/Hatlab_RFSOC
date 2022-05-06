from proxy.socProxy import soccfg, soc
from qick import *
import matplotlib.pyplot as plt
import numpy as np

from helpers.pulseConfig import set_pulse_registers_IQ

from qubitMSMT.config import config

class LoopbackProgram(AveragerProgram):
    def initialize(self):
        cfg = self.cfg

        self.declare_gen(ch=cfg["res_ch_I"], nqz=cfg["res_nzq_I"])  # resonator drive I
        self.declare_gen(ch=cfg["res_ch_Q"], nqz=cfg["res_nzq_Q"])  # resonator drive Q

        self.declare_readout(ch=cfg["ro_ch"], length=cfg["readout_length"],freq=cfg["res_freq"], gen_ch=cfg["res_ch_I"])

        res_freq = self.freq2reg(cfg["res_freq"], gen_ch=cfg["res_ch_I"], ro_ch=cfg["ro_ch"])  # convert frequency to dac frequency (ensuring it is an available adc frequency)

        set_pulse_registers_IQ(self, cfg["res_ch_I"], cfg["res_ch_Q"], cfg["skewPhase"],  cfg["IQScale"],
                               style="const", freq=res_freq, phase=cfg["res_phase"], gain=cfg["res_gain"],
                                length=cfg["res_length"])

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


readout_cfg = {
    "reps": 20000,  #
    "readout_length": 1020,  # [clock ticks]

    "res_freq": 90,  # [MHz]
    # "res_gain": 25000,  # [DAC units]
    "res_length": 500,  # [clock ticks]

    "rounds": 1,
    "relax_delay": 250  # [us]
}

config = {**config, **readout_cfg}

prog = LoopbackProgram(soccfg, config)
avgi, avgq = prog.acquire(soc, load_pulses=True, progress=True, debug=False)

print("plotting")
# Plot results.
fig, ax = plt.subplots()
hist = ax.hist2d(prog.di_buf[0], prog.dq_buf[0], bins=101)#, range=[[-400, 400], [-400, 400]])
ax.set_aspect(1)
fig.colorbar(hist[3])
plt.show()
