from proxy.socProxy import soccfg, soc
from qick import *
import matplotlib.pyplot as plt
import numpy as np


class LoopbackProgram(AveragerProgram):
    def initialize(self):
        cfg = self.cfg

        # set the nyquist zone
        self.declare_gen(ch=cfg["res_ch"], nqz=1)

        # configure the readout lengths and downconversion frequencies
        self.declare_readout(ch=cfg["ro_ch"], length=self.cfg["readout_length"],
                             freq=self.cfg["pulse_freq"], gen_ch=cfg["res_ch"])

        freq = self.freq2reg(cfg["pulse_freq"], gen_ch=cfg["res_ch"], ro_ch=cfg[
            "ro_ch"])  # convert frequency to dac frequency (ensuring it is an available adc frequency)
        self.set_pulse_registers(ch=cfg["res_ch"], style="const", freq=freq, phase=0, gain=cfg["pulse_gain"],
                                 length=cfg["length"])
        self.synci(200)  # give processor some time to configure pulses

    def body(self):
        self.measure(pulse_ch=self.cfg["res_ch"],
                     adcs=[self.cfg["ro_ch"]],
                     adc_trig_offset=self.cfg["adc_trig_offset"],
                     wait=True,
                     syncdelay=self.us2cycles(self.cfg["relax_delay"]))


class SweepProgram(RAveragerProgram):
    def initialize(self):
        cfg = self.cfg

        # set the nyquist zone
        self.declare_gen(ch=cfg["res_ch"], nqz=1)

        self.r_rp = self.ch_page(self.cfg["res_ch"])  # get register page for res_ch
        self.r_gain = self.sreg(cfg["res_ch"], "gain")  # Get gain register for res_ch

        # configure the readout lengths and downconversion frequencies
        self.declare_readout(ch=cfg["ro_ch"], length=self.cfg["readout_length"],
                             freq=self.cfg["pulse_freq"], gen_ch=cfg["res_ch"])

        # convert frequency to dac frequency (ensuring it is an available adc frequency)
        freq = self.freq2reg(cfg["pulse_freq"], gen_ch=cfg["res_ch"], ro_ch=cfg["ro_ch"])

        self.set_pulse_registers(ch=cfg["res_ch"], style="const", freq=freq, phase=0, gain=cfg["start"],
                                 length=cfg["length"])
        self.synci(200)  # give processor some time to configure pulses

    def body(self):
        self.measure(pulse_ch=self.cfg["res_ch"],
                     adcs=[self.cfg["ro_ch"]],
                     adc_trig_offset=self.cfg["adc_trig_offset"],
                     wait=True,
                     syncdelay=self.us2cycles(self.cfg["relax_delay"]))

    def update(self):
        self.mathi(self.r_rp, self.r_gain, self.r_gain, '+', self.cfg["step"])  # update gain of the pulse

config={"res_ch":6, # --Fixed
        "ro_ch":0, # --Fixed
        "relax_delay":1, # --Fixed
        "res_phase":0, # --Fixed
        "pulse_style": "const", # --Fixed
        "length":100, # [Clock ticks]
        "readout_length":200, # [Clock ticks]
        "pulse_gain":0, # [DAC units]
        "pulse_freq": 100, # [MHz]
        "adc_trig_offset": 100, # [Clock ticks]
        "reps":50,
        # New variables
        "expts": 20,
        "start":0, # [DAC units]
        "step":100 # [DAC units]
       }

prog =SweepProgram(soccfg, config)
expt_pts, avgi, avgq = prog.acquire(soc, load_pulses=True)

# Plot results.
sig = avgi[0][0] + 1j*avgq[0][0]
avgamp0 = np.abs(sig)
plt.figure(1)
plt.plot(expt_pts, avgi[0][0], label="I value; ADC 0")
plt.plot(expt_pts, avgq[0][0], label="Q value; ADC 0")
plt.plot(expt_pts, avgamp0, label="Amplitude; ADC 0")
# plt.plot(expt_pts, avg_di1, label="I value; ADC 1")
# plt.plot(expt_pts, avg_dq1, label="Q value; ADC 1")
# plt.plot(expt_pts, avg_amp1, label="Amplitude; ADC 1")
plt.ylabel("a.u.")
plt.xlabel("Pulse gain (DAC units)")
plt.title("Averages = " + str(config["reps"]))
plt.legend()