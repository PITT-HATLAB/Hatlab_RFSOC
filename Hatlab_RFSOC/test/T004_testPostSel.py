from proxy.socProxy import soccfg, soc
from qick import *
from qick.averager_program import PAveragerProgram
import matplotlib.pyplot as plt
import numpy as np




class SweepProgram(PAveragerProgram):
    def initialize(self):
        cfg = self.cfg
        self.prepare_by_msmt = cfg.get("prepare_by_msmt", False)
        # set the nyquist zone
        self.declare_gen(ch=cfg["res_ch"], nqz=1)

        self.r_rp = self.ch_page(self.cfg["res_ch"])  # get register page for res_ch
        self.r_gain = self.sreg(cfg["res_ch"], "gain")  # Get gain register for res_ch
        self.r_gain_update = 1
        self.safe_regwi(self.r_rp, self.r_gain_update, cfg["start"])


        # configure the readout lengths and downconversion frequencies
        self.declare_readout(ch=cfg["ro_ch"], length=self.cfg["readout_length"],
                             freq=self.cfg["pulse_freq"], gen_ch=cfg["res_ch"])

        # convert frequency to dac frequency (ensuring it is an available adc frequency)
        self.freq = self.freq2reg(cfg["pulse_freq"], gen_ch=cfg["res_ch"], ro_ch=cfg["ro_ch"])

        self.set_pulse_registers(ch=cfg["res_ch"], style="const", freq=self.freq, phase=0, gain=cfg["start"],
                                 length=cfg["length"])
        self.synci(200)  # give processor some time to configure pulses

    def body(self):
        cfg = self.cfg

        #
        #
        # self.measure(pulse_ch=self.cfg["res_ch"],
        #              adcs=[self.cfg["ro_ch"]],
        #              adc_trig_offset=self.cfg["adc_trig_offset"],
        #              wait=True,
        #              syncdelay=self.us2cycles(self.cfg["relax_delay"]))




        self.trigger([cfg["ro_ch"]], adc_trig_offset=cfg["adc_trig_offset"])  # trigger the adc acquisition
        self.pulse(ch=self.cfg["res_ch"])  #play probe pulse
        self.wait_all()
        self.sync_all(self.us2cycles(1))

        self.set_pulse_registers(ch=self.cfg["res_ch"], style="const", freq=self.freq, phase=0, length=cfg["length"], gain=0)
        self.trigger([cfg["ro_ch"]], adc_trig_offset=cfg["adc_trig_offset"])  # trigger the adc acquisition
        self.pulse(ch=self.cfg["res_ch"])  #play probe pulse
        self.wait_all()

        self.sync_all(self.us2cycles(cfg["relax_delay"])) # wait for qubit to relax



    def update(self):
        self.mathi(self.r_rp, self.r_gain, self.r_gain_update, '+', self.cfg["step"])  # update gain of the pulse
        self.mathi(self.r_rp, self.r_gain_update, self.r_gain_update, '+', self.cfg["step"])  # update gain of the pulse

config={"res_ch":6, # --Fixed
        "ro_ch":0, # --Fixed
        "relax_delay":10, # --Fixed
        "res_phase":0, # --Fixed
        "pulse_style": "const", # --Fixed
        "length":100, # [Clock ticks]
        "readout_length":200, # [Clock ticks]
        "pulse_gain":0, # [DAC units]
        "pulse_freq": 100, # [MHz]
        "adc_trig_offset": 100, # [Clock ticks]
        "reps":500,
        # New variables
        "expts": 20,
        "start":100, # [DAC units]
        "step":100, # [DAC units]
        "rounds": 1,

        "prepare_by_msmt": False
       }

prog =SweepProgram(soccfg, config)
expt_pts, avgi, avgq = prog.acquire(soc, readouts_per_experiment=2, save_experiments=[0,1], load_pulses=True, progress=True)
# expt_pts, avgi, avgq = prog.acquire(soc, load_pulses=True, progress=True)

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

plt.figure()
plt.plot(prog.di_buf_p[0,:,::2])
plt.plot(prog.di_buf_p[0,:,1::2], "-*")