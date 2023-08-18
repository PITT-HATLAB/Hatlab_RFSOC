from Hatlab_RFSOC.proxy import getSocProxy
from qick import *
from qick.averager_program import QickSweep, merge_sweeps
import matplotlib.pyplot as plt
import numpy as np
soc, soccfg = getSocProxy("myqick216-01")

config = {"res_ch": 4,  # --Fixed
          "ro_chs": [0],  # --Fixed
          "reps": 100,  # --Fixed
          "relax_delay": 1.0,  # --us

          "length": 50,  # [Clock ticks]
          # Try varying length from 10-100 clock ticks

          "readout_length": 100,  # [Clock ticks]
          # Try varying readout_length from 50-1000 clock ticks

          "pulse_freq": 1500,  # [MHz]
          # In this program the signal is up and downconverted digitally so you won't see any frequency
          # components in the I/Q traces below. But since the signal gain depends on frequency,
          # if you lower pulse_freq you will see an increased gain.

          "adc_trig_offset": 150,  # [Clock ticks]
          # Try varying adc_trig_offset from 100 to 220 clock ticks

          "soft_avgs": 1,

          "rounds":1

          }

expt_cfg = {
    "g_start": 100,  # [DAC units]
    "g_stop": 10100,  # [DAC units]
    "g_expts": 101,

    "phi_start": 0,  # --degrees
    "phi_stop": 360,  # --degrees
    "phi_expts": 51,
}

config.update(**expt_cfg)

class NDSweepProgram(NDAveragerProgram):
    def initialize(self):
        cfg = self.cfg
        res_ch = cfg["res_ch"]

        # set the nyquist zone
        self.declare_gen(ch=cfg["res_ch"], nqz=1, ro_ch=cfg["ro_chs"][0])

        # configure the readout lengths and downconversion frequencies (ensuring it is an available DAC frequency)
        for ch in cfg["ro_chs"]:
            self.declare_readout(ch=ch, length=self.cfg["readout_length"],
                                 freq=self.cfg["pulse_freq"], gen_ch=cfg["res_ch"])

        # convert frequency to DAC frequency (ensuring it is an available ADC frequency)
        freq = self.freq2reg(cfg["pulse_freq"], gen_ch=res_ch, ro_ch=cfg["ro_chs"][0])
        phase = self.deg2reg(cfg["phi_start"], gen_ch=res_ch)
        gain = cfg["g_start"]

        self.set_pulse_registers(ch=res_ch, style="const", freq=freq, phase=phase, gain=gain, length=cfg["length"])

        # add pulse gain and phase sweep, first added will be first swept
        self.res_r_gain = self.get_gen_reg(cfg["res_ch"], "gain")
        self.res_r_gain_update = self.new_gen_reg(cfg["res_ch"], init_val=cfg["g_start"], name="gain_update")
        self.res_r_phase = self.get_gen_reg(cfg["res_ch"], "phase")
        self.res_r_phase_update = self.new_gen_reg(cfg["res_ch"], init_val=cfg["phi_start"], name="phase_update", reg_type="phase")

        self.add_sweep(QickSweep(self, self.res_r_gain_update, cfg["g_start"], cfg["g_stop"], cfg["g_expts"]))
        self.add_sweep(QickSweep(self, self.res_r_phase, cfg["phi_start"], cfg["phi_stop"], cfg["phi_expts"]))

        # g_sweep = QickSweep(self, self.res_r_gain_update, cfg["g_start"], cfg["g_stop"], cfg["g_expts"])
        # phi_sweep = QickSweep(self, self.res_r_phase, cfg["phi_start"], cfg["phi_stop"], cfg["phi_expts"])
        # self.add_sweep(merge_sweeps([g_sweep, phi_sweep]))


        self.synci(200)  # give processor some time to configure pulses

    def body(self):
        # fire the pulse
        # trigger all declared ADCs
        # pulse PMOD0_0 for a scope trigger
        # pause the tProc until readout is done
        # increment the time counter to give some time before the next measurement
        # (the syncdelay also lets the tProc get back ahead of the clock)


        self.res_r_gain.set_to(self.res_r_gain_update, "+", 0)
        # same as :
        # self.mathi(self.res_r_gain.page, self.res_r_gain.addr, self.res_r_gain_update.addr, "+", 0)

        self.measure(pulse_ch=self.cfg["res_ch"],
                     adcs=self.ro_chs,
                     pins=[0],
                     adc_trig_offset=self.cfg["adc_trig_offset"],
                     wait=True,
                     syncdelay=self.us2cycles(self.cfg["relax_delay"]))


###################
# Try it yourself !
###################

prog = NDSweepProgram(soccfg, config)
expt_pts, avg_di, avg_dq = prog.acquire(soc, load_pulses=True, progress=True, debug=False)

avg_abs, avg_angle = np.abs(avg_di + 1j * avg_dq), np.angle(avg_di + 1j * avg_dq)

# plot data
fig, axes = plt.subplots(1, 2, figsize=(12,5))
for i, d in enumerate([avg_di, avg_dq]):
    pcm = axes[i].pcolormesh(prog.get_expt_pts()[1], prog.get_expt_pts()[0], d[0,0].T, shading="Auto", cmap="RdBu")
    axes[i].set_xlabel("Phase(deg)")
    axes[i].set_ylabel("Gain")
    axes[i].set_title("I data" if i ==0  else "Q data")
    plt.colorbar(pcm, ax=axes[i])

fig, axes = plt.subplots(1, 2, figsize=(12,5))
for i, d in enumerate([avg_abs, avg_angle]):
    if i==0:
        pcm = axes[i].pcolormesh(prog.get_expt_pts()[1], prog.get_expt_pts()[0], d[0,0].T, shading="Auto", cmap="hot")
    else:
        pcm = axes[i].pcolormesh(prog.get_expt_pts()[1], prog.get_expt_pts()[0], np.unwrap(d[0, 0].T), shading="Auto",cmap="twilight")
    axes[i].set_xlabel("Phase(deg)")
    axes[i].set_ylabel("Gain")
    axes[i].set_title("IQ Amp" if i ==0  else "IQ phase (rad)")
    plt.colorbar(pcm, ax=axes[i])


