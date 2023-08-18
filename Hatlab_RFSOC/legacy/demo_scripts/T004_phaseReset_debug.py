from Hatlab_RFSOC.proxy import getSocProxy
from qick import *
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
soc, soccfg = getSocProxy("myqick216-01")

class LoopbackProgram(AveragerProgram):
    def initialize(self):
        cfg = self.cfg
        res_ch = cfg["res_ch"]

        # set the nyquist zone
        self.declare_gen(ch=cfg["res_ch"], nqz=1)

        # configure the readout lengths and downconversion frequencies (ensuring it is an available DAC frequency)
        for ch in cfg["ro_chs"]:
            if self.soccfg['readouts'][ch]['tproc_ctrl'] is None:
                self.declare_readout(ch=ch, length=cfg["readout_length"],
                                     freq=cfg["readout_freq"], gen_ch=cfg["res_ch"])
            else:
                self.declare_readout(ch=ch, length=cfg["readout_length"])
                freq_ro = self.freq2reg_adc(cfg["readout_freq"], ro_ch=ch, gen_ch=cfg['res_ch']) #todo: why we need the conversion? shouldn't this be already done inside?
                self.set_readout_registers(ch=ch, freq=freq_ro, length=3, #todo: this is tproc cycle?
                                           mode='oneshot', outsel='product', phrst=1)


        # convert frequency to DAC frequency (ensuring it is an available ADC frequency)
        freq = self.freq2reg(cfg["pulse_freq"], gen_ch=res_ch, ro_ch=cfg["ro_chs"][0])
        print("DAC freq!!!!!!!!!!!!!!!!!!!", self.reg2freq(freq, gen_ch=res_ch))
        phase = self.deg2reg(cfg["res_phase"], gen_ch=res_ch)
        gain = cfg["pulse_gain"]

        style = self.cfg["pulse_style"]

        self.set_pulse_registers(ch=res_ch, style=style, freq=freq, phase=phase, gain=gain, length=cfg["length"], phrst=1)

        self.synci(200)  # give processor some time to configure pulses

    def body(self):
        # Configure readout v3.
        for ch in self.cfg["ro_chs"]:
            if self.soccfg['readouts'][ch]['tproc_ctrl'] is not None:
                self.readout(ch=ch, t=0)

        self.sync_all(2)
        #note: this will give different results from run to run, because now the readout and generator are not reset at
        #      the same time

        self.measure(pulse_ch=self.cfg["res_ch"],
                     adcs=self.ro_chs,
                     pins=[0],
                     adc_trig_offset=self.cfg["adc_trig_offset"],
                     wait=True,
                     syncdelay=self.us2cycles(self.cfg["relax_delay"]))




config = {"res_ch": 0,  # --Fixed
          "ro_chs": [0],  # --Fixed
          "reps": 1,  # --Fixed
          "relax_delay": 1.0,  # --us
          "res_phase": 0,  # --degrees
          "pulse_style": "const",  # --Fixed

          "length": 500,  # [Clock ticks]
          # Try varying length from 10-100 clock ticks

          "readout_length": 1000,  # [Clock ticks]
          # Try varying readout_length from 50-1000 clock ticks

          "pulse_gain": 10000,  # [DAC units]
          # Try varying pulse_gain from 500 to 30000 DAC units

          "pulse_freq": 4000,  # [MHz]
          # In this program the signal is up and downconverted digitally so you won't see any frequency
          # components in the I/Q traces below. But since the signal gain depends on frequency,
          # if you lower pulse_freq you will see an increased gain.

          "readout_freq": 4000 ,  # [MHz]

          "adc_trig_offset": 100,  # [Clock ticks]
          # Try varying adc_trig_offset from 100 to 220 clock ticks

          "soft_avgs": 1
          # Try varying soft_avgs from 1 to 200 averages

          }

###################
# Try it yourself !
###################

prog = LoopbackProgram(soccfg, config)


n_rounds = 10
iq_array = np.zeros((n_rounds, 2, config["readout_length"]))
for i in tqdm(range(n_rounds)):
    iq_list = prog.acquire_decimated(soc, load_pulses=True, progress=False, debug=False)
    iq_array[i] = iq_list[0]
#
# Plot results.
plt.figure()
for ii, iq in enumerate(iq_array):
    # plt.plot(iq[0], label="I value")
    plt.plot(iq[1], label="Q value")
    # plt.plot(np.abs(iq[0]+1j*iq[1]), label="mag")
plt.ylabel("a.u.")
plt.xlabel("Clock ticks")
plt.legend()

plt.figure()
plt.plot(np.average(iq_array, axis=0)[0])
plt.plot(np.average(iq_array, axis=0)[1])
# plt.savefig("images/Send_recieve_pulse_const.pdf", dpi=350)