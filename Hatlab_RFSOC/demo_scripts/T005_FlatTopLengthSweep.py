from Hatlab_RFSOC.proxy import getSocProxy
from qick import *
import matplotlib.pyplot as plt
import numpy as np
soc, soccfg = getSocProxy("myqick216-01")


class LengthSweepProgram(RAveragerProgram):
    def initialize(self):
        cfg = self.cfg
        res_ch = cfg["res_ch"]

        # set the nyquist zone
        self.declare_gen(ch=res_ch, nqz=cfg["res_nqz"])

        # configure the readout lengths and downconversion frequencies (ensuring it is an available DAC frequency)
        for ch in cfg["ro_chs"]:
            self.declare_readout(ch=ch, length=self.cfg["readout_length"],
                                 freq=self.cfg["pulse_freq"], gen_ch=cfg["res_ch"])



        # define registers for sweep
        self.res_rp=self.ch_page(res_ch)   # get register page for res_ch
        self.res_r_mode=self.sreg(res_ch, "mode")  # length register is packed in the last 16 bits of mode register
        self.res_r_length = 1 # declare a register for keeping track of the flat top length, (used in sync() after the pulse)
        start_length_reg = self.us2cycles(cfg["start"])
        self.safe_regwi(self.res_rp, self.res_r_length, start_length_reg)

        # define raising and lower edge part of the flat top pulse,
        # The first half of the waveform ramps up the pulse, the second half ramps down the pulse
        sigma = self.us2cycles(cfg["sigma"], gen_ch=res_ch)
        self.add_gauss(ch=res_ch, name="drive", sigma=sigma, length=sigma * 6)
        self.ramp_length_reg = self.us2cycles(cfg["sigma"]*6) # used in synci(), so this should be in tproc cycles

        # set pulse registers
        freq = self.freq2reg(cfg["pulse_freq"], gen_ch=res_ch, ro_ch=cfg["ro_chs"][0])
        phase = self.deg2reg(cfg["res_phase"], gen_ch=res_ch)
        gain = cfg["pulse_gain"]
        style = "flat_top"
        self.set_pulse_registers(ch=res_ch, style=style, freq=freq, phase=phase, gain=gain,
                                 waveform="drive", length=start_length_reg)


        self.sync_all(self.us2cycles(1))  # give processor some time to configure pulses



    def body(self):
        cfg = self.cfg

        self.trigger(self.ro_chs, pins=[0], adc_trig_offset=self.cfg["adc_trig_offset"])
        self.pulse(ch=cfg["res_ch"])  # play flat top pulse
        self.synci(self.ramp_length_reg) # total ramp length (raising and lowering)
        self.sync(self.res_rp, self.res_r_length)  # align channels and wait the length of the flat part
        self.reset_ts()  # reset the soft counted adc and dac timestamps (to avoid adding the predefined pump pulse length)
        # These sync and synci are not necessary here, since there is no other pulses after this res pulse, but in a
        # real experiment, there usually is. and we need the res_r_length register to control the waiting time after the
        # flat top length

        self.wait_all()
        self.sync_all(self.us2cycles(self.cfg["relax_delay"]))


    def update(self):
        cfg=self.cfg
        step_reg_tproc = self.us2cycles(cfg["step"]) # note that the waiting happens on tproc
        step_reg_gen = self.us2cycles(cfg["step"], gen_ch=cfg["res_ch"]) # but the pulse length is controlled in generator
        # update length of the pulse and the waiting time after the pulse
        self.mathi(self.res_rp, self.res_r_mode, self.res_r_mode, '+', step_reg_gen)
        self.mathi(self.res_rp, self.res_r_length, self.res_r_length, '+', step_reg_tproc)



if __name__ == "__main__":
    soc, soccfg = getSocProxy("myqick216-01")
    config = {"res_ch": 0,  # --Fixed
              "res_nqz": 1,
              "ro_chs": [0],  # --Fixed
              "reps": 1,  # --Fixed
              "relax_delay": 1.0,  # --us
              "res_phase": 0,  # --degrees

              "readout_length": 1000,  # [Clock ticks]
              "adc_trig_offset": 100,  # [Clock ticks]

              "pulse_gain": 10000,  # [DAC units]
              "pulse_freq": 1000,  # [MHz]

              # "soft_avgs":100
              }

    expt_cfg = {
        "sigma": 0.01,

        "start": 0.01,
        "step": 0.01,
        "expts": 100,
        "reps": 100,
        "rounds": 1,
    }
    config.update(expt_cfg)  # combine configs


    prog = LengthSweepProgram(soccfg, config)
    expts, avg_di, avg_dq = prog.acquire(soc, load_pulses=True, progress=True, debug=False)

    # Plot results
    plt.figure(1)
    for ii, (i,q) in enumerate(zip(avg_di, avg_dq)):
        plt.plot(i[0], label="I value, ADC %d" % (config['ro_chs'][ii]))
        plt.plot(q[0], label="Q value, ADC %d" % (config['ro_chs'][ii]))
        plt.plot(np.abs(i[0] + 1j * q[0]), label="mag, ADC %d" % (config['ro_chs'][ii]))
    plt.ylabel("a.u.")
    plt.xlabel("Clock ticks")
    plt.legend()