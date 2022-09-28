from qick import *
import matplotlib.pyplot as plt

from legacy.pulseConfig import set_pulse_registers_IQ, declareMuxedGenAndReadout


class PulseSpecProgram(PAveragerProgram):
    def initialize(self):
        cfg = self.cfg
        self.f_start = self.freq2reg(cfg["start"], gen_ch=cfg["qubit_ch"])  # get start/step frequencies
        self.f_step = self.freq2reg(cfg["step"], gen_ch=cfg["qubit_ch"])


        self.declare_gen(ch=cfg["qubit_ch"], nqz=cfg["qubit_nzq"])  # qubit drive
        self.declare_gen(ch=cfg["res_ch_I"], nqz=cfg["res_nzq_I"])  # resonator drive I
        self.declare_gen(ch=cfg["res_ch_Q"], nqz=cfg["res_nzq_Q"])  # resonator drive Q

        self.declare_readout(ch=cfg["ro_ch"], length=cfg["readout_length"],freq=cfg["res_freq"], gen_ch=cfg["res_ch_I"])

        self.q_rp=self.ch_page(self.cfg["qubit_ch"])     # get register page for qubit_ch
        self.r_freq=self.sreg(cfg["qubit_ch"], "freq")   # get frequency register for qubit_ch

        res_freq = self.freq2reg(cfg["res_freq"], gen_ch=cfg["res_ch_I"], ro_ch=cfg["ro_ch"])  # convert frequency to dac frequency (ensuring it is an available adc frequency)

        set_pulse_registers_IQ(self, cfg["res_ch_I"], cfg["res_ch_Q"], cfg["skewPhase"],  cfg["IQScale"],
                               style="const", freq=res_freq, phase=cfg["res_phase"], gain=cfg["res_gain"],
                                length=cfg["res_length"])

        self.set_pulse_registers(ch=self.cfg["qubit_ch"], style="const", length=soc.us2cycles(self.cfg["probe_length"]),
                                 phase=0, freq=self.f_start, gain=cfg["qubit_gain"])

        self.sync_all(self.us2cycles(1))  # give processor some time to configure pulses

    def body(self):
        cfg = self.cfg
        self.pulse(ch=self.cfg["qubit_ch"])  # play probe pulse
        self.sync_all(self.us2cycles(0.05))  # align channels and wait 50ns

        # --- msmt
        self.trigger([cfg["ro_ch"]], adc_trig_offset=cfg["adc_trig_offset"])  # trigger the adc acquisition
        self.pulse(ch=cfg["res_ch_I"], t=0)
        self.pulse(ch=cfg["res_ch_Q"], t=0)
        self.wait_all()
        self.sync_all(self.us2cycles(cfg["relax_delay"])) # wait for qubit to relax

    def update(self):
        self.mathi(self.q_rp, self.r_freq, self.r_freq, '+', self.f_step)  # update frequency list index


class MuxedPulseSpecProgram(PAveragerProgram):
    def initialize(self):
        cfg = self.cfg

        # declare muxed generator and readout channels
        declareMuxedGenAndReadout(self, cfg["res_ch"], cfg["res_nqz"], cfg["res_mixer_freq"],
                                  cfg["res_freqs"], cfg["res_gains"], cfg["ro_chs"], cfg["readout_length"])

        # set readout pulse registers
        self.set_pulse_registers(ch=cfg["res_ch"], style="const", length=cfg["res_length"], mask=[0, 1, 2, 3])

        # set / config qubit DAC channel
        qubit_mixer_freq = cfg.get("qubit_mixer_freq", 0)
        self.declare_gen(ch=cfg["qubit_ch"], mixer_freq=qubit_mixer_freq, nqz=cfg["qubit_nqz"])  # qubit drive

        self.f_start = self.freq2reg(cfg["start"], gen_ch=cfg["qubit_ch"])  # get start/step frequencies
        self.f_step = self.freq2reg(cfg["step"], gen_ch=cfg["qubit_ch"])

        self.q_rp=self.ch_page(self.cfg["qubit_ch"])     # get register page for qubit_ch
        self.r_freq=self.sreg(cfg["qubit_ch"], "freq")   # get frequency register for qubit_ch

        self.set_pulse_registers(ch=self.cfg["qubit_ch"], style="const", length=soc.us2cycles(self.cfg["probe_length"]),
                                 phase=0, freq=self.f_start, gain=cfg["qubit_gain"])

        self.sync_all(self.us2cycles(1))  # give processor some time to configure pulses

    def body(self):
        cfg = self.cfg
        self.pulse(ch=self.cfg["qubit_ch"])  # play probe pulse
        self.sync_all(self.us2cycles(0.05))  # align channels and wait 50ns

        # --- msmt
        self.measure(pulse_ch=self.cfg["res_ch"],
                     adcs=self.ro_chs,
                     pins=[0],
                     adc_trig_offset=self.cfg["adc_trig_offset"],
                     wait=True,
                     syncdelay=self.us2cycles(self.cfg["relax_delay"]))

    def update(self):
        self.mathi(self.q_rp, self.r_freq, self.r_freq, '+', self.f_step)  # update frequency list index



if __name__ == "__main__":
    from legacy.qubitMSMT import config, PyroServer
    from Hatlab_RFSOC.proxy import getSocProxy
    soc, soccfg = getSocProxy(PyroServer)

    expt_cfg={"start":3794.5, # MHz
              "step":0.001,
              "expts":1000,
              "reps": 200,
              "rounds":1,
              "probe_length":5,
              "qubit_gain":100,
              "relax_delay": 200 #[us]
             }


    config.update(expt_cfg)

    print("running...")
    qspec=PulseSpecProgram(soccfg, config)
    expt_pts, avgi, avgq = qspec.acquire(soc, load_pulses=True,progress=True, debug=False)
    print("done...\n plotting...")

    #Plotting Results
    plt.figure()
    plt.subplot(111,title="Qubit Spectroscopy", xlabel="Qubit Frequency (GHz)", ylabel="Qubit I")
    plt.plot(expt_pts, avgi[0][0],'o-', markersize = 1)
    plt.plot(expt_pts, avgq[0][0],'o-', markersize = 1)
    plt.show()