from qick import *
import matplotlib.pyplot as plt

from legacy.pulseConfig import set_pulse_registers_IQ, declareMuxedGenAndReadout

class AmplitudeRabiProgram(PAveragerProgram):
    def initialize(self):
        cfg = self.cfg

        self.declare_gen(ch=cfg["qubit_ch"], nqz=cfg["qubit_nzq"])  # qubit drive
        self.declare_gen(ch=cfg["res_ch_I"], nqz=cfg["res_nzq_I"])  # resonator drive I
        self.declare_gen(ch=cfg["res_ch_Q"], nqz=cfg["res_nzq_Q"])  # resonator drive Q

        self.declare_readout(ch=cfg["ro_ch"], length=cfg["readout_length"],freq=cfg["res_freq"], gen_ch=cfg["res_ch_I"])

        self.q_rp=self.ch_page(self.cfg["qubit_ch"])     # get register page for qubit_ch
        self.r_gain=self.sreg(cfg["qubit_ch"], "gain")   # get gain register for qubit_ch
        self.r_gain_update = 1 # register for keeping the update value of gain
        self.safe_regwi(self.q_rp, self.r_gain_update, cfg["start"])


        res_freq = self.freq2reg(cfg["res_freq"], gen_ch=cfg["res_ch_I"], ro_ch=cfg["ro_ch"])  # convert frequency to dac frequency (ensuring it is an available adc frequency)
        qubit_freq = self.freq2reg(cfg["ge_freq"])
        self.qubit_freq = qubit_freq

        # add qubit and readout pulses to respective channels
        n_sigma = cfg.get("n_sigma", 4)
        self.add_gauss(ch=cfg["qubit_ch"], name="qubit", sigma=self.us2cycles(cfg["sigma"]), length=self.us2cycles(cfg["sigma"]*n_sigma))

        set_pulse_registers_IQ(self, cfg["res_ch_I"], cfg["res_ch_Q"], cfg["skewPhase"],  cfg["IQScale"],
                               style="const", freq=res_freq, phase=cfg["res_phase"], gain=cfg["res_gain"],
                                length=cfg["res_length"])

        self.set_pulse_registers(ch=self.cfg["qubit_ch"], style="arb",waveform="qubit",
                                 phase=self.deg2reg(90, gen_ch=cfg["qubit_ch"]),
                                 freq=qubit_freq, gain=cfg["start"])

        self.sync_all(self.us2cycles(1))  # give processor some time to configure pulses

    def body(self):
        cfg = self.cfg
        prepareWithMSMT = cfg.get("prepareWithMSMT", False)
        #
        if prepareWithMSMT:
            self.set_pulse_registers(ch=self.cfg["qubit_ch"], style="arb", waveform="qubit",
                                     phase=self.deg2reg(90, gen_ch=cfg["qubit_ch"]),
                                     freq=self.qubit_freq, gain=cfg["pi2_gain"])
            self.pulse(ch=self.cfg["qubit_ch"])  # play gaussian pulse
            self.sync_all(self.us2cycles(0.05))  # align channels and wait 50ns
            self.measure(pulse_ch=[cfg["res_ch_I"], cfg["res_ch_Q"]],
                         adcs=[self.cfg["ro_ch"]],
                         adc_trig_offset=self.cfg["adc_trig_offset"],
                         t=0,
                         wait=True,
                         syncdelay=self.us2cycles(0.5))
        #
        #
        # drive and measure
        self.mathi(self.q_rp, self.r_gain, self.r_gain_update, '+', 0)  # set the updated gain value
        self.pulse(ch=self.cfg["qubit_ch"])  # play gaussian pulse
        self.sync_all(self.us2cycles(0.05))  # align channels and wait 50ns
        self.measure(pulse_ch=[cfg["res_ch_I"], cfg["res_ch_Q"]],
                     adcs=[self.cfg["ro_ch"]],
                     adc_trig_offset=self.cfg["adc_trig_offset"],
                     t=0,
                     wait=True,
                     syncdelay=self.us2cycles(self.cfg["relax_delay"]))


    def update(self):
        self.mathi(self.q_rp, self.r_gain_update, self.r_gain_update, '+', self.cfg["step"])  # update gain of the pulse
        # self.mathi(self.q_rp, self.r_gain, self.r_gain, '+', self.cfg["step"])  # update gain of the pulse



class MuxedAmplitudeRabiProgram(PAveragerProgram):
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

        self.q_rp=self.ch_page(self.cfg["qubit_ch"])     # get register page for qubit_ch
        self.r_gain=self.sreg(cfg["qubit_ch"], "gain")   # get gain register for qubit_ch
        self.r_gain_update = 1 # register for keeping the update value of gain
        self.safe_regwi(self.q_rp, self.r_gain_update, cfg["start"])

        self.qubit_freq = self.freq2reg(cfg["ge_freq"], gen_ch=cfg["qubit_ch"])

        # add qubit pulses to respective channels
        n_sigma = cfg.get("n_sigma", 4)
        self.add_gauss(ch=cfg["qubit_ch"], name="qubit", sigma=self.us2cycles(cfg["sigma"]), length=self.us2cycles(cfg["sigma"]*n_sigma))

        self.set_pulse_registers(ch=self.cfg["qubit_ch"], style="arb",waveform="qubit",
                                 phase=self.deg2reg(90, gen_ch=cfg["qubit_ch"]),
                                 freq=self.qubit_freq, gain=cfg["start"])

        self.sync_all(self.us2cycles(1))  # give processor some time to configure pulses

    def body(self):
        cfg = self.cfg
        prepareWithMSMT = cfg.get("prepareWithMSMT", False)
        #
        if prepareWithMSMT:
            self.set_pulse_registers(ch=self.cfg["qubit_ch"], style="arb", waveform="qubit",
                                     phase=self.deg2reg(90, gen_ch=cfg["qubit_ch"]),
                                     freq=self.qubit_freq, gain=cfg["pi2_gain"])
            self.pulse(ch=self.cfg["qubit_ch"])  # play gaussian pulse
            self.sync_all(self.us2cycles(0.05))  # align channels and wait 50ns
            self.measure(pulse_ch=self.cfg["res_ch"],
                         adcs=self.ro_chs,
                         pins=[0],
                         adc_trig_offset=self.cfg["adc_trig_offset"],
                         wait=True,
                         syncdelay=self.us2cycles(self.cfg["relax_delay"]))
        #
        #
        # drive and measure
        self.mathi(self.q_rp, self.r_gain, self.r_gain_update, '+', 0)  # set the updated gain value
        self.pulse(ch=self.cfg["qubit_ch"])  # play gaussian pulse
        self.sync_all(self.us2cycles(0.05))  # align channels and wait 50ns
        # --- msmt
        self.measure(pulse_ch=self.cfg["res_ch"],
                     adcs=self.ro_chs,
                     pins=[0],
                     adc_trig_offset=self.cfg["adc_trig_offset"],
                     wait=True,
                     syncdelay=self.us2cycles(self.cfg["relax_delay"]))


    def update(self):
        self.mathi(self.q_rp, self.r_gain_update, self.r_gain_update, '+', self.cfg["step"])  # update gain of the pulse
        # self.mathi(self.q_rp, self.r_gain, self.r_gain, '+', self.cfg["step"])  # update gain of the pulse





if __name__ == "__main__":
    from Hatlab_DataProcessing.analyzer import qubit_functions_rot as qfr
    from legacy.qubitMSMT import config, PyroServer
    from Hatlab_RFSOC.proxy import getSocProxy

    soc, soccfg = getSocProxy(PyroServer)

    expt_cfg={
        "start":-30000,
        "step":200,
        "expts":300,
        "reps": 300,
        "relax_delay":500
           }
    config.update(expt_cfg) #combine configs

    print("running...")

    rabi=AmplitudeRabiProgram(soccfg, config)

    if config.get("prepareWithMSMT", False) :
        x_pts, avgi, avgq = rabi.acquire(soc, load_pulses=True, readouts_per_experiment=2, save_experiments=[0, 1],
                                         progress=True, debug=False)
        Idata = rabi.di_buf_p[0]
        Qdata = rabi.dq_buf_p[0]

        from Hatlab_DataProcessing.post_selection.postSelectionProcess import simpleSelection_1Qge
        g_pct, I_vld, Q_vld, selData = simpleSelection_1Qge(Idata, Qdata, plot=True, xData={"amp": x_pts},
                                                            selCircleSize=1)

        plt.figure()
        plt.figure("g_pct")
        plt.plot(x_pts, g_pct)

        piPul = qf.PiPulseTuneUp(x_pts, g_pct)
        piResult = piPul.run()
        piResult.plot()
    else:
        x_pts, avgi, avgq  = rabi.acquire(soc,load_pulses=True,progress=True, debug=False)

        #Plotting Results
        plt.figure()
        plt.subplot(111, title= f"Amplitude Rabi, $\sigma={soc.cycles2us(config['sigma'])*1000}$ ns", xlabel="Gain", ylabel="Qubit Population" )
        plt.plot(x_pts,avgi[0][0],'o-', markersize = 1)
        plt.plot(x_pts,avgq[0][0],'o-', markersize = 1)

        piPul = qfr.PiPulseTuneUp(x_pts, avgi[0][0]+1j*avgq[0][0])
        piResult = piPul.run()
        piResult.plot()
        piResult.print_ge_rotation()


