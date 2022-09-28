from qick import *
import matplotlib.pyplot as plt

from Hatlab_RFSOC.helpers.pulseConfig import set_pulse_registers_IQ, declareMuxedGenAndReadout


class T2EProgram(PAveragerProgram):
    def initialize(self):
        cfg = self.cfg

        self.q_rp = self.ch_page(self.cfg["qubit_ch"])  # get register page for qubit_ch
        self.r_wait = 3
        self.regwi(self.q_rp, self.r_wait, self.us2cycles(cfg["start"]))

        self.declare_gen(ch=cfg["qubit_ch"], nqz=cfg["qubit_nzq"])  # qubit drive
        self.declare_gen(ch=cfg["res_ch_I"], nqz=cfg["res_nzq_I"])  # resonator drive I
        self.declare_gen(ch=cfg["res_ch_Q"], nqz=cfg["res_nzq_Q"])  # resonator drive Q

        self.declare_readout(ch=cfg["ro_ch"], length=cfg["readout_length"], freq=cfg["res_freq"],
                             gen_ch=cfg["res_ch_I"])


        res_freq = self.freq2reg(cfg["res_freq"], gen_ch=cfg["res_ch_I"], ro_ch=cfg["ro_ch"])  # convert frequency to dac frequency (ensuring it is an available adc frequency)
        qubit_freq = self.freq2reg(cfg["ge_freq"], gen_ch=cfg["qubit_ch"])
        self.qubit_freq = qubit_freq

        # add qubit and readout pulses to respective channels
        n_sigma = cfg.get("n_sigma", 4)
        self.add_gauss(ch=cfg["qubit_ch"], name="qubit", sigma=self.us2cycles(cfg["sigma"]), length=self.us2cycles(cfg["sigma"]*n_sigma))
        self.set_pulse_registers(ch=self.cfg["qubit_ch"], style="arb",waveform="qubit",
                                 phase=self.deg2reg(90, gen_ch=cfg["qubit_ch"]),
                                 freq=qubit_freq, gain=cfg["pi2_gain"])
        set_pulse_registers_IQ(self, cfg["res_ch_I"], cfg["res_ch_Q"], cfg["skewPhase"],  cfg["IQScale"],
                               style="const", freq=res_freq, phase=cfg["res_phase"], gain=cfg["res_gain"],
                                length=cfg["res_length"])
        self.sync_all(self.us2cycles(0.2))

    def body(self):
        cfg = self.cfg

        self.pulse(ch=self.cfg["qubit_ch"])  #play probe pulse
        self.sync_all()
        self.sync(self.q_rp,self.r_wait)
        self.set_pulse_registers(ch=self.cfg["qubit_ch"], style="arb",waveform="qubit",
                                 phase=self.deg2reg(90, gen_ch=cfg["qubit_ch"]),
                                 freq=self.qubit_freq, gain=cfg["pi_gain"])
        self.pulse(ch=self.cfg["qubit_ch"])  #play probe pulse
        self.sync_all()
        self.sync(self.q_rp,self.r_wait)
        self.set_pulse_registers(ch=self.cfg["qubit_ch"], style="arb",waveform="qubit",
                                 phase=self.deg2reg(90, gen_ch=cfg["qubit_ch"]),
                                 freq=self.qubit_freq, gain=cfg["pi2_gain"])
        self.pulse(ch=self.cfg["qubit_ch"])  #play probe pulse
        self.sync_all(self.us2cycles(0.05))

        # --- msmt
        self.trigger([cfg["ro_ch"]], adc_trig_offset=cfg["adc_trig_offset"])  # trigger the adc acquisition
        self.pulse(ch=cfg["res_ch_I"], t=0)
        self.pulse(ch=cfg["res_ch_Q"], t=0)
        self.wait_all()
        self.sync_all(self.us2cycles(cfg["relax_delay"])) # wait for qubit to relax

    def update(self):
        self.mathi(self.q_rp, self.r_wait, self.r_wait, '+',
                   self.us2cycles(self.cfg["step"] / 2))  # update the time between two π/2 pulses




class MuxedT2EProgram(PAveragerProgram):
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

        self.q_rp = self.ch_page(self.cfg["qubit_ch"])  # get register page for qubit_ch
        self.r_wait = 3
        self.regwi(self.q_rp, self.r_wait, self.us2cycles(cfg["start"]))

        self.qubit_freq = self.freq2reg(cfg["ge_freq"], gen_ch=cfg["qubit_ch"])

        # add qubit and readout pulses to respective channels
        n_sigma = cfg.get("n_sigma", 4)
        self.add_gauss(ch=cfg["qubit_ch"], name="qubit", sigma=self.us2cycles(cfg["sigma"]), length=self.us2cycles(cfg["sigma"]*n_sigma))
        self.set_pulse_registers(ch=self.cfg["qubit_ch"], style="arb",waveform="qubit",
                                 phase=self.deg2reg(90, gen_ch=cfg["qubit_ch"]),
                                 freq=self.qubit_freq, gain=cfg["pi2_gain"])

        self.sync_all(self.us2cycles(1))

    def body(self):
        cfg = self.cfg
        # pi/2 pulse
        self.pulse(ch=self.cfg["qubit_ch"])  #play probe pulse
        self.sync_all()
        self.sync(self.q_rp,self.r_wait)
        # pi pulse
        self.set_pulse_registers(ch=self.cfg["qubit_ch"], style="arb",waveform="qubit",
                                 phase=self.deg2reg(90, gen_ch=cfg["qubit_ch"]),
                                 freq=self.qubit_freq, gain=cfg["pi_gain"])
        self.pulse(ch=self.cfg["qubit_ch"])  #play probe pulse
        self.sync_all()
        self.sync(self.q_rp,self.r_wait)
        # pi/2 pulse
        self.set_pulse_registers(ch=self.cfg["qubit_ch"], style="arb",waveform="qubit",
                                 phase=self.deg2reg(90, gen_ch=cfg["qubit_ch"]),
                                 freq=self.qubit_freq, gain=cfg["pi2_gain"])
        self.pulse(ch=self.cfg["qubit_ch"])  #play probe pulse
        self.sync_all(self.us2cycles(1))

        # --- msmt
        self.measure(pulse_ch=self.cfg["res_ch"],
                     adcs=self.ro_chs,
                     pins=[0],
                     adc_trig_offset=self.cfg["adc_trig_offset"],
                     wait=True,
                     syncdelay=self.us2cycles(self.cfg["relax_delay"]))

    def update(self):
        self.mathi(self.q_rp, self.r_wait, self.r_wait, '+',
                   self.us2cycles(self.cfg["step"] / 2))  # update the time between two π/2 pulses



if __name__ == "__main__":
    from data.data_transfer import saveData
    from Hatlab_DataProcessing.analyzer import qubit_functions_rot as qfr
    from legacy.qubitMSMT import config, rotResult, dataPath, sampleName, PyroServer
    from Hatlab_RFSOC.proxy import getSocProxy
    soc, soccfg = getSocProxy(PyroServer)

    expt_cfg = {
        "start": 0,  # [us]
        "step": 0.8,  # [us]
        "expts": 1000,
        "reps": 200,
        "rounds": 1,
        "relax_delay": 600  # [us]
    }

    config.update(expt_cfg)  # combine configs

    print("running...")
    t2ep=T2EProgram(soccfg, config)
    x_pts, avgi, avgq= t2ep.acquire(soc, load_pulses=True,progress=True, debug=False)
    print("done...\n plotting...")


    #Plotting Results
    plt.figure()
    plt.subplot(111, title="T2 echo Experiment", xlabel="Delay time ($\mu$s)", ylabel="Qubit Population")
    plt.plot(x_pts,avgi[0][0],'o-')
    plt.plot(x_pts,avgq[0][0],'o-')

    t2EDecay = qfr.T1Decay(x_pts, avgi[0][0]+1j*avgq[0][0])
    t2EResult = t2EDecay.run(rotResult)
    t2EResult.plot()



    # ----- save data to pc --------
    data_temp = {"x_data":x_pts, "i_data": avgi[0][0], "q_data": avgq[0][0]}
    saveData(data_temp, sampleName+"_t2e", dataPath)