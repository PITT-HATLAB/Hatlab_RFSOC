from qick import *
import matplotlib.pyplot as plt
import numpy as np

from Hatlab_RFSOC.helpers.pulseConfig import set_pulse_registers_IQ


class PulseSpecProgram_ef(PAveragerProgram):
    def initialize(self):
        cfg = self.cfg

        self.declare_gen(ch=cfg["qubit_ch"], nqz=cfg["qubit_nzq"])  # qubit drive
        self.declare_gen(ch=cfg["res_ch_I"], nqz=cfg["res_nzq_I"])  # resonator drive I
        self.declare_gen(ch=cfg["res_ch_Q"], nqz=cfg["res_nzq_Q"])  # resonator drive Q

        self.declare_readout(ch=cfg["ro_ch"], length=cfg["readout_length"],freq=cfg["res_freq"], gen_ch=cfg["res_ch_I"])

        self.q_rp=self.ch_page(self.cfg["qubit_ch"])     # get register page for qubit_ch
        self.r_freq=self.sreg(cfg["qubit_ch"], "freq")   # get frequency register for qubit_ch
        self.r_wait = 3
        self.r_freq_ef = 1  # a register to store qubit ef freq

        self.f_start = soc.freq2reg(cfg["start"])  # get start/step frequencies
        self.f_step = soc.freq2reg(cfg["step"])
        self.safe_regwi(self.q_rp, self.r_freq_ef, self.f_start)


        res_freq = self.freq2reg(cfg["res_freq"], gen_ch=cfg["res_ch_I"], ro_ch=cfg["ro_ch"])  # convert frequency to dac frequency (ensuring it is an available adc frequency)

        self.qubit_freq = soc.freq2reg(cfg["ge_freq"])


        set_pulse_registers_IQ(self, cfg["res_ch_I"], cfg["res_ch_Q"], cfg["skewPhase"],  cfg["IQScale"],
                               style="const", freq=res_freq, phase=cfg["res_phase"], gain=cfg["res_gain"],
                                length=cfg["res_length"])
        n_sigma = cfg.get("n_sigma", 4)
        self.add_gauss(ch=cfg["qubit_ch"], name="qubit", sigma=cfg["sigma"], length=self.us2cycles(cfg["sigma"]*n_sigma))
        self.set_pulse_registers(ch=self.cfg["qubit_ch"], style="arb",waveform="qubit",
                                 phase=self.deg2reg(90, gen_ch=cfg["qubit_ch"]),
                                 freq=self.qubit_freq, gain=cfg["pi_gain"])
        self.sync_all(self.us2cycles(1))  # give processor some time to configure pulses

    def body(self):
        cfg = self.cfg
        self.set_pulse_registers(ch=self.cfg["qubit_ch"], style="arb",waveform="qubit",
                                 phase=self.deg2reg(90, gen_ch=cfg["qubit_ch"]),
                                 freq=self.qubit_freq, gain=cfg["pi_gain"])
        self.pulse(ch=self.cfg["qubit_ch"])  #play ge gaussian pulse
        self.sync_all(soc.us2cycles(0.05))  # align channels and wait 50ns

        self.set_pulse_registers(ch=self.cfg["qubit_ch"], style="const", length=self.cfg["probe_length"],
                                 phase=0, freq=self.f_start, gain=cfg["qubit_gain_ef"])

        self.mathi(self.q_rp, self.r_freq, self.r_freq_ef, '+', 0)  # update frequency list index

        self.pulse(ch=self.cfg["qubit_ch"])  #play probe pulse
        self.sync_all()
        self.sync_all(self.us2cycles(0.05))

        # --- msmt
        self.trigger([cfg["ro_ch"]], adc_trig_offset=cfg["adc_trig_offset"])  # trigger the adc acquisition
        self.pulse(ch=cfg["res_ch_I"], t=0)
        self.pulse(ch=cfg["res_ch_Q"], t=0)
        self.wait_all()
        self.sync_all(self.us2cycles(cfg["relax_delay"])) # wait for qubit to relax

    def update(self):
        self.mathi(self.q_rp, self.r_freq_ef, self.r_freq_ef, '+', self.f_step)  # update frequency list index



if __name__ == "__main__":
    from Hatlab_RFSOC.helpers.dataTransfer import saveData
    from Hatlab_DataProcessing.analyzer import qubit_functions_rot as qfr
    from Hatlab_DataProcessing.analyzer.rotateIQ import RotateData

    from Hatlab_RFSOC.qubitMSMT.exampleConfig import config, rotResult, dataPath, sampleName, PyroServer
    from Hatlab_RFSOC.proxy import getSocProxy
    soc, soccfg = getSocProxy(PyroServer)

    expt_cfg = {"start": 3065,  # MHz
                "step": 0.01,
                "expts": 1000,
                "reps": 200,
                "rounds": 1,
                "probe_length": soc.us2cycles(5),
                "qubit_gain_ef": 200,
                "relax_delay": 300  # [us]
                }
    config.update(expt_cfg)

    print("running...")
    qspec=PulseSpecProgram_ef(soccfg, config)
    expt_pts, avgi, avgq = qspec.acquire(soc, load_pulses=True, progress=True, debug=False)
    print("done...\nplotting...")

    rot=RotateData(expt_pts, avgi[0][0]+1j*avgq[0][0])
    newIQ = rot.run(rotResult["rot_angle"])
    newIQ.plot()

    #Plotting Results
    plt.figure()
    plt.subplot(111,title="Qubit Spectroscopy for ef \n(after rotation)", xlabel="Qubit Frequency ef (MHz)", ylabel="Qubit I Q")
    plt.plot(expt_pts, newIQ.params["q_data"].value,'o-', markersize = 1)
    plt.plot(expt_pts, newIQ.params["i_data"].value,'o-', markersize = 1)
    plt.show()

    plt.figure()
    plt.subplot(111,title="Qubit Spectroscopy for ef\n(before rotation)", xlabel="Qubit Frequency ef (MHz)", ylabel="Qubit I Q")
    plt.plot(expt_pts, avgi[0][0],'o-', markersize = 1)
    plt.plot(expt_pts, avgq[0][0],'o-', markersize = 1)
    plt.show()