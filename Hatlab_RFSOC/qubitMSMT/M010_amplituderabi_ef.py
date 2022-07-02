from proxy.socProxy import soccfg, soc
from qick import *
import matplotlib.pyplot as plt
import numpy as np
import lmfit
from importlib import reload

from Hatlab_RFSOC.helpers.pulseConfig import set_pulse_registers_IQ
from Hatlab_RFSOC.helpers.dataTransfer import saveData
from Hatlab_DataProcessing.analyzer import qubit_functions_rot as qfr

from Hatlab_RFSOC.qubitMSMT.config import config, rotResult, dataPath, sampleName
reload(qfr)

class AmplitudeRabiProgram_ef(PAveragerProgram):
    def initialize(self):
        cfg = self.cfg

        self.declare_gen(ch=cfg["qubit_ch"], nqz=cfg["qubit_nzq"])  # qubit drive
        self.declare_gen(ch=cfg["res_ch_I"], nqz=cfg["res_nzq_I"])  # resonator drive I
        self.declare_gen(ch=cfg["res_ch_Q"], nqz=cfg["res_nzq_Q"])  # resonator drive Q

        self.declare_readout(ch=cfg["ro_ch"], length=cfg["readout_length"], freq=cfg["res_freq"],
                             gen_ch=cfg["res_ch_I"])

        self.gain_start = cfg["start"]  # get start/step gains for ef
        self.gain_step = cfg["step"]

        self.q_rp = self.ch_page(self.cfg["qubit_ch"])  # get register page for qubit_ch
        self.r_gain = self.sreg(cfg["qubit_ch"], "gain")  # get gain register for qubit_ch
        self.r_gain_ef = 1
        self.safe_regwi(self.q_rp, self.r_gain_ef, self.gain_start)

        res_freq = self.freq2reg(cfg["res_freq"], gen_ch=cfg["res_ch_I"], ro_ch=cfg[
            "ro_ch"])  # convert frequency to dac frequency (ensuring it is an available adc frequency)
        self.qubit_freq_ge = soc.freq2reg(cfg["ge_freq"])
        self.qubit_freq_ef = soc.freq2reg(cfg["ef_freq"])

        self.pi_gain = int(cfg["pi_gain"])

        # add qubit and readout pulses to respective channels
        n_sigma = cfg.get("n_sigma", 4)
        set_pulse_registers_IQ(self, cfg["res_ch_I"], cfg["res_ch_Q"], cfg["skewPhase"],  cfg["IQScale"],
                               style="const", freq=res_freq, phase=cfg["res_phase"], gain=cfg["res_gain"],
                                length=cfg["res_length"])


        self.add_gauss(ch=cfg["qubit_ch"], name="qubit", sigma=cfg["sigma"], length=cfg["sigma"]*cfg["n_sigma"])
        self.add_gauss(ch=cfg["qubit_ch"], name="qubit_ef", sigma=cfg["sigma_ef"], length=cfg["sigma_ef"]*cfg["n_sigma"])

        self.sync_all(self.us2cycles(1))  # give processor some time to configure pulses

    def body(self):
        cfg = self.cfg
        if cfg['prepare_e']:
            self.set_pulse_registers(ch=self.cfg["qubit_ch"], style="arb", waveform="qubit",
                                     phase=self.deg2reg(90, gen_ch=cfg["qubit_ch"]),
                                     freq=self.qubit_freq_ge, gain=cfg["pi_gain"])
            self.pulse(ch=self.cfg["qubit_ch"])  # play ge gaussian pulse
            self.sync_all(soc.us2cycles(0.05))  # align channels and wait 50ns


        self.set_pulse_registers(ch=self.cfg["qubit_ch"], style="arb", waveform="qubit_ef",
                                 phase=self.deg2reg(90, gen_ch=cfg["qubit_ch"]),
                                 freq=self.qubit_freq_ef, gain=self.gain_start)
        self.mathi(self.q_rp, self.r_gain, self.r_gain_ef, '+', 0)  # update gain list index
        self.pulse(ch=self.cfg["qubit_ch"])  # play ef gaussian pulse
        self.sync_all(soc.us2cycles(0.05)) # align channels and wait 50ns

        # --- msmt
        self.trigger([cfg["ro_ch"]], adc_trig_offset=cfg["adc_trig_offset"])  # trigger the adc acquisition
        self.pulse(ch=cfg["res_ch_I"], t=0)
        self.pulse(ch=cfg["res_ch_Q"], t=0)
        self.wait_all()
        self.sync_all(self.us2cycles(cfg["relax_delay"]))  # wait for qubit to relax

    def update(self):
        self.mathi(self.q_rp, self.r_gain_ef, self.r_gain_ef, '+',
                   self.cfg["step"])  # update gain of the Gaussian pi pulse



if __name__ == "__main__":
    expt_cfg = {
        "start": -30000,
        "step": 200,
        "expts": 300,
        "reps": 500,
        "relax_delay": 400,
        "prepare_e": True
    }

    config.update(expt_cfg)  # combine configs

    print("running...")
    rabi = AmplitudeRabiProgram_ef(soccfg, config)
    x_pts, avgi_e, avgq_e = rabi.acquire(soc, load_pulses=True, progress=True, debug=False)

    config["prepare_e"] = False
    rabi = AmplitudeRabiProgram_ef(soccfg, config)
    x_pts, avgi_g, avgq_g = rabi.acquire(soc, load_pulses=True, progress=True, debug=False)
    print("done...\n plotting...")


    #Plotting Results
    plt.figure(figsize=(10, 4))
    plt.subplot(121, title=f"Amplitude Rabi ef, prepare e,", xlabel="Gain", ylabel="Qubit Population")
    plt.plot(x_pts, avgi_e[0][0], 'o-', markersize=1)
    plt.plot(x_pts, avgq_e[0][0], 'o-', markersize=1)
    plt.subplot(122, title=f"Amplitude Rabi ef, prepare g, ", xlabel="Gain", ylabel="Qubit Population")
    plt.plot(x_pts, avgi_g[0][0], 'o-', markersize=1)
    plt.plot(x_pts, avgq_g[0][0], 'o-', markersize=1)

    piPulef_e = qfr.PiPulseTuneUp(x_pts, avgi_e[0][0] + 1j * avgq_e[0][0])
    piResultef_e = piPulef_e.run()


    ef_fit_f = piResultef_e.lmfit_result.params["f"].value
    ef_fit_phi = piResultef_e.lmfit_result.params["phi"].value

    piPulef_g = qfr.PiPulseTuneUp(x_pts, avgi_g[0][0] + 1j * avgq_g[0][0])
    piResultef_g = piPulef_g.run(params=dict(f=lmfit.Parameter("f", value=ef_fit_f, vary=False),
                                             phi=lmfit.Parameter("phi", value=ef_fit_phi, vary=False)))

    piResultef_e.plot()
    piResultef_g.plot()

    pi_e = abs(piResultef_e.lmfit_result.params["A"].value)
    pi_g = abs(piResultef_g.lmfit_result.params["A"].value)

    print("e_pct:", pi_g / (pi_g + pi_e))


    # ----- save data to pc --------
    data_temp = {"x_data": x_pts, "i_data": avgi_e[0][0], "q_data": avgq_e[0][0]}
    saveData(data_temp, sampleName + "_ef_rabi", dataPath)

    data_temp = {"x_data": x_pts, "i_data": avgi_g[0][0], "q_data": avgq_g[0][0]}
    saveData(data_temp, sampleName + "_ge_amplitude", dataPath)