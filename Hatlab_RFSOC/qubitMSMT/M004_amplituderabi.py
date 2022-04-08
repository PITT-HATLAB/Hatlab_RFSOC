from proxy.socProxy import soccfg, soc
from qick import *
import matplotlib.pyplot as plt
import numpy as np

from helpers.pulseConfig import set_pulse_registers_IQ
from helpers.dataTransfer import saveData
from Hatlab_DataProcessing.analyzer import qubit_functions_rot as qfr

from qubitMSMT.config import config, rotResult, dataPath, sampleName


class AmplitudeRabiProgram(PAveragerProgram):
    def initialize(self):
        cfg = self.cfg

        self.declare_gen(ch=cfg["qubit_ch"], nqz=cfg["qubit_nzq"])  # qubit drive
        self.declare_gen(ch=cfg["res_ch_I"], nqz=cfg["res_nzq_I"])  # resonator drive I
        self.declare_gen(ch=cfg["res_ch_Q"], nqz=cfg["res_nzq_Q"])  # resonator drive Q

        self.declare_readout(ch=cfg["ro_ch"], length=cfg["readout_length"],freq=cfg["res_freq"], gen_ch=cfg["res_ch_I"])

        self.q_rp=self.ch_page(self.cfg["qubit_ch"])     # get register page for qubit_ch
        self.r_gain=self.sreg(cfg["qubit_ch"], "gain")   # get gain register for qubit_ch

        res_freq = self.freq2reg(cfg["res_freq"], gen_ch=cfg["res_ch_I"], ro_ch=cfg["ro_ch"])  # convert frequency to dac frequency (ensuring it is an available adc frequency)
        qubit_freq = soc.freq2reg(cfg["ge_freq"])

        # add qubit and readout pulses to respective channels
        self.add_gauss(ch=cfg["qubit_ch"], name="qubit", sigma=cfg["sigma"], length=cfg["sigma"]*4)

        set_pulse_registers_IQ(self, cfg["res_ch_I"], cfg["res_ch_Q"], cfg["skewPhase"],  cfg["IQScale"],
                               style="const", freq=res_freq, phase=cfg["res_phase"], gain=cfg["res_gain"],
                                length=cfg["res_length"])

        self.set_pulse_registers(ch=self.cfg["qubit_ch"], style="arb",waveform="qubit",
                                 phase=self.deg2reg(90, gen_ch=cfg["qubit_ch"]),
                                 freq=qubit_freq, gain=cfg["start"])

        self.sync_all(self.us2cycles(1))  # give processor some time to configure pulses

    def body(self):
        cfg = self.cfg
        self.pulse(ch=self.cfg["qubit_ch"])  # play gaussian pulse
        self.sync_all(soc.us2cycles(0.05))  # align channels and wait 50ns

        # --- msmt
        self.trigger([cfg["ro_ch"]], adc_trig_offset=cfg["adc_trig_offset"])  # trigger the adc acquisition
        self.pulse(ch=cfg["res_ch_I"], t=0)
        self.pulse(ch=cfg["res_ch_Q"], t=0)
        self.wait_all()
        self.sync_all(self.us2cycles(cfg["relax_delay"])) # wait for qubit to relax

    def update(self):
        self.mathi(self.q_rp, self.r_gain, self.r_gain, '+', self.cfg["step"])  # update gain of the Gaussian pi pulse

if __name__ == "__main__":
    expt_cfg={
        "start":-30000,
        "step":200,
        "expts":300,
        "reps": 500,
        "relax_delay":200
           }
    config.update(expt_cfg) #combine configs

    print("running...")
    rabi=AmplitudeRabiProgram(soccfg, config)
    x_pts, avgi, avgq  = rabi.acquire(soc,load_pulses=True,progress=True, debug=False)
    print("done...\n plotting...")


    #Plotting Results
    plt.figure()
    plt.subplot(111, title= f"Amplitude Rabi, $\sigma={soc.cycles2us(config['sigma'])*1000}$ ns", xlabel="Gain", ylabel="Qubit Population" )
    plt.plot(x_pts,avgi[0][0],'o-', markersize = 1)
    plt.plot(x_pts,avgq[0][0],'o-', markersize = 1)

    piPul = qfr.PiPulseTuneUp(x_pts, avgi[0][0]+1j*avgq[0][0])
    piResult = piPul.run()
    piResult.plot()
    piResult.print_ge_rotation()


    # ----- save data to pc --------
    data_temp = {"x_data":x_pts, "i_data": avgi[0][0], "q_data": avgq[0][0]}
    saveData(data_temp, sampleName+"_pipulse", dataPath)

