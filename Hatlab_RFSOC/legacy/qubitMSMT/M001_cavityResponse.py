from qick import *
import matplotlib.pyplot as plt

from Hatlab_RFSOC.helpers.pulseConfig import set_pulse_registers_IQ, declareMuxedGenAndReadout

class CavityResponseProgram(AveragerProgram):
    def initialize(self):
        cfg = self.cfg

        self.declare_gen(ch=cfg["res_ch_I"], nqz=cfg["res_nzq_I"])  # resonator drive I
        self.declare_gen(ch=cfg["res_ch_Q"], nqz=cfg["res_nzq_Q"])  # resonator drive Q

        self.declare_readout(ch=cfg["ro_ch"], length=cfg["readout_length"],freq=cfg["res_freq"], gen_ch=cfg["res_ch_I"])

        res_freq = self.freq2reg(cfg["res_freq"], gen_ch=cfg["res_ch_I"], ro_ch=cfg["ro_ch"])  # convert frequency to dac frequency (ensuring it is an available adc frequency)

        set_pulse_registers_IQ(self, cfg["res_ch_I"], cfg["res_ch_Q"], cfg["skewPhase"],  cfg["IQScale"],
                               style="const", freq=res_freq, phase=cfg["res_phase"], gain=cfg["res_gain"],
                                length=cfg["res_length"])

        self.synci(200)  # give processor some time to configure pulses

    def body(self):
        cfg = self.cfg
        self.trigger([cfg["ro_ch"]], adc_trig_offset=cfg["adc_trig_offset"])
        self.pulse(ch=cfg["res_ch_I"], t=0)
        self.pulse(ch=cfg["res_ch_Q"], t=0)
        self.wait_all()
        # tProc should wait for the readout to complete.
        # This prevents loop counters from getting incremented before the data is available.
        self.sync_all(self.us2cycles(cfg["relax_delay"]))


class MuxedCavityResponseProgram(AveragerProgram):
    def initialize(self):
        cfg = self.cfg

        # declare muxed generator and readout channels
        declareMuxedGenAndReadout(self, cfg["res_ch"], cfg["res_nqz"], cfg["res_mixer_freq"],
                                  cfg["res_freqs"], cfg["res_gains"], cfg["ro_chs"], cfg["readout_length"])

        # set readout pulse registers
        self.set_pulse_registers(ch=cfg["res_ch"], style="const", length=cfg["res_length"], mask=[0, 1, 2, 3])

        self.synci(200)  # give processor some time to configure pulses

    def body(self):
        self.measure(pulse_ch=self.cfg["res_ch"],
                     adcs=self.ro_chs,
                     pins=[0],
                     adc_trig_offset=self.cfg["adc_trig_offset"],
                     wait=True,
                     syncdelay=self.us2cycles(self.cfg["relax_delay"]))


if __name__ == "__main__":
    from legacy.qubitMSMT import config, PyroServer
    soc, soccfg = getSocProxy(PyroServer)

    ## ------------- to get response trace ----------------
    prog = msmt.CavityResponseProgram(soccfg, config)
    adc1, = prog.acquire_decimated(soc, load_pulses=True, progress=True, debug=False)

    print("plotting")
    # Plot results.
    plt.figure()
    ax1 = plt.subplot(111, title=f"Averages = {config['soft_avgs']}", xlabel="Clock ticks",
                      ylabel="Transmission (adc levels)")
    ax1.plot(adc1[0], label="I value; ADC 0")
    ax1.plot(adc1[1], label="Q value; ADC 0")
    ax1.legend()

    ## ---- fit detuning ---
    # from Hatlab_DataProcessing.fitter.generic_functions import ExponentialDecayWithCosine
    # Idata = adc1[0][110:]
    # Qdata = adc1[1][110:]
    # tData = np.arange(0, len(Idata) * 1/384e6, 1/384e6)
    #
    # fit_ = ExponentialDecayWithCosine(tData, Idata)
    # fit_result = fit_.run()
    # fit_result.plot()


    ## -------------- to get integrated IQ points ---------------
    # readout_cfg = {
    #     "reps": 20000,  #
    #     "readout_length": 1020,  # [clock ticks]
    #
    #     "res_freq": 90,  # [MHz]
    #     # "res_gain": 25000,  # [DAC units]
    #     "res_length": 500,  # [clock ticks]
    #
    #     "rounds": 1,
    #     "relax_delay": 250  # [us]
    # }
    #
    # config = {**config, **readout_cfg}
    # prog = CavityResponseProgram(soccfg, config)
    # avgi, avgq = prog.acquire(soc, load_pulses=True, progress=True, debug=False)
    #
    # print("plotting")
    # # Plot results.
    # fig, ax = plt.subplots()
    # hist = ax.hist2d(prog.di_buf[0], prog.dq_buf[0], bins=101)#, range=[[-400, 400], [-400, 400]])
    # ax.set_aspect(1)
    # fig.colorbar(hist[3])
    # plt.show()





