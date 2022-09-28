from importlib import reload
import M000_ConfigSel; reload(M000_ConfigSel) # just to make sure the data in config.py will update when running in same console

import matplotlib.pyplot as plt

from Hatlab_RFSOC.proxy import getSocProxy
from Hatlab_RFSOC.core.averager_program import NDAveragerProgram, QickSweep
from legacy.pulseConfig import add_prepare_msmt


from M000_ConfigSel import config, info
from Hatlab_DataProcessing.analyzer import qubit_functions_rot as qfr


class AmplitudeRabiProgram(NDAveragerProgram):
    def initialize(self):
        cfg = self.cfg
        self.res_ch = self.cfg["gen_chs"]["muxed_res"]["ch"]
        self.qubit_ch = self.cfg["gen_chs"]["q_drive"]["ch"]

        # set readout pulse registers
        self.set_pulse_registers(ch=self.res_ch, style="const", length=cfg["res_length"], mask=[0, 1, 2, 3])


        # add qubit pulse to q_drive channel
        self.add_waveform_from_cfg("q_drive", "q_gauss")
        self.set_pulse_params("q_drive", style="arb", waveform="q_gauss", phase=0,
                                freq=cfg["q_pulse_cfg"]["ge_freq"], gain=cfg["g_start"])

        # add qubit pulse gain sweep
        self.q_r_gain = self.get_reg("q_drive", "gain")
        self.q_r_gain_update = self.new_reg("q_drive", init_val=cfg["g_start"])
        self.add_sweep(QickSweep(self, self.q_r_gain_update, cfg["g_start"], cfg["g_stop"], cfg["g_expts"]))

        self.sync_all(self.us2cycles(1))  # give processor some time to configure pulses

    def body(self):
        cfg = self.cfg
        sel_msmt = cfg.get("sel_msmt", False)

        if sel_msmt:
            add_prepare_msmt(self, "q_drive", cfg["q_pulse_cfg"], "muxed_res", syncdelay=1)

        # drive and measure
        self.mathi(self.q_r_gain.page, self.q_r_gain.addr, self.q_r_gain_update.addr, '+', 0)  # set the updated gain value
        self.pulse(ch=self.qubit_ch)  # play gaussian pulse
        self.sync_all(self.us2cycles(0.05))  # align channels and wait 50ns
        # --- msmt
        self.measure(pulse_ch=self.res_ch,
                     adcs=self.ro_chs,
                     pins=[0],
                     adc_trig_offset=self.cfg["adc_trig_offset"],
                     wait=True,
                     syncdelay=self.us2cycles(self.cfg["relax_delay"]))


if __name__ == "__main__":
    soc, soccfg = getSocProxy(info["PyroServer"])
    ADC_idx = info["ADC_idx"]

    expt_cfg = {
        "g_start": -30000,
        "g_stop": 30000,
        "g_expts": 101,

        "reps": 200,
        "rounds": 1,

        "relax_delay": 200,  # [us]
        "sel_msmt": False
    }
    config.update(expt_cfg)  # combine configs

    prog = AmplitudeRabiProgram(soccfg, config)
    x_pts, avgi, avgq = prog.acquire(soc, load_pulses=True, progress=True, debug=False)
    x_pts = x_pts[0]

    # Plotting Results
    plt.figure()
    plt.subplot(111, title=f"Amplitude Rabi", xlabel="Gain", ylabel="Qubit IQ")
    plt.plot(x_pts, avgi[ADC_idx][0], 'o-', markersize=1)
    plt.plot(x_pts, avgq[ADC_idx][0], 'o-', markersize=1)

    piPul = qfr.PiPulseTuneUp(x_pts, avgi[ADC_idx][0] + 1j * avgq[ADC_idx][0])
    piResult = piPul.run()
    piResult.plot()
    piResult.print_ge_rotation()

    # histogram
    fig, ax = plt.subplots()
    hist = ax.hist2d(prog.di_buf[ADC_idx], prog.dq_buf[ADC_idx], bins=101)#, range=[[-400, 400], [-400, 400]])
    ax.set_aspect(1)
    fig.colorbar(hist[3])
    plt.show()

    from Hatlab_DataProcessing.slider_plot.sliderPlot import sliderHist2d
    sld = sliderHist2d(prog.di_buf_p[ADC_idx].T, prog.dq_buf_p[ADC_idx].T, {"amp":x_pts}, bins=101)

