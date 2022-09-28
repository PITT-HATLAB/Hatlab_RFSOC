"""
To demonstrate how to sweep the flat part length of a flat_top pulse.
"""

from importlib import reload
import M000_ConfigSel; reload(M000_ConfigSel) # just to make sure the data in config.py will update when running in same console

import matplotlib.pyplot as plt

from Hatlab_RFSOC.proxy import getSocProxy
from Hatlab_RFSOC.core.averager_program import NDAveragerProgram, FlatTopLengthSweep
from Hatlab_RFSOC.helpers.pulseConfig import add_prepare_msmt


from M000_ConfigSel import config, info
from Hatlab_DataProcessing.analyzer import qubit_functions_rot as qfr


class LengthRabiProgram(NDAveragerProgram):
    def initialize(self):
        cfg = self.cfg
        self.res_ch = self.cfg["gen_chs"]["muxed_res"]["ch"]
        self.qubit_ch = self.cfg["gen_chs"]["q_drive"]["ch"]

        # set readout pulse registers
        self.set_pulse_registers(ch=self.res_ch, style="const", length=cfg["res_length"], mask=[0, 1, 2, 3])


        # add qubit pulse to q_drive channel, which is a flat_top pulse that raises and falls with the first and second half of a gaussian
        self.add_waveform_from_cfg("q_drive", "q_gauss")
        self.set_pulse_params("q_drive", style="flat_top", waveform="q_gauss", phase=0,
                                freq=cfg["q_pulse_cfg"]["ge_freq"], gain=cfg["gain"], length=cfg["l_start"])

        # sweeps the pulse flat part length, and waiting time in tproc after the pulse
        self.t_r_wait = self.new_reg("q_drive", init_val=cfg["l_start"], reg_type="time", tproc_reg=True)
        self.q_r_mode = self.get_reg("q_drive", "mode")
        self.q_r_mode_update = self.new_reg("q_drive")
        swp = FlatTopLengthSweep(self, self.q_r_mode_update, cfg["l_start"], cfg["l_stop"], cfg["l_expts"], self.t_r_wait)
        self.add_sweep(swp)

        # ramp part length
        self.t_ramp_length_reg = self.t_r_wait.val2reg(cfg["waveforms"]["q_gauss"]["length"] + 0.02)

        self.sync_all(self.us2cycles(1))  # give processor some time to configure pulses

    def body(self):
        cfg = self.cfg
        sel_msmt = cfg.get("sel_msmt", False)

        if sel_msmt:
            add_prepare_msmt(self, "q_drive", cfg["q_pulse_cfg"], "muxed_res", syncdelay=1)
            # set the q drive pulse shape back to flat top
            self.set_pulse_params("q_drive", style="flat_top", waveform="q_gauss", phase=0,
                                  freq=cfg["q_pulse_cfg"]["ge_freq"], gain=cfg["gain"], length=cfg["l_start"])

        # set the updated mode value (update pulse length)
        self.mathi(self.q_r_mode.page, self.q_r_mode.addr, self.q_r_mode_update.addr, '+', 0)
        # drive and measure
        self.pulse(ch=self.qubit_ch)  # play gaussian pulse
        self.synci(self.t_ramp_length_reg) # total ramp length (raising and lowering)
        self.sync(self.t_r_wait.page, self.t_r_wait.addr)  # flat part width

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
        "l_start": 0.01,
        "l_stop": 1.01,
        "l_expts": 101,
        "gain": 3000,

        "reps": 200,
        "rounds": 1,

        "relax_delay": 200,  # [us]
        "sel_msmt": False
    }
    config.update(expt_cfg)  # combine configs

    prog = LengthRabiProgram(soccfg, config)
    x_pts, avgi, avgq = prog.acquire(soc, load_pulses=True, progress=True, debug=False,
                                     readouts_per_experiment=int(expt_cfg["sel_msmt"])+1)
    x_pts = x_pts[0]

    # Plotting Results
    plt.figure()
    plt.subplot(111, title=f"Length Rabi", xlabel="Length (us)", ylabel="Qubit IQ")
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

