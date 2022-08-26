from importlib import reload
import M000_ConfigSel; reload(M000_ConfigSel) # just to make sure the data in config.py will update when running in same console

import matplotlib.pyplot as plt
import numpy as np


from Hatlab_RFSOC.proxy import getSocProxy
from Hatlab_RFSOC.core import NDAveragerProgram, QickSweep
from Hatlab_RFSOC.helpers import get_sweep_vals, add_prepare_msmt


from M000_ConfigSel import config, info
from Hatlab_DataProcessing.analyzer import qubit_functions_rot as qfr


class T2EProgram(NDAveragerProgram):
    def initialize(self):
        cfg = self.cfg
        self.res_ch = self.cfg["gen_chs"]["muxed_res"]["ch"]
        self.qubit_ch = self.cfg["gen_chs"]["q_drive"]["ch"]

        # set readout pulse registers
        self.set_pulse_registers(ch=self.res_ch, style="const", length=cfg["res_length"], mask=[0, 1, 2, 3])


        # add qubit pulse to q_drive channel
        self.add_waveform_from_cfg("q_drive", "q_gauss")
        self.set_pulse_params("q_drive", style="arb", waveform="q_gauss", phase=0,
                                freq=cfg["q_pulse_cfg"]["ge_freq"], gain=cfg["q_pulse_cfg"]["pi2_gain"])

        # add t_proc waiting time sweep
        self.t_r_wait = self.new_reg("q_drive", init_val=cfg["t_start"], reg_type="time", tproc_reg=True)
        self.add_sweep(QickSweep(self, self.t_r_wait, cfg["t_start"]/2, cfg["t_stop"]/2, cfg["t_expts"]))

        self.sync_all(self.us2cycles(1))  # give processor some time to configure pulses

    def body(self):
        cfg = self.cfg
        sel_msmt = cfg.get("sel_msmt", False)

        if sel_msmt:
            add_prepare_msmt(self, "q_drive", cfg["q_pulse_cfg"], "muxed_res", syncdelay=1, setback_pi_gain=False)

        # drive and measure
        # play pi/2 pulse
        self.pulse(ch=self.qubit_ch)
        self.sync_all()  # align channels and wait
        self.sync(self.t_r_wait.page, self.t_r_wait.addr)

        # play pi pulse
        self.set_pulse_params("q_drive", style="arb", waveform="q_gauss", phase=0,
                              freq=cfg["q_pulse_cfg"]["ge_freq"], gain=cfg["q_pulse_cfg"]["pi_gain"])
        self.pulse(ch=self.qubit_ch)
        self.sync_all()  # align channels and wait
        self.sync(self.t_r_wait.page, self.t_r_wait.addr)

        # play pi/2 pulse
        self.set_pulse_params("q_drive", style="arb", waveform="q_gauss", phase=0,
                              freq=cfg["q_pulse_cfg"]["ge_freq"], gain=cfg["q_pulse_cfg"]["pi2_gain"])
        self.pulse(ch=self.qubit_ch)  # play pi/2 pulse
        self.sync_all(0.05)  # align channels and wait
        # --- msmt
        self.measure(pulse_ch=self.res_ch,
                     adcs=self.ro_chs,
                     pins=[0],
                     adc_trig_offset=self.cfg["adc_trig_offset"],
                     wait=True,
                     syncdelay=self.us2cycles(self.cfg["relax_delay"]))


if __name__ == "__main__":
    soc, soccfg = getSocProxy(info["PyroServer"])
    ADC_ch = info["ADC_ch"]

    expt_cfg = {
        "t_start": 0.05,
        "t_stop": 20.05,
        "t_expts": 101,

        "reps": 300,
        "rounds": 1,

        "relax_delay": 200,  # [us]
        "sel_msmt":False
    }
    config.update(expt_cfg)  # combine configs

    prog = T2EProgram(soccfg, config)
    _, avgi, avgq = prog.acquire(soc, load_pulses=True, progress=True, debug=False)
    x_pts = get_sweep_vals(expt_cfg, "t")

    # Plotting Results
    plt.figure()
    plt.subplot(111, title=f"T2E", xlabel="time (us)", ylabel="Qubit IQ")
    plt.plot(x_pts, avgi[ADC_ch][0], 'o-', markersize=1)
    plt.plot(x_pts, avgq[ADC_ch][0], 'o-', markersize=1)


    t2eDecay = qfr.T1Decay(x_pts, avgi[ADC_ch][0] + 1j * avgq[ADC_ch][0])
    t2eresult = t2eDecay.run(info["rotResult"])
    t2eresult.plot()

