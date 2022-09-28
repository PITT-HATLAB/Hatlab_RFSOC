from importlib import reload
import M000_ConfigSel; reload(M000_ConfigSel) # just to make sure the data in config.py will update when running in same console

import matplotlib.pyplot as plt

from Hatlab_RFSOC.proxy import getSocProxy
from Hatlab_RFSOC.core.averager_program import NDAveragerProgram, QickSweep
from Hatlab_RFSOC.helpers.pulseConfig import add_prepare_msmt


from M000_ConfigSel import config, info
from Hatlab_DataProcessing.analyzer import qubit_functions_rot as qfr


class T2RProgram(NDAveragerProgram):
    def initialize(self):
        cfg = self.cfg
        self.res_ch = self.cfg["gen_chs"]["muxed_res"]["ch"]
        self.qubit_ch = self.cfg["gen_chs"]["q_drive"]["ch"]

        # set readout pulse registers
        self.set_pulse_registers(ch=self.res_ch, style="const", length=cfg["res_length"], mask=[0, 1, 2, 3])


        # add qubit pulse to q_drive channel
        self.add_waveform_from_cfg("q_drive", "q_gauss")
        self.set_pulse_params("q_drive", style="arb", waveform="q_gauss", phase=0,
                                freq=cfg["q_pulse_cfg"]["t2r_freq"], gain=cfg["q_pulse_cfg"]["pi2_gain"])

        # add t_proc waiting time sweep
        self.t_r_wait = self.new_reg("q_drive", init_val=cfg["t_start"], reg_type="time", tproc_reg=True)
        self.add_sweep(QickSweep(self, self.t_r_wait, cfg["t_start"], cfg["t_stop"], cfg["t_expts"]))

        self.sync_all(self.us2cycles(1))  # give processor some time to configure pulses

    def body(self):
        cfg = self.cfg
        sel_msmt = cfg.get("sel_msmt", False)

        if sel_msmt:
            add_prepare_msmt(self, "q_drive", cfg["q_pulse_cfg"], "muxed_res", syncdelay=1)

        # drive and measure
        self.pulse(ch=self.qubit_ch)  # play pi/2 pulse
        self.sync_all()  # align channels and wait
        self.sync(self.t_r_wait.page, self.t_r_wait.addr)
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
    ADC_idx = info["ADC_idx"]

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

    prog = T2RProgram(soccfg, config)
    x_pts, avgi, avgq = prog.acquire(soc, load_pulses=True, progress=True, debug=False)
    x_pts = x_pts[0]

    # Plotting Results
    plt.figure()
    plt.subplot(111, title=f"T2R", xlabel="time (us)", ylabel="Qubit IQ")
    plt.plot(x_pts, avgi[ADC_idx][0], 'o-', markersize=1)
    plt.plot(x_pts, avgq[ADC_idx][0], 'o-', markersize=1)


    t1Decay = qfr.T2Ramsey(x_pts, avgi[ADC_idx][0] + 1j * avgq[ADC_idx][0])
    t1Result = t1Decay.run(info["rotResult"])
    t1Result.plot()

