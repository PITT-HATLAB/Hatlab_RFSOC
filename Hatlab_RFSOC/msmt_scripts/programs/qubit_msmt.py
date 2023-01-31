from typing import List
from qick.averager_program import AveragerProgram
from qick.qick_asm import QickProgram
from Hatlab_RFSOC.core.averager_program import NDAveragerProgram, QickSweep, QubitMsmtMixin



class CavityResponseProgram(QubitMsmtMixin, NDAveragerProgram):
    def initialize(self):
        cfg = self.cfg
        # declare res generator and readout channels
        self.res_ch = self.cfg["gen_chs"]["res_drive"]["ch"]
        for ro_cfg in cfg["ro_chs"].values():
            self.declare_readout(**ro_cfg)

        # set readout pulse registers
        self.set_pulse_params_auto_gen_type("res_drive", **cfg["res_pulse_config"])

        self.synci(200)  # give processor some time to configure pulses

    def body(self):
        self.measure(pulse_ch=self.res_ch,
                     adcs=self.ro_chs,
                     pins=[0],
                     adc_trig_offset=self.cfg["adc_trig_offset"],
                     wait=True,
                     syncdelay=self.us2cycles(self.cfg["relax_delay"]))

class PrepareQubitCavityResponseProgram(QubitMsmtMixin, NDAveragerProgram):
    def initialize(self):
        cfg = self.cfg
        # declare res generator and readout channels
        self.res_ch = self.cfg["gen_chs"]["res_drive"]["ch"]
        self.qubit_ch = self.cfg["gen_chs"]["q_drive"]["ch"]

        # set readout pulse registers
        self.set_pulse_params_auto_gen_type("res_drive", **cfg["res_pulse_config"])

        # set qubit prob pulse registers
        self.add_waveform_from_cfg("q_drive", "q_gauss")
        self.set_pulse_params("q_drive", style="arb", waveform="q_gauss", phase=0,
                                freq=cfg["q_pulse_cfg"]["ge_freq"], gain=cfg["prepare_gain"])

        self.synci(200)  # give processor some time to configure pulses

    def body(self):
        cfg = self.cfg
        # drive and measure
        self.set_pulse_params("q_drive", style="arb", waveform="q_gauss", phase=0,
                                freq=cfg["q_pulse_cfg"]["ge_freq"], gain=cfg["prepare_gain"])

        self.pulse(ch=self.qubit_ch)  # play gaussian pulse
        self.sync_all(self.us2cycles(0.05))  # align channels and wait 50ns

        self.measure(pulse_ch=self.res_ch,
                     adcs=self.ro_chs,
                     pins=[0],
                     adc_trig_offset=self.cfg["adc_trig_offset"],
                     wait=True,
                     syncdelay=self.us2cycles(self.cfg["relax_delay"]))

class PulseSpecProgram(QubitMsmtMixin, NDAveragerProgram):
    def initialize(self):
        cfg = self.cfg
        self.res_ch = self.cfg["gen_chs"]["res_drive"]["ch"]
        self.qubit_ch = self.cfg["gen_chs"]["q_drive"]["ch"]

        # set readout pulse registers
        self.set_pulse_params_auto_gen_type("res_drive", **cfg["res_pulse_config"])

        # set qubit prob pulse registers
        self.add_waveform_from_cfg("q_drive", "q_gauss") # for post sel
        self.set_pulse_params("q_drive", style="const", length=cfg["prob_length"],
                                phase=0, freq=cfg["f_start"], gain=cfg["prob_gain"])

        # add prob pulse frequency sweep
        self.q_r_freq = self.get_reg("q_drive", "freq")
        self.q_r_freq_update = self.new_reg("q_drive", reg_type="freq")
        self.add_sweep(QickSweep(self, self.q_r_freq_update, cfg["f_start"], cfg["f_stop"], cfg["f_expts"]))

        self.sync_all(self.us2cycles(1))  # give processor some time to configure pulses

    def body(self):
        cfg = self.cfg
        sel_msmt = cfg.get("sel_msmt", False)

        if sel_msmt:
            self.add_prepare_msmt("q_drive", cfg["q_pulse_cfg"], "res_drive", syncdelay=cfg["msmt_leakout_time"])

        # set pulse shape to prob pulse
        self.set_pulse_params("q_drive", style="const", length=cfg["prob_length"],
                                 phase=0, freq=cfg["f_start"], gain=cfg["prob_gain"])
        self.mathi(self.q_r_freq.page, self.q_r_freq.addr, self.q_r_freq_update.addr, '+', 0)  # set the updated freq value
        self.pulse(ch=self.qubit_ch)  # play prob pulse
        self.sync_all(self.us2cycles(0.05))  # align channels and wait 50ns

        # --- msmt
        self.measure(pulse_ch=self.res_ch,
                     adcs=self.ro_chs,
                     pins=[0],
                     adc_trig_offset=self.cfg["adc_trig_offset"],
                     wait=True,
                     syncdelay=self.us2cycles(self.cfg["relax_delay"]))


class AmplitudeRabiProgram(QubitMsmtMixin, NDAveragerProgram):
    def initialize(self):
        cfg = self.cfg
        self.res_ch = self.cfg["gen_chs"]["res_drive"]["ch"]
        self.qubit_ch = self.cfg["gen_chs"]["q_drive"]["ch"]

        # set readout pulse registers
        self.set_pulse_params_auto_gen_type("res_drive", **cfg["res_pulse_config"])


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
            self.add_prepare_msmt("q_drive", cfg["q_pulse_cfg"], "res_drive", syncdelay=cfg["msmt_leakout_time"])

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


class T1Program(QubitMsmtMixin, NDAveragerProgram):
    def initialize(self):
        cfg = self.cfg
        self.res_ch = self.cfg["gen_chs"]["res_drive"]["ch"]
        self.qubit_ch = self.cfg["gen_chs"]["q_drive"]["ch"]

        # set readout pulse registers
        self.set_pulse_params_auto_gen_type("res_drive", **cfg["res_pulse_config"])


        # add qubit pulse to q_drive channel
        self.add_waveform_from_cfg("q_drive", "q_gauss")
        self.set_pulse_params("q_drive", style="arb", waveform="q_gauss", phase=0,
                                freq=cfg["q_pulse_cfg"]["ge_freq"], gain=cfg["q_pulse_cfg"]["pi_gain"])

        # add t_proc waiting time sweep
        self.t_r_wait = self.new_reg("q_drive", init_val=cfg["t_start"], reg_type="time", tproc_reg=True)
        self.add_sweep(QickSweep(self, self.t_r_wait, cfg["t_start"], cfg["t_stop"], cfg["t_expts"]))

        self.sync_all(self.us2cycles(1))  # give processor some time to configure pulses

    def body(self):
        cfg = self.cfg
        sel_msmt = cfg.get("sel_msmt", False)

        if sel_msmt:
            self.add_prepare_msmt("q_drive", cfg["q_pulse_cfg"], "res_drive", syncdelay=cfg["msmt_leakout_time"])

        # drive and measure
        self.set_pulse_params("q_drive", style="arb", waveform="q_gauss", phase=0,
                                freq=cfg["q_pulse_cfg"]["ge_freq"], gain=cfg["q_pulse_cfg"]["pi_gain"])
        self.pulse(ch=self.qubit_ch)  # play gaussian pulse
        self.sync_all()  # align channels and wait 50ns
        self.sync(self.t_r_wait.page, self.t_r_wait.addr)
        # --- msmt
        self.measure(pulse_ch=self.res_ch,
                     adcs=self.ro_chs,
                     pins=[0],
                     adc_trig_offset=self.cfg["adc_trig_offset"],
                     wait=True,
                     syncdelay=self.us2cycles(self.cfg["relax_delay"]))


class T2RProgram(QubitMsmtMixin, NDAveragerProgram):
    def initialize(self):
        cfg = self.cfg
        self.res_ch = self.cfg["gen_chs"]["res_drive"]["ch"]
        self.qubit_ch = self.cfg["gen_chs"]["q_drive"]["ch"]

        # set readout pulse registers
        self.set_pulse_params_auto_gen_type("res_drive", **cfg["res_pulse_config"])


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
            self.add_prepare_msmt("q_drive", cfg["q_pulse_cfg"], "res_drive", syncdelay=cfg["msmt_leakout_time"])

        # drive and measure
        self.set_pulse_params("q_drive", style="arb", waveform="q_gauss", phase=0,
                                freq=cfg["q_pulse_cfg"]["t2r_freq"], gain=cfg["q_pulse_cfg"]["pi2_gain"])
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


class T2EProgram(QubitMsmtMixin, NDAveragerProgram):
    def initialize(self):
        cfg = self.cfg
        self.res_ch = self.cfg["gen_chs"]["res_drive"]["ch"]
        self.qubit_ch = self.cfg["gen_chs"]["q_drive"]["ch"]

        # set readout pulse registers
        self.set_pulse_params_auto_gen_type("res_drive", **cfg["res_pulse_config"])


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
            self.add_prepare_msmt("q_drive", cfg["q_pulse_cfg"], "res_drive", syncdelay=cfg["msmt_leakout_time"])

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


class EfPulseSpecProgram(QubitMsmtMixin, NDAveragerProgram):
    def initialize(self):
        cfg = self.cfg
        self.res_ch = self.cfg["gen_chs"]["res_drive"]["ch"]
        self.qubit_ch = self.cfg["gen_chs"]["q_drive"]["ch"]

        # set readout pulse registers
        self.set_pulse_params_auto_gen_type("res_drive", **cfg["res_pulse_config"])

        # set qubit prob pulse registers
        self.add_waveform_from_cfg("q_drive", "q_gauss")
        self.set_pulse_params("q_drive", style="const", length=cfg["prob_length"],
                                 phase=0, freq=cfg["f_start"], gain=cfg["prob_gain"])

        # add prob pulse frequency sweep
        self.q_r_freq = self.get_reg("q_drive", "freq")
        self.q_r_freq_update = self.new_reg("q_drive", reg_type="freq")
        self.add_sweep(QickSweep(self, self.q_r_freq_update, cfg["f_start"], cfg["f_stop"], cfg["f_expts"]))

        self.sync_all(self.us2cycles(1))  # give processor some time to configure pulses

    def body(self):
        cfg = self.cfg
        sel_msmt = cfg.get("sel_msmt", False)

        if sel_msmt:
            self.add_prepare_msmt("q_drive", cfg["q_pulse_cfg"], "res_drive", syncdelay=cfg["msmt_leakout_time"])

        # set and play ge gaussian pulse
        self.set_pulse_params("q_drive", style="arb", waveform="q_gauss", phase=0, freq=cfg["q_pulse_cfg"]["ge_freq"],
                              gain=cfg["q_pulse_cfg"]["pi_gain"])
        self.pulse(ch=self.qubit_ch)
        self.sync_all(self.us2cycles(0.05))  # align channels and wait 50ns

        # set pulse to prob pulse
        self.set_pulse_params("q_drive", style="const", length=cfg["prob_length"], phase=0, freq=cfg["f_start"], gain=cfg["prob_gain"])
        self.mathi(self.q_r_freq.page, self.q_r_freq.addr, self.q_r_freq_update.addr, '+', 0)  # set the updated freq value
        self.pulse(ch=self.qubit_ch)  # play prob pulse
        self.sync_all(self.us2cycles(0.05))  # align channels and wait 50ns

        if cfg.get("flip_back_g", True): # for better final msmt resolution (usually the cavity is driven at best g/e separation)
            self.set_pulse_params("q_drive", style="arb", waveform="q_gauss", phase=0, freq=cfg["q_pulse_cfg"]["ge_freq"],
                                  gain=cfg["q_pulse_cfg"]["pi_gain"])
            self.pulse(ch=self.qubit_ch)
            self.sync_all(self.us2cycles(0.05))  # align channels and wait 50ns

        # --- msmt
        self.measure(pulse_ch=self.res_ch,
                     adcs=self.ro_chs,
                     pins=[0],
                     adc_trig_offset=self.cfg["adc_trig_offset"],
                     wait=True,
                     syncdelay=self.us2cycles(self.cfg["relax_delay"]))


class EfRabiProgram(QubitMsmtMixin, NDAveragerProgram):
    def initialize(self):
        cfg = self.cfg
        self.res_ch = self.cfg["gen_chs"]["res_drive"]["ch"]
        self.qubit_ch = self.cfg["gen_chs"]["q_drive"]["ch"]

        # set readout pulse registers
        self.set_pulse_params_auto_gen_type("res_drive", **cfg["res_pulse_config"])

        # set qubit drive pulse registers
        self.add_waveform_from_cfg("q_drive", "q_gauss")
        self.set_pulse_params("q_drive", style="arb", waveform="q_gauss", phase=0, freq=cfg["q_pulse_cfg"]["ge_freq"],
                              gain=cfg["q_pulse_cfg"]["pi_gain"])

        # add ef pulse gain sweep
        self.q_r_gain = self.get_reg("q_drive", "gain")
        self.q_r_gain_update = self.new_reg("q_drive")
        self.add_sweep(QickSweep(self, self.q_r_gain_update, cfg["g_start"], cfg["g_stop"], cfg["g_expts"]))

        self.sync_all(self.us2cycles(1))  # give processor some time to configure pulses

    def body(self):
        cfg = self.cfg
        sel_msmt = cfg.get("sel_msmt", False)

        if sel_msmt:
            self.add_prepare_msmt("q_drive", cfg["q_pulse_cfg"], "res_drive", syncdelay=cfg["msmt_leakout_time"])

        # set and play ge gaussian pulse
        if not cfg["prepare_g"]: # for temperature msmt
            self.set_pulse_params("q_drive", style="arb", waveform="q_gauss", phase=0, freq=cfg["q_pulse_cfg"]["ge_freq"],
                                  gain=cfg["q_pulse_cfg"]["pi_gain"])
            self.pulse(ch=self.qubit_ch)
            self.sync_all(self.us2cycles(0.05))  # align channels and wait 50ns

        # set and play ef gaussian pulse
        self.set_pulse_params("q_drive", style="arb", waveform="q_gauss", phase=0, freq=cfg["q_pulse_cfg"]["ef_freq"],
                              gain=cfg["q_pulse_cfg"]["pi_gain"])
        self.mathi(self.q_r_gain.page, self.q_r_gain.addr, self.q_r_gain_update.addr, '+', 0)  # set the updated gain value
        self.pulse(ch=self.qubit_ch)  # play ef pulse
        self.sync_all(self.us2cycles(0.05))  # align channels and wait 50ns

        if cfg["flip_back_g"]: # for better final msmt resolution (usually the cavity is driven at best g/e separation)
            self.set_pulse_params("q_drive", style="arb", waveform="q_gauss", phase=0, freq=cfg["q_pulse_cfg"]["ge_freq"],
                                  gain=cfg["q_pulse_cfg"]["pi_gain"])
            self.pulse(ch=self.qubit_ch)
            self.sync_all(self.us2cycles(0.05))  # align channels and wait 50ns

        # --- msmt
        self.measure(pulse_ch=self.res_ch,
                     adcs=self.ro_chs,
                     pins=[0],
                     adc_trig_offset=self.cfg["adc_trig_offset"],
                     wait=True,
                     syncdelay=self.us2cycles(self.cfg["relax_delay"]))