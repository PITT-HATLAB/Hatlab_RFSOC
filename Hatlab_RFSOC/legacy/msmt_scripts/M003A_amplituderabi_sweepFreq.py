"""
To demonstrate how to run a 2D sweep of driving amplitude and freq, in which both sweeps are done in qick.
The data is saved to plottr DDH5 format with the "HatDDH5Writer" manager
"""
from importlib import reload
import M000_ConfigSel; reload(M000_ConfigSel) # just to make sure the data in config.py will update when running in same console

import matplotlib.pyplot as plt

from Hatlab_RFSOC.proxy import getSocProxy
from Hatlab_RFSOC.core.averager_program import NDAveragerProgram, QickSweep
from Hatlab_RFSOC.helpers import add_prepare_msmt, get_sweep_vals
from Hatlab_RFSOC.data import QickDataDict


from M000_ConfigSel import config, info
from plottr.data import DataDict
from Hatlab_DataProcessing.data_saving import HatDDH5Writer

class AmplitudeRabiSweepFreqProgram(NDAveragerProgram):
    def initialize(self):
        cfg = self.cfg
        self.res_ch = self.cfg["gen_chs"]["muxed_res"]["ch"]
        self.qubit_ch = self.cfg["gen_chs"]["q_drive"]["ch"]

        # set readout pulse registers
        self.set_pulse_registers(ch=self.res_ch, style="const", length=cfg["res_length"], mask=[0, 1, 2, 3])

        # add qubit pulses to respective channels
        self.add_waveform_from_cfg("q_drive", "q_gauss")
        self.set_pulse_params("q_drive", style="arb", waveform="q_gauss", phase=0,
                                freq=cfg["f_start"], gain=cfg["g_start"])

        # add qubit pulse gain and freq sweep, first added will be first swept
        self.q_r_gain = self.get_reg("q_drive", "gain")
        self.q_r_gain_update = self.new_reg("q_drive", init_val=cfg["g_start"], name="gain_update")
        self.q_r_freq = self.get_reg("q_drive", "freq")
        self.q_r_freq_update = self.new_reg("q_drive", init_val=cfg["f_start"], name="freq_update", reg_type="freq")

        self.add_sweep(QickSweep(self, self.q_r_gain_update, cfg["g_start"], cfg["g_stop"], cfg["g_expts"]))
        self.add_sweep(QickSweep(self, self.q_r_freq_update, cfg["f_start"], cfg["f_stop"], cfg["f_expts"]))

        self.sync_all(self.us2cycles(1))  # give processor some time to configure pulses


    def body(self):
        cfg = self.cfg
        sel_msmt = cfg.get("sel_msmt", False)
        #
        if sel_msmt:
            add_prepare_msmt(self, "q_drive", cfg["q_pulse_cfg"], "muxed_res", syncdelay=1)
        #
        #
        # drive and measure
        self.mathi(self.q_r_gain.page, self.q_r_gain.addr, self.q_r_gain_update.addr, '+', 0)  # set the updated gain value
        self.mathi(self.q_r_freq.page, self.q_r_freq.addr, self.q_r_freq_update.addr, '+', 0)  # set the updated freq value
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
        "g_start": 0,
        "g_stop": 30000,
        "g_expts": 101,

        "f_start": 4871.344 - 40,
        "f_stop": 4871.344 + 40,
        "f_expts": 41,


        "reps": 50,
        "rounds": 1,
        "relax_delay": 200,  # [us]
        "sel_msmt":False
    }
    config.update(expt_cfg)  # combine configs


    # generate ddh5 writter
    gainList = get_sweep_vals(expt_cfg, "g")
    freqList = get_sweep_vals(expt_cfg, "f")

    inner_sweeps = DataDict(gain={"unit": "DAC", "values": gainList}, freq={"unit": "MHz", "values": freqList})

    qdd = QickDataDict(config["ro_chs"], inner_sweeps)
    filename = info["sampleName"] + f"test_ampRabi_sweepFreq"
    ddw = HatDDH5Writer(qdd, r"L:\Data\tests\\", filename=filename)


    with ddw as dw:
        dw.save_config(config)
        prog = AmplitudeRabiSweepFreqProgram(soccfg, config)
        x_pts, avgi, avgq = prog.acquire(soc, load_pulses=True, progress=True, debug=False)
        dw.add_data(inner_sweeps=inner_sweeps, avg_i=avgi, avg_q=avgq)

    plt.figure()
    plt.pcolormesh(freqList, gainList, avgi[ADC_idx][0].reshape((len(freqList),len(gainList))).T, shading="auto")
    plt.figure()
    plt.plot(gainList, avgi[ADC_idx][0].reshape((len(freqList),len(gainList)))[0])


