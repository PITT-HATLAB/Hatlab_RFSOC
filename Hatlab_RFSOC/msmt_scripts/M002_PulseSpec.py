from importlib import reload
import M000_ConfigSel; reload(M000_ConfigSel) # just to make sure the data in config.py will update when running in same console

import matplotlib.pyplot as plt
import numpy as np


from Hatlab_RFSOC.proxy import getSocProxy
from Hatlab_RFSOC.core.averager_program import NDAveragerProgram, QickSweep

from M000_ConfigSel import config, info



class PulseSpecProgram(NDAveragerProgram):
    def initialize(self):
        cfg = self.cfg
        self.res_ch = self.cfg["gen_chs"]["muxed_res"]["ch"]
        self.qubit_ch = self.cfg["gen_chs"]["q_drive"]["ch"]

        # set readout pulse registers
        self.set_pulse_registers(ch=self.res_ch, style="const", length=cfg["res_length"], mask=[0, 1, 2, 3])

        # set qubit prob pulse registers
        self.set_pulse_params("q_drive", style="const", length=cfg["prob_length"],
                                 phase=0, freq=cfg["f_start"], gain=cfg["prob_gain"])

        # add prob pulse frequency sweep
        self.q_r_freq = self.get_reg("q_drive", "freq")
        self.add_sweep(QickSweep(self, self.q_r_freq, cfg["f_start"], cfg["f_stop"], cfg["f_expts"]))

        self.sync_all(self.us2cycles(1))  # give processor some time to configure pulses


    def body(self):
        self.pulse(ch=self.qubit_ch)  # play probe pulse
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

    expt_cfg={"f_start": 4866, # MHz
              "f_stop": 4876,
              "f_expts": 101,

              "prob_length": 10,
              "prob_gain":10,

              "reps": 200,
              "rounds": 1,
              "relax_delay": 100 #[us]
             }


    config.update(expt_cfg)

    prog=PulseSpecProgram(soccfg, config)
    expt_pts, avgi, avgq = prog.acquire(soc, load_pulses=True,progress=True, debug=False)

    #Plotting Results
    ADC_ch = info["ADC_ch"]
    sweepFreq = expt_pts[0] + config.get("qubit_mixer_freq", 0)
    plt.figure()
    plt.subplot(111,title="Qubit Spectroscopy", xlabel="Qubit Frequency (MHz)", ylabel="Qubit IQ")
    plt.plot(sweepFreq, avgi[ADC_ch][0],'o-', markersize = 1)
    plt.plot(sweepFreq, avgq[ADC_ch][0],'o-', markersize = 1)
    plt.show()

    print(sweepFreq[np.argmax(avgq[ADC_ch][0])])