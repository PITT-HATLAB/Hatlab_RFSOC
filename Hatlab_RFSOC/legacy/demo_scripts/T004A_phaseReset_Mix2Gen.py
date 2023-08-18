import time

from Hatlab_RFSOC.proxy import getSocProxy
from qick import *
import matplotlib.pyplot as plt
import numpy as np
soc, soccfg = getSocProxy("myqick216-01")

class LoopbackProgram(AveragerProgram):
    def initialize(self):
        cfg = self.cfg
        res_ch_0 = cfg["res_ch_0"]
        res_ch_1 = cfg["res_ch_1"]

        phrst = cfg["phrst"]
        # set the nyquist zone
        self.declare_gen(ch=cfg["res_ch_0"], nqz=1)
        self.declare_gen(ch=cfg["res_ch_1"], nqz=2)

        for ch in cfg["ro_chs"]:
            if self.soccfg['readouts'][ch]['tproc_ctrl'] is None:
                # freq_reg = self.freq2reg(cfg["pulse_freq_1"]-cfg["pulse_freq_0"], ro_ch=cfg["ro_chs"][ch], gen_ch=cfg['res_ch_0'])
                # freq_ro = self.reg2freq(freq_reg, gen_ch=cfg['res_ch_0'])
                self.declare_readout(ch=ch, length=cfg["readout_length"], freq=cfg["pulse_freq_1"]-cfg["pulse_freq_0"], gen_ch=cfg["res_ch_0"])
                # self.declare_readout(ch=ch, length=cfg["readout_length"], freq=cfg["pulse_freq_0"], gen_ch=cfg["res_ch_0"])
            else:
                # configure the readout lengths and downconversion frequencies (ensuring it is an available DAC frequency)
                self.declare_readout(ch=cfg["ro_chs"][ch], length=cfg["readout_length"])
                freq_ro = self.freq2reg_adc(cfg["pulse_freq_1"]-cfg["pulse_freq_0"], ro_ch=cfg["ro_chs"][ch], gen_ch=cfg['res_ch_0'])
                self.set_readout_registers(ch=cfg["ro_chs"][ch], freq=freq_ro, length=3, mode='oneshot', outsel='product', phrst=phrst)


        # convert frequency to DAC frequency (ensuring it is an available ADC frequency)
        freq_0 = self.freq2reg(cfg["pulse_freq_0"], gen_ch=res_ch_0, ro_ch=cfg["ro_chs"][0])
        freq_1 = self.freq2reg(cfg["pulse_freq_1"], gen_ch=res_ch_1, ro_ch=cfg["ro_chs"][0])
        gain = cfg["pulse_gain"]


        self.set_pulse_registers(ch=res_ch_0, style="const", freq=freq_0, phase=0, gain=gain, length=cfg["length"], phrst=phrst)
        self.set_pulse_registers(ch=res_ch_1, style="const", freq=freq_1, phase=0, gain=gain, length=cfg["length"], phrst=phrst)

        self.synci(200)  # give processor some time to configure pulses

    def body(self):
        # Configure readout v3.
        for ch in self.cfg["ro_chs"]:
            if self.soccfg['readouts'][ch]['tproc_ctrl'] is not None:
                self.readout(ch=ch, t=0)

        # self.sync_all(self.us2cycles(0.1))

        self.measure(pulse_ch=[self.cfg["res_ch_0"], self.cfg["res_ch_1"]],
                     adcs=self.ro_chs,
                     pins=[0],
                     adc_trig_offset=self.cfg["adc_trig_offset"],
                     wait=True,
                     syncdelay=self.us2cycles(self.cfg["relax_delay"]))




###################
# Try it yourself !
###################
if __name__ == "__main__":

    config = {"res_ch_0": 1,  # 1-4 GHz
              "res_ch_1": 3,  # > 6 GHz
              "ro_chs": [0],  #
              "reps": 1,  #
              "relax_delay": 1.0,  # --us

              "length": 150,  # [Clock ticks]

              "readout_length": 250,  # [Clock ticks]

              "pulse_gain": 30000,  # [DAC units]
              # Try varying pulse_gain from 500 to 30000 DAC units

              "pulse_freq_0": 4000,  # [MHz]
              "pulse_freq_1": 6000,  # [MHz]

              "adc_trig_offset": 100,  # [Clock ticks]
              # Try varying adc_trig_offset from 100 to 220 clock ticks
              "phrst": 1,
              "soft_avgs": 100
              # Try varying soft_avgs from 1 to 200 averages
              }

    # i_list = []
    # q_list = []
    # for n in range(30):
    prog = LoopbackProgram(soccfg, config)
    iq_list = prog.acquire_decimated(soc, load_pulses=True, progress=True, debug=False)

    # Plot results.
    plt.figure(1)
    for ii, iq in enumerate(iq_list):
        plt.plot(iq[0], label="I value, ADC %d"%(config['ro_chs'][ii]))
        plt.plot(iq[1], label="Q value, ADC %d"%(config['ro_chs'][ii]))
        plt.plot(np.abs(iq[0]+1j*iq[1]), label="mag, ADC %d"%(config['ro_chs'][ii]))
    plt.ylabel("a.u.")
    plt.xlabel("Clock ticks")
    plt.title("Averages = " + str(config["soft_avgs"]))
    plt.legend()
    #     i_list.append(np.sum(iq_list[0], axis=1)[0])
    #     q_list.append(np.sum(iq_list[0], axis=1)[1])
    #     time.sleep(0.01)
    # # plt.savefig("images/Send_recieve_pulse_const.pdf", dpi=350)
    # plt.figure()
    # plt.plot(i_list)
    # plt.plot(q_list)


    # from numpy.fft import fft, fftshift
    # # Phase.
    # i0 = 25
    # i1 = 180
    # iMean = iq[0][i0:i1].mean()
    # qMean = iq[1][i0:i1].mean()
    # ai = np.abs(iMean + 1j*qMean)
    # fi = np.angle(iMean + 1j*qMean)
    # print("Phase: {}".format(fi))
    #
    # # plt.figure()
    # # plt.plot(iq[0][i0:i1])
    # # plt.plot(iq[1][i0:i1])
    #
    # fs = 614.4
    # x = iq[0] + 1j*iq[1]
    # w = np.hanning(len(x))
    # xw = x*w
    # F = (np.arange(len(x))/len(x)-0.5)*fs
    # Y = fftshift(fft(xw))
    # Y = 20*np.log10(abs(Y))
    #
    # plt.figure(2,dpi=150)
    # plt.plot(F,Y)
    # plt.title("Spectrum")
    # plt.xlabel("F [MHz]")
