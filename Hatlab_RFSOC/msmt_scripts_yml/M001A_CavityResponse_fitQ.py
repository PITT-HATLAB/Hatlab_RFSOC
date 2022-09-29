import matplotlib.pyplot as plt
import numpy as np

import programs as msmt
from Hatlab_RFSOC.proxy import getSocProxy
import Hatlab_RFSOC.helpers.plotData as plotdata
from Hatlab_DataProcessing.fitter import generic_functions as gf

from M000_ConfigSel import get_cfg_info
config, info = get_cfg_info()

if __name__ == "__main__":
    soc, soccfg = getSocProxy(info["PyroServer"])
    ro_ch = "ro_0"

    config["soft_avgs"] = 1000
    config["ro_chs"][ro_ch]["length"] = 1000
    config["reps"] = 1

    trigOffset0 = config["adc_trig_offset"]
    config["adc_trig_offset"] = trigOffset0 +  soc.us2cycles(config["res_pulse_config"]["length"]) + 10

    prog = msmt.CavityResponseProgram(soccfg, config)
    adc1, = prog.acquire_decimated(soc, load_pulses=True, progress=True, debug=False)

    # Plot results.
    plt.figure()
    ax1 = plt.subplot(111, title=f"Averages = {config['soft_avgs']}", xlabel="Clock ticks", ylabel="IQ (adc levels)")
    ax1.plot(adc1[0], label=f"I value; {ro_ch}")
    ax1.plot(adc1[1], label=f"Q value; {ro_ch}")
    ax1.legend()

    I_trace = adc1[0]
    Q_trace = adc1[1]
    time_trace = soc.cycles2us(np.arange(0, len(I_trace)), ro_ch=config["ro_chs"][ro_ch]["ch"])
    PwrTrace = I_trace ** 2 + Q_trace ** 2

    # Fitting
    cavDecay = gf.ExponentialDecay(time_trace, PwrTrace)
    fitResult = cavDecay.run()
    fitResult.plot()
    cavT1 = fitResult.params["tau"].value
    print(f"T1(us): {cavT1}")
    print(f"kappa/2/pi (MHz): {1 / cavT1 / 2 / np.pi}")




