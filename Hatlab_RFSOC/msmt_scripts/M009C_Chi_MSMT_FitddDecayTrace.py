import matplotlib.pyplot as plt
import numpy as np

import programs as msmt
from Hatlab_RFSOC.proxy import getSocProxy
import Hatlab_RFSOC.helpers.plotData as plotdata
from Hatlab_DataProcessing.fitter import generic_functions as gf
from Hatlab_RFSOC.data.data_transfer import saveData
from Hatlab_DataProcessing.analyzer.qubit_functions_rot import T2Ramsey
from Hatlab_DataProcessing.fitter.time_domain_cavity_functions import CavTraceRefDecay
from Hatlab_DataProcessing.fitter.generic_functions import Linear, ExponentialDecay

from M000_ConfigSel import get_cfg_info
config, info = get_cfg_info()

if __name__ == "__main__":
    soc, soccfg = getSocProxy(info["PyroServer"])


    expt_cfg = {
        "soft_avgs": 2000,
        "reps": 1,
        "adc_trig_offset": config["adc_trig_offset"] +  soc.us2cycles(config["res_pulse_config"]["length"]) + 100
    }

    config["ro_chs"]["ro_0"]["length"] = 1000

    config.update(expt_cfg)


    fig, axs = plt.subplots(1, 2, figsize=(12,5))
    delta_list = np.zeros(2)
    kappa_list = np.zeros(2)
    for i in range(2):
        config["prepare_gain"] = config["q_pulse_cfg"]["pi_gain"] * i

        prog = msmt.PrepareQubitCavityResponseProgram(soccfg, config)
        adc1, = prog.acquire_decimated(soc, load_pulses=True, progress=True, debug=False)
        I_trace = adc1[0]
        Q_trace = adc1[1]
        t_trace = soc.cycles2us(np.arange(0, len(I_trace)), ro_ch=config["ro_chs"]["ro_0"]["ch"])
        pwr_trace =  I_trace ** 2 + Q_trace ** 2
        phase_trace = np.unwrap(np.angle(I_trace + 1j * Q_trace))

        # fit phase results.
        fit = CavTraceRefDecay(t_trace*1e-6, I_trace + 1j * Q_trace)
        result = fit.run()
        result.plot(axs[i])
        delta_list[i] = result.params['delta_f'].value/1e6
        kappa_list[i] = result.params['k'].value/1e6/2/np.pi

        if i==0:
            saveData({"t_trace":t_trace,"I_trace": I_trace,"Q_trace": Q_trace}, f"{info['sampleName']}_chiMSMT_g_decayTrace", info["dataPath"]+"qubitMSMT\\" )
            axs[i].set_title(f"prepare g, delta={np.round(delta_list[i],5)} MHz, kappa/2pi={np.round(kappa_list[i],5)} MHz")

        else:
            saveData({"t_trace":t_trace,"I_trace": I_trace,"Q_trace": Q_trace}, f"{info['sampleName']}_chiMSMT_e_decayTrace", info["dataPath"]+"qubitMSMT\\" )
            axs[i].set_title(f"prepare e, delta={np.round(delta_list[i],5)} MHz, kappa/2pi={np.round(kappa_list[i],5)} MHz")

    chi = delta_list[0]-delta_list[1]
    # this could give either positive or negative based on how the readout down-conversion circuit
    # was built (RF>LO or RF<L0) or which ADC nyquist zone was used. But the abs value should always be correct.

    print(f"!!!!!!!!!!!! kappa/2pi={np.average(kappa_list)} MHz")
    print(f"!!!!!!!!!!!! chi/2pi={-abs(chi)} MHz")
    # the best res drive freq should always be correct when calculated in this way, no matter how the ADC was configured.
    print(f"!!!!!!!!!!!! best res drive freq: {config['res_pulse_config']['freq']-(delta_list[1]+delta_list[0])/2}")



