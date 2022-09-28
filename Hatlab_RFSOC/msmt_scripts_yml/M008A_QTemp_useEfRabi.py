import matplotlib.pyplot as plt
import numpy as np
import lmfit

from Hatlab_DataProcessing.analyzer import qubit_functions_rot as qfr
from Hatlab_DataProcessing.fitter import qubit_functions as qf

from Hatlab_RFSOC.proxy import getSocProxy
from Hatlab_RFSOC.data import quick_save
from Hatlab_RFSOC.helpers import get_sweep_vals, plotData

import programs as msmt

from M000_ConfigSel import get_cfg_info
config, info = get_cfg_info()


if __name__ == "__main__":
    soc, soccfg = getSocProxy(info["PyroServer"])
    ADC_idx = info.get("ADC_idx",0)

    expt_cfg={
        "g_start": -30000, # MHz
        "g_stop": 30000,
        "g_expts": 201,

        "reps": 600,
        "sel_msmt": False, # to measure the qubit equilibrium temperature, THIS IS ALWAYS "False"!!!
        "relax_delay": 200, # make sure this is long enough that the qubit is relaxed to equilibrium
        "prepare_g": True,
        "flip_back_g": True # when true, add an extra pipulse after ef_rabi. Usually the cavity is driven at best g/e
                            # separation frequency, so setting this to True gives a better final msmt resolution.
        }

    config.update(expt_cfg)
    sweepGain = get_sweep_vals(config, "g")

    # -------- prepare g first--------------------------------------------------------------
    prog=msmt.EfRabiProgram(soccfg, config)
    expt_pts, avgi_g, avgq_g = prog.acquire(soc, load_pulses=True,progress=True, debug=False)
    # save data to ddh5
    quick_save(info["dataPath"], f"{info['sampleName']}_TemperatureMSMT_prepare_g", avgi_g, avgq_g, prog.di_buf_p, prog.dq_buf_p, config=config, sweepGain=sweepGain)

    # -------- prepare e ---------------------------------------------------------------------
    config["prepare_g"] = False
    prog=msmt.EfRabiProgram(soccfg, config)
    expt_pts, avgi_e, avgq_e = prog.acquire(soc, load_pulses=True,progress=True, debug=False)
    # save data to ddh5
    quick_save(info["dataPath"], f"{info['sampleName']}_TemperatureMSMT_prepare_e", avgi_e, avgq_e, prog.di_buf_p, prog.dq_buf_p, config=config, sweepGain=sweepGain)


    # plot IQ results
    plotData.plotAvgIQresults(sweepGain, [avgi_g[0], avgi_e[0]], [avgq_g[0], avgq_e[0]], title="Temperature MSMT using ef Rabi ",
                              xlabel="Drive Gain (DAC)", ylabel="Qubit IQ", sub_titles=["prepare g", "prepare e"])

    # fit IQ result
    # fit prepare e result first
    fit = qfr.PiPulseTuneUp(sweepGain, avgi_e[ADC_idx][0] + 1j * avgq_e[ADC_idx][0])
    fitResult1 = fit.run()
    amp1, f1 = fitResult1.get_fit_value("A"), fitResult1.get_fit_value("f")
    fitResult1.plot()
    # fit prepare g result using the amplitude period found above
    fit = qfr.PiPulseTuneUp(sweepGain, avgi_g[ADC_idx][0] + 1j * avgq_g[ADC_idx][0])
    fitResult2 = fit.run(f=lmfit.Parameter("f", f1, vary=False))
    amp2= fitResult2.get_fit_value("A")
    fitResult2.plot()

    print(f"thermo population: {amp2/(amp1+amp2)}")

