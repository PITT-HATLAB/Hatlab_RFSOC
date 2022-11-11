import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
import lmfit

# from Hatlab_DataProcessing.fitter import cavity_functions_hanger as cfr
from Hatlab_DataProcessing.fitter import cavity_functions as cfr

from Hatlab_RFSOC.proxy import getSocProxy
from Hatlab_RFSOC.data import quick_save
from Hatlab_RFSOC.experiment import Experiment

from instrumentserver import Client
from plottr.data import DataDict

import programs as msmt

cli = Client()
SC_C = cli.get_instrument("SC_C")


from M000_ConfigSel import get_cfg_info
config, info = get_cfg_info()


if __name__ == "__main__":
    freqList = np.linspace(-5e6, 5e6, 101) + 8.87525383e9
    avgi_array = np.zeros((2, len(freqList)))
    avgq_array = np.zeros((2, len(freqList)))
    data_base_dir = r"L:\Data\chen_wang\20220902_cooldown\test\\"
    filename = info["sampleName"] + f"_chi_msmt"

    soc, soccfg = getSocProxy(info["PyroServer"])

    readout_cfg = {
        "g_start": config["q_pulse_cfg"]["pi_gain"],
        "g_stop": 0,
        "g_expts": 2,


        "reps": 2000,
        "relax_delay": 200 # incase the value in config was a short time used for postSel expts

    }
    config.update(readout_cfg)


    # sweep cavity drive LO frequency with sweep manger "Experiment"
    inner_sweeps = DataDict(q_drive_gain={"unit": "DAC", "values": [config["q_pulse_cfg"]["pi_gain"], 0]})
    outer_sweeps = DataDict(res_LO_freq={"unit": "GHz", "values": freqList})
    expt = Experiment(msmt.AmplitudeRabiProgram, config, info, inner_sweeps, outer_sweeps,
                      data_base_dir=data_base_dir, data_file_name=filename)
    for i, freq in enumerate(tqdm(freqList)):
        SC_C.frequency(freq)
        x_pts, avgi, avgq = expt.run(save_data=True, save_buf=False, inner_progress=False, res_LO_freq=freq)
        avgi_array[:, i], avgq_array[:, i] = avgi[0][0], avgq[0][0]

    # for i, freq in enumerate(tqdm(freqList)):
    #     SC_C.frequency(freq)
    #     prog = msmt.AmplitudeRabiProgram(soccfg, config)
    #     x_pts, avgi, avgq = prog.acquire(soc, load_pulses=True, progress=False, debug=False)


    g_trace = avgi_array[0] + 1j*avgq_array[0]
    e_trace = avgi_array[1] + 1j*avgq_array[1]

    # --------- processing result-------------------------
    plt.figure()
    plt.subplot(111, title= f"drive ge pi pulse, vary f cavity", xlabel="freq", ylabel="I Q" )
    plt.plot(freqList, avgi_array[0],'o-', markersize = 1)
    plt.plot(freqList, avgq_array[0],'o-', markersize = 1)
    plt.plot(freqList, avgi_array[1],'*-', markersize = 1)
    plt.plot(freqList, avgq_array[1],'*-', markersize = 1)
    plt.figure(figsize=(13,5))
    plt.subplot(121, title= f"drive ge pi pulse, vary f cavity", xlabel="freq", ylabel="amp" )
    plt.plot(freqList, np.abs(g_trace),'o-', markersize = 1)
    plt.plot(freqList, np.abs(e_trace),'o-', markersize = 1)
    plt.subplot(122, title= f"drive ge pi pulse, vary f cavity", xlabel="freq", ylabel="phase" )
    plt.plot(freqList, np.unwrap(np.angle(g_trace)),'o-', markersize = 1)
    plt.plot(freqList, np.unwrap(np.angle(e_trace)),'o-', markersize = 1)
    chi_fromMag = np.abs(freqList[np.argmin(np.abs(g_trace))]-freqList[np.argmin(np.abs(e_trace))])/1e6
    print(f"chi from magnitude: {chi_fromMag} MHz")


    cavRef_g = cfr.CavReflectionPhaseOnly(freqList, g_trace, conjugate=True)
    futCavRef_g = cavRef_g.run()
    futCavRef_g.plot()

    cavRef_e = cfr.CavReflectionPhaseOnly(freqList, e_trace, conjugate=True)
    futCavRef_e = cavRef_e.run(params={"Qext":lmfit.Parameter("Qext", value=futCavRef_g.Qext)})
    futCavRef_e.plot()


    plt.figure()
    plt.plot(freqList, futCavRef_g.lmfit_result.data, ".")
    plt.plot(freqList, futCavRef_g.lmfit_result.best_fit)
    plt.plot(freqList, futCavRef_e.lmfit_result.data, ".")
    plt.plot(freqList, futCavRef_e.lmfit_result.best_fit)

    print("chi from fitting:", np.abs(futCavRef_g.f0-futCavRef_e.f0)/1e6, "MHz")
    print("best res freq:", (futCavRef_g.f0+futCavRef_e.f0)/2/1e9, "GHz")

    print(cavRef_e.guess(cavRef_e.coordinates, cavRef_e.data))