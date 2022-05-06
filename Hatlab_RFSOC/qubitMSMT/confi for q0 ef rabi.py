from proxy.socProxy import soccfg, soc

hw_cfg={"res_ch_I": 5,
        "res_ch_Q": 6,
        "qubit_ch": 2,
        "ro_ch": 0,

        "res_nzq_I": 1,
        "res_nzq_Q": 1,
        "qubit_nzq": 2
       }


readout_cfg = {
        "reps": 1,  # --Fixed
        "adc_trig_offset": 200,  # [clock ticks]
        "readout_length": 1020,  # [clock ticks]

        'res_freq': 90,
        'res_gain': 20000,
        'res_length': 500,
        'res_phase': 0,
        'skewPhase': 90,
        'IQScale': 1.0,

        "soft_avgs": 5000,
        "relax_delay": 200  # [us]
}

qubit_cfg={
    "sigma": soc.us2cycles(0.15),
    "ge_freq":3219.63343149,
    "sigma_ef": soc.us2cycles(0.7),
    "t2r_freq":3219.60343149,
    "ef_freq":3073.71,
    "pi_gain": 12195,
    "pi2_gain":6100
}

rotResult={'g_val': -0.11672272363672004, 'e_val': -0.19615214153844546, 'rot_angle': 0.619856103079681}


dataPath = r"L:\Data\LL_candlequbits\20220320_cooldown\\"
sampleName = "LL_Candle_0320_Q0"

config = {**hw_cfg, **readout_cfg, **qubit_cfg}
