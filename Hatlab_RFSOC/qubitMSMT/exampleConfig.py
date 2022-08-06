PyroServer = "myqick216-01"

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
        "relax_delay": 250  # [us]
}

qubit_cfg={
    "sigma": 0.15, # us
    "ge_freq":3219.63343149,
    "sigma_ef": 0.15, # us
    "t2r_freq":3795.02,
    "ef_freq":3073.54,
    "pi_gain": 12882,
    "pi2_gain":6425
}

rotResult={'g_val': -0.06378638923450004, 'e_val': 0.08586107413682158, 'rot_angle': 2.65176794353093}


dataPath = r"L:\Data\LL_candlequbits\20220320_cooldown\\"
sampleName = "LL_Candle_0320_Q0"

config = {**hw_cfg, **readout_cfg, **qubit_cfg}
