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
        "adc_trig_offset": 180,  # [clock ticks]
        "readout_length": 1010,  # [clock ticks]

        'res_freq': 90,
        'res_gain': 25000,
        'res_length': 500,
        'res_phase': 0,
        'skewPhase': 90,
        'IQScale': 1.0,

        "soft_avgs": 5000,
        "relax_delay": 200  # [us]
}

rotResult={'g_val': -0.09101353892194847, 'e_val': -0.4851972140193664, 'rot_angle': 0.03955298928492021}

dataPath = r"L:\Data\WISPE\LL_WISPE\s6\cooldown_20220401\\"
sampleName = "LL_Wispe_0401_candle1"


config = {**hw_cfg, **readout_cfg}
