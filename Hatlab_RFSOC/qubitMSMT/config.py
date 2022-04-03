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
        "adc_trig_offset": 0,  # [clock ticks]
        "readout_length": 1010,  # [clock ticks]

        "res_freq": 90,  # [MHz]
        "res_gain": 3000,  # [DAC units]
        "res_length": 500,  # [clock ticks]
        "res_phase": 0,  # [deg]
        "skewPhase": 83,  # [Degrees]
        "IQScale": 1.03,

        "soft_avgs": 1000,
        "relax_delay": 1  # [us]
}


config = {**hw_cfg, **readout_cfg}
