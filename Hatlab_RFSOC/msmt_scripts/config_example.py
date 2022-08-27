# ---------------- qick configs ------------------
gen_chs = {
    "muxed_res":{
        "ch": 6,
        "nqz": 1,
        "mixer_freq": 1200,
        "mux_freqs": [203.9827, 824.5708, 0, 0],
        "mux_gains": [0.5, 1, 0, 0],
        "ro_ch": 0
    },
    "q_drive":{"ch": 2, "nqz": 2},
    "snail_c":{"ch": 0, "nqz": 1, "ro_ch": 0},

}


ro_chs = {
    "ro_0":{ "ch": 0, "freq": 203.9827, "length": 1000, "gen_ch": 6},
    "ro_1":{ "ch": 1, "freq": 824.5708, "length": 1000, "gen_ch": 6}
}


waveforms = {             
    "q_gauss":{"shape": "gaussian", "sigma": 0.01, "length": 0.05},

}


q_pulse_cfg={
    "waveform": "q_gauss",
    "ge_freq": 4871.3421102,
    "pi_gain": 12366,
    "pi2_gain":6185,
    "t2r_freq":4871.8421102,
    "ef_freq":4697.21
}


readout_cfg={
    'res_length': 800, # cycles
     "adc_trig_offset": 90,  # [clock ticks]
     "relax_delay": 200,  # [us]
}


config = {"gen_chs":gen_chs, "ro_chs":ro_chs, "waveforms":waveforms,
    "q_pulse_cfg": q_pulse_cfg, **readout_cfg}



# ---------------- miscellaneous infos ------------------
PyroServer = "myqick216-01"
rotResult = {'g_val': 126.02057122904796, 'e_val': -78.79001905612225, 'rot_angle': 25.673808378975302}
dataPath = r"L:\Data\SNAIL_Pump_Limitation\\"
sampleName = "Q3"

ADC_idx = 1

info = {"PyroServer":PyroServer, "rotResult":rotResult, "dataPath":dataPath,
        "sampleName":sampleName, "ADC_idx":ADC_idx}


print("cfg reloaded")