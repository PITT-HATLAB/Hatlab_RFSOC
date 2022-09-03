# todo: write this as a function or class
if __name__ == "__main__":
    from Hatlab_RFSOC.proxy import getSocProxy
    import numpy as np
    import time
    from importlib import reload
    from tqdm import tqdm

    from data.data_transfer import saveData
    from Hatlab_DataProcessing.analyzer import qubit_functions_rot as qfr

    from qubitMSMT.exampleConfig import config, dataPath, sampleName, PyroServer

    from M004_T1 import T1Program
    from M005_T2R import RamseyProgram
    from M006_T2E import T2EProgram
    reload(qfr)

    soc, soccfg = getSocProxy(PyroServer)
    steps = 10
    t1_time_array = np.zeros(steps)
    t2_r_time_array = np.zeros(steps)
    t2_e_time_array = np.zeros(steps)
    t1_result_array = np.zeros(steps)
    t2_r_result_array = np.zeros(steps)
    t2_e_result_array = np.zeros(steps)

    t0 = time.time() #start time

    expt_cfg_t1={
        "start":0, # [us]
        "step": 1,  # [us]
        "expts": 450,
        "reps": 200,
        "relax_delay":300,
           }

    expt_cfg_t2={
        "start":0,  # [us]
        "step":0.8, # [us]
        "expts":1000,
        "reps": 200,
        "rounds": 1,
        "relax_delay":600 # [us]
           }

    config_t1 = {**config, **expt_cfg_t1} #combine configs
    config_t2 = {**config, **expt_cfg_t2}

    print("running...")
    for i in tqdm(range(steps)):
        t1p=T1Program(soccfg, config_t1)
        x_pts_t1, avgi_t1, avgq_t1 = t1p.acquire(soc, load_pulses=True, progress=False, debug=False)
        saveData(dict(t=time.time()-t0, i_data=avgi_t1[0][0], q_data=avgq_t1[0][0], x_pts_t1=x_pts_t1), sampleName+f"_t1_{i}", dataPath)

        t2p=RamseyProgram(soccfg, config_t2)
        x_pts_t2r, avgi_t2r, avgq_t2r= t2p.acquire(soc, load_pulses=True,progress=False, debug=False)
        saveData(dict(t=time.time()-t0, i_data=avgi_t2r[0][0], q_data=avgq_t2r[0][0], x_pts_t2r=x_pts_t2r), sampleName+f"_t2r_{i}", dataPath)

        t2ep=T2EProgram(soccfg, config_t2)
        x_pts_T2E, avgi_T2E, avgq_T2E= t2ep.acquire(soc, load_pulses=True,progress=False, debug=False)
        saveData(dict(t=time.time()-t0, i_data=avgi_T2E[0][0], q_data=avgq_T2E[0][0], x_pts_T2E=x_pts_T2E), sampleName+f"_t2e_{i}", dataPath)

    print("done...")