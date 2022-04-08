from proxy.socProxy import soccfg, soc
from qick import *
import matplotlib.pyplot as plt
import numpy as np
import time
import lmfit
from importlib import reload
from tqdm import tqdm

from helpers.pulseConfig import set_pulse_registers_IQ
from helpers.dataTransfer import saveData
from Hatlab_DataProcessing.analyzer import qubit_functions_rot as qfr

from qubitMSMT.config import config, rotResult, dataPath, sampleName

from M005_T1 import T1Program
from M006_T2R import RamseyProgram
from M007_T2E import T2EProgram
reload(qfr)

steps = 25
t1_time_array = np.zeros(steps)
t2_r_time_array = np.zeros(steps)
t2_e_time_array = np.zeros(steps)
t1_result_array = np.zeros(steps)
t2_r_result_array = np.zeros(steps)
t2_e_result_array = np.zeros(steps)

t0 = time.time() #start time

expt_cfg_t1={
    "start":0, # [us]
    "step": 2.5,  # [us]
    "expts": 100,
    "reps": 400,
    "relax_delay":250,
       }

expt_cfg_t2={
    "start":0,  # [us]
    "step":1, # [us]
    "expts":500,
    "reps": 200,
    "rounds": 1,
    "relax_delay":200 # [us]
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