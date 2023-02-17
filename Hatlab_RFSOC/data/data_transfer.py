"""
This module was originally written for transferring data from RFSoC to local PC.
We probably don't need these functions anymore if we run experiments with pyro.
"""


import subprocess
from typing import Dict
import json
import h5py
import yaml
import numpy as np
from os.path import exists
import os
from Hatlab_RFSOC.helpers.yaml_editor import to_yaml_friendly

HOMEPATH = "/home/xilinx"


def saveData(data:Dict, fileName, filePath, fileType:str="h5", cfg:Dict=None):
    try:
        os.makedirs(filePath)
    except FileExistsError:
        pass

    fn_ = fileName
    ii = 2
    while True:
        if exists(filePath+fn_+f".{fileType}"):
            fn_ = fileName + f"-{ii}"
            ii += 1
        else:
            break

    if fileType in ["json", "JSON"]:
        with open(filePath + fn_ + ".json", 'w') as outfile:
            json.dump(data, outfile, default=_jsonDefaultRules)
    elif fileType in ["h5", "H5", "hdf5", "HDF5"]:
        with h5py.File(filePath + fn_ + ".h5", "w-") as f:
            for s, var in data.items():
                f.create_dataset(s, data=var)
    elif fileType in ["yaml", "yml", "YAML"]:
        with open(filePath + fn_ + ".yml", "w-") as file:
            yaml.dump(data, file)
    else:
        raise NotImplementedError(f"Don't know what to do with file type {fileType}, try 'json', 'h5', or 'yml'")

    if cfg is not None:
        save_config(f"{fn_}_cfg.yaml", filePath, to_yaml_friendly(cfg))

    return fn_


def loadData(fileName:str, filePath:str, loadCfg=True):
    data = {}
    fileType = fileName.split(".")[-1]
    bare_name = ".".join(fileName.split(".")[:-1])
    cfg_name = bare_name + "_cfg.yaml"
    if fileType in ["json", "JSON"]:
        with open(filePath + fileName, 'r') as datafile:
            data = json.load(datafile, default=_jsonDefaultRules)
    elif fileType in ["h5", "H5", "hdf5", "HDF5"]:
        with h5py.File(filePath + fileName, "r") as f:
            for s in f:
                data[s] = f[s][()]
    elif fileType in ["yaml", "yml", "YAML"]:
        with open(filePath + fileName, "r") as datafile:
            data = yaml.load(datafile)
    else:
        raise NotImplementedError(f"Don't know what to do with file type {fileType}, try 'json', 'h5', or 'yml'")

    if loadCfg:
        try:
            cfg = load_config(cfg_name, filePath)
            return data, cfg
        except FileNotFoundError as e:
            print(e)
            return data

    return data


def save_config(fileName, filePath, cfg: Dict):
    cfg = to_yaml_friendly(cfg)
    with open(filePath+ fileName, 'w') as file:
        yaml.dump(cfg, file)

def load_config(fileName, filePath):
    with open(filePath + fileName, 'r') as file:
        cfg = yaml.safe_load(file)
    return cfg


def _jsonDefaultRules(obj):
    if type(obj).__module__ == np.__name__:
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        else:
            return obj.item()
    raise TypeError('Unknown type:', type(obj))

def transferFileToRemote(fileName, localPath, remotePath, remoteIP="192.168.2.2", remoteUser="hatlab", removeLocalData=False):
    command = f"scp {localPath+fileName} {remoteUser}@{remoteIP}:/" + remotePath
    subprocess.call(command, shell=True)
    if removeLocalData:
        rm_cmd = f"rm {localPath+fileName} "
        subprocess.call(rm_cmd, shell=True)

def saveDataToRemote(data:Dict, fileName, remotePath, remoteIP="192.168.2.2", remoteUser="hatlab"):
    tempLocalDir = HOMEPATH + "/jupyter_notebooks/data/"
    saveData(data, fileName, tempLocalDir)
    transferFileToRemote(fileName, tempLocalDir, remotePath, remoteIP, remoteUser, removeLocalData=True)

def convertAndSaveFullData(Ibuff, Qbuff, fileName, filePath, config, saveConfig=False, **kwData):
    dataDict = {}
    dataDict["I_rot"] = np.array(Ibuff).reshape((-1, config["reps"])).T
    dataDict["Q_rot"] = np.array(Qbuff).reshape((-1, config["reps"])).T
    conflict_keys = set(kwData).intersection(dataDict)
    if len(conflict_keys) != 0:
        raise NameError(f"kwData has entries that are conflict with preserved entries {conflict_keys}")
    dataDict.update(kwData)
    saveData(dataDict, fileName, filePath)
    if saveConfig:
        saveData(config, fileName + "_config", filePath)

def saveFullDataToRemote(Ibuff, Qbuff, fileName, remotePath, config, remoteIP="192.168.2.2", remoteUser="hatlab", saveConfig=False, **kwData):
    tempLocalDir = HOMEPATH + "/jupyter_notebooks/data/"
    convertAndSaveFullData(Ibuff, Qbuff, fileName, tempLocalDir, config, saveConfig, **kwData)
    transferFileToRemote(fileName, tempLocalDir, remotePath, remoteIP, remoteUser, removeLocalData=True)
    if saveConfig:
        saveDataToRemote(config, fileName + "_config", remotePath, remoteIP, remoteUser)
        transferFileToRemote(fileName + "_config", tempLocalDir, remotePath, remoteIP, remoteUser, removeLocalData=True)
    

if __name__ == "__main__":
    data1 = np.linspace(1,100,1000)
    data2 = np.linspace(2,200,1000)
    data = dict(data1=data1, data2=data2)
    config={"res_ch":7, # --Fixed
        "relax_delay":0, # --Fixed
        "res_phase":0, # --Fixed
        "pulse_style": "const", # --Fixed
        "length":100, # [Clock ticks]        
        "readout_length":200, # [Clock ticks]
        "pulse_gain":0, # [DAC units]
        "pulse_freq": 100, # [MHz]
        "adc_trig_offset": 100, # [Clock ticks]
        "reps":50, 
        # New variables
        "expts": 20,
        "start":0, # [DAC units]
        "step":100 # [DAC units]
       }
    
    

    fileName = "testData_Full"
    # localPath = "./data/"
    remotePath = "C:/Users/hatla/Downloads"

    # saveData(data, localPath, fileName)
    # transferFileToRemote(fileName, localPath, remotePath, removeLocalData=True)
    saveFullDataToRemote(data1, data2, fileName, remotePath, config, aa=11)


