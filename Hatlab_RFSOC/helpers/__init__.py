import numpy as np
from Hatlab_RFSOC.helpers import plotData
from Hatlab_RFSOC.helpers.dataTransfer import saveData
from Hatlab_RFSOC.helpers.dataDict import QickDataDict, DataFromQDDH5
from Hatlab_RFSOC.helpers.pulseConfig import declareMuxedGenAndReadout, add_tanh, add_prepare_msmt


def get_expt_pts(start, step, expts, **kw):
    return start + np.arange(expts) * step


def get_sweep_vals(cfg: dict, var_name):
    return np.linspace(cfg[f"{var_name}_start"], cfg[f"{var_name}_stop"], cfg[f"{var_name}_expts"])
