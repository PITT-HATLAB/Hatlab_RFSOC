import numpy as np
from Hatlab_RFSOC.helpers import plotData
from Hatlab_RFSOC.helpers.dataTransfer import saveData
from Hatlab_RFSOC.helpers.dataDict import QickDataDict
from Hatlab_RFSOC.helpers.pulseConfig import  declareMuxedGenAndReadout, add_tanh

def get_expt_pts(start, step, expts, **kw):
    return start + np.arange(expts) * step
