from plottr.data.datadict import DataDict, MeshgridDataDict, DataDictBase
from plottr.data.datadict import datadict_to_meshgrid
from plottr.data.datadict_storage import datadict_to_hdf5, DDH5Writer
import h5py
import numpy as np
from plottr.apps.autoplot import autoplot

from Hatlab_DataProcessing.data_siving import hatDDH5Writer

import warnings
import copy as cp
import re

import numpy as np
from functools import reduce
from typing import List, Tuple, Dict, Sequence, Union, Any, Iterator, Optional, TypeVar

from plottr.utils import num, misc

class QickDataDict(DataDict):
    def __init__(self, ro_chs, outer_sweeps:DataDictBase=None):
        if outer_sweeps is None:
            outer_sweeps = {}

        self.ro_chs = ro_chs
        self.sweeps_axes = outer_sweeps

        dd = {
            "msmts": {},
            "expts": {},
            "reps": {},
        }

        for k, v in outer_sweeps.items():
            dd[k] = {"unit": v.get("unit"), "__list__": v["values"]} # __list__ for easier access to the sweep variables

        for ch in ro_chs:
            dd[f"avg_iq_{ch}"] = {
                "axes": [*list(outer_sweeps.keys()), "reps", "expts", "msmts"],
                "unit": "a.u."
            }
            dd[f"buf_iq_{ch}"] = {
                "axes": [*list(outer_sweeps.keys()), "reps", "expts", "msmts"],
                "unit": "a.u.",
            }

        super().__init__(**dd)

    def add_data(self, x_pts, avg_i, avg_q, buf_i=None, buf_q=None, **outer_sweeps) -> None:
        """
        todo: explain the shape of each input here
        todo: split inner sweeps
        :param x_pts:
        :param avg_i:
        :param avg_q:
        :param buf_i:
        :param buf_q:
        :return:
        """
        new_data = {}
        msmt_per_exp = avg_i.shape[-2]
        reps = 1 if buf_i is None else buf_i.shape[-2]
        expts = len(x_pts)

        new_data["msmts"] = np.tile(range(msmt_per_exp), expts * reps)

        for i, ch in enumerate(self.ro_chs):
            new_data[f"avg_iq_{ch}"] = np.tile((avg_i[i] + 1j * avg_q[i]).transpose().flatten(), reps)
            if buf_i is not None:
                new_data[f"buf_iq_{ch}"] = (buf_i[i] + 1j * buf_q[i]).flatten()
            else:
                new_data[f"buf_iq_{ch}"] = np.zeros(msmt_per_exp * expts)

        new_data["reps"] = np.repeat(np.arange(reps), msmt_per_exp * expts)
        new_data["expts"] = np.tile(np.repeat(x_pts, msmt_per_exp), reps)

        for k, v in outer_sweeps.items():
            new_data[k] = v

        super().add_data(**new_data)


if __name__ == "__main__":
    ro_chs = [0,1]
    n_msmts = 2
    reps = 200

    # make fake data
    x_pts = np.arange(50)
    avgi = np.zeros((len(ro_chs), n_msmts, len(x_pts)))
    avgq = np.zeros((len(ro_chs), n_msmts, len(x_pts)))
    bufi = np.zeros((len(ro_chs), reps, n_msmts * len(x_pts)))
    bufq = np.zeros((len(ro_chs), reps, n_msmts * len(x_pts)))

    for i, ch in enumerate(ro_chs):
        for m in range(n_msmts):
            avgi[i, m] = x_pts * (m+1) + i
            avgq[i, m] = -x_pts * (m + 1) +i

        bufi[i] = avgi[i].transpose().flatten() + (np.random.rand(reps, n_msmts * len(x_pts))-0.5)*10
        bufq[i] = avgq[i].transpose().flatten() + (np.random.rand(reps, n_msmts * len(x_pts))-0.5)*10



#-------------------------------
    """
    sweep_axes = {}
    dd = {
        "msmts": {},
        "expts": {},
        "reps": {},
    }

    dd.update(sweep_axes)

    for ch in ro_chs:
        dd.update({
            f"avg_iq_{ch}": {
                "axes": [*list(sweep_axes.keys()), "msmts", "expts"],
                "unit": "a.u."
            },
            f"buf_iq_{ch}": {
                "axes": [*list(sweep_axes.keys()), "reps", "msmts", "expts"],
                "unit": "a.u.",
            }
        })

    def transform_new_data( x_pts, avg_i, avg_q, buf_i=None, buf_q=None) :
        new_data = {}
        msmt_per_exp = avg_i.shape[-2]
        reps = 1 if buf_i is None else buf_i.shape[-2]
        expts = len(x_pts)
        for i, ch in enumerate(ro_chs):
            new_data[f"avg_iq_{ch}"] = np.tile((avg_i[i] + 1j * avg_q[i]).flatten(), reps)
            if buf_i is not None:
                new_data[f"buf_iq_{ch}"] = (buf_i[i] + 1j * buf_q[i]).reshape((reps, -1, msmt_per_exp)).transpose(0,2,1).flatten()
                new_data[f"reps"] = np.repeat(np.arange(reps), msmt_per_exp * expts)
            else:
                new_data[f"buf_iq_{ch}"] = np.zeros(msmt_per_exp * expts)

        new_data["expts"] = np.tile(x_pts, msmt_per_exp * reps)
        new_data["msmts"] = np.repeat(range(msmt_per_exp), expts * reps)

        return  new_data
        
    qdd = DataDict(**dd)
    qdd.add_data(**transform_new_data(x_pts, avgi, avgq))
    """
# ----------------------------------------------------------------------

    outer_sweeps = DataDictBase(amp={"unit": "dBm", "values": np.linspace(-20,-5, 16) },
                                freq={"unit": "MHz", "values": np.linspace(1000, 1005, 6)}
                                )


    qdd = QickDataDict(ro_chs, outer_sweeps)
    # qdd.add_data(x_pts, avgi, avgq)
    qdd.add_data(x_pts, avgi, avgq, bufi, bufq, amp=1, freq=3)
    qdd.add_data(x_pts, avgi, avgq, bufi, bufq, amp=1, freq=4)
    qdd.add_data(x_pts, avgi, avgq, bufi, bufq, amp=1, freq=5)


    ap = autoplot(qdd)

