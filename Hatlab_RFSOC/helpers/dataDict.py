from plottr.data.datadict import DataDict, MeshgridDataDict, DataDictBase
from plottr.data.datadict import datadict_to_meshgrid
from plottr.data.datadict_storage import datadict_to_hdf5, DDH5Writer
import h5py
import numpy as np
from plottr.apps.autoplot import autoplot


import warnings
import copy as cp
import re

import numpy as np
from functools import reduce
from typing import List, Tuple, Dict, Sequence, Union, Any, Iterator, Optional, TypeVar

from plottr.utils import num, misc




class QickDataDict(DataDict):
    def __init__(self, ro_chs, inner_sweeps:DataDictBase=None, outer_sweeps:DataDictBase=None):
        if outer_sweeps is None:
            outer_sweeps = {}
        if inner_sweeps is None:
            inner_sweeps = {}

        self.ro_chs = ro_chs
        self.outer_sweeps = outer_sweeps
        self.inner_sweeps = inner_sweeps

        dd = {
            "msmts": {},
            "reps": {},
        }

        for k, v in outer_sweeps.items():
            dd[k] = {"unit": v.get("unit"), "__list__": v.get("values")} # __list__ for easier access to the sweep variables

        for k, v in inner_sweeps.items():
            dd[k] = {"unit": v.get("unit"), "__list__": v.get("values")} # __list__ for easier access to the sweep variables

        for ch in ro_chs:
            dd[f"avg_iq_{ch}"] = {
                "axes": [*list(outer_sweeps.keys())[::-1], "reps", *list(inner_sweeps.keys())[::-1], "msmts"], # in the order of outer to inner axes
                "unit": "a.u."
            }
            dd[f"buf_iq_{ch}"] = {
                "axes": [*list(outer_sweeps.keys())[::-1], "reps", *list(inner_sweeps.keys())[::-1], "msmts"],
                "unit": "a.u.",
            }

        super().__init__(**dd)

    def add_data(self, inner_sweeps, avg_i, avg_q, buf_i=None, buf_q=None, **outer_sweeps) -> None:
        """
        assume we add data to DataDict after each qick tproc inner sweep.
        todo: explain the shape of each input here
        todo: split expts to inner sweeps
        :param inner_sweeps:
        :param avg_i:
        :param avg_q:
        :param buf_i:
        :param buf_q:
        :return:
        """
        new_data = {}
        msmt_per_exp = avg_i.shape[-2]
        reps = 1 if buf_i is None else buf_i.shape[-2]
        flatten_inner = flattenSweepDict(inner_sweeps) # assume inner sweeps have a square shape
        expts = len(list(flatten_inner.values())[0]) # total inner sweep points

        new_data["msmts"] = np.tile(range(msmt_per_exp), expts * reps)

        for i, ch in enumerate(self.ro_chs):
            new_data[f"avg_iq_{ch}"] = np.tile((avg_i[i] + 1j * avg_q[i]).transpose().flatten(), reps)
            if buf_i is not None:
                new_data[f"buf_iq_{ch}"] = (buf_i[i] + 1j * buf_q[i]).flatten()
            else:
                new_data[f"buf_iq_{ch}"] = np.zeros(msmt_per_exp * expts)

        new_data["reps"] = np.repeat(np.arange(reps), msmt_per_exp * expts)

        for k, v in flatten_inner.items():
            new_data[k] = np.tile(np.repeat(v, msmt_per_exp), reps)

        for k, v in outer_sweeps.items():
            new_data[k] = np.repeat([v], msmt_per_exp * expts * reps)


        super().add_data(**new_data)


def flattenSweepDict(sweeps:Union[DataDictBase, Dict]):
    """
    Flatten a square sweep dictionary to 1d arrays.

    :param sweeps: dictionary of sweep variable arrays
    :return:
    """
    try:
        py_dict = sweeps.to_dict()
    except AttributeError:
        py_dict = sweeps

    flatten_sweeps = {}
    sweep_vals = map(np.ndarray.flatten, np.meshgrid(*py_dict.values()))
    for k in sweeps.keys():
        flatten_sweeps[k] = next(sweep_vals)
    return flatten_sweeps


if __name__ == "__main__":
    ro_chs = [0,1]
    n_msmts = 2
    reps = 3

    # make fake data

    inner_sweeps = DataDictBase(length={"unit": "ns", "values": np.linspace(0, 100, 11)},
                                phase={"unit": "deg", "values": np.linspace(0, 90, 2)}
                                )

    x1_pts = inner_sweeps["length"]["values"]
    x2_pts = inner_sweeps["phase"]["values"]

    avgi = np.zeros((len(ro_chs), n_msmts, len(x1_pts)*len(x2_pts)))
    avgq = np.zeros((len(ro_chs), n_msmts, len(x1_pts)*len(x2_pts)))
    bufi = np.zeros((len(ro_chs), reps, n_msmts * len(x1_pts)*len(x2_pts)))
    bufq = np.zeros((len(ro_chs), reps, n_msmts * len(x1_pts)*len(x2_pts)))

    for i, ch in enumerate(ro_chs):
        for m in range(n_msmts):
            avgi[i, m] = (flattenSweepDict(inner_sweeps)["length"] + flattenSweepDict(inner_sweeps)["phase"] )* (m+1) + i
            avgq[i, m] = -(flattenSweepDict(inner_sweeps)["length"] + flattenSweepDict(inner_sweeps)["phase"] ) * (m + 1) + i

        bufi[i] = avgi[i].transpose().flatten() + (np.random.rand(reps, n_msmts * len(x1_pts)*len(x2_pts))-0.5)*10
        bufq[i] = avgq[i].transpose().flatten() + (np.random.rand(reps, n_msmts * len(x1_pts)*len(x2_pts))-0.5)*10



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


    outer_sweeps = DataDictBase(amp={"unit": "dBm", "values": np.linspace(-20,-5, 16)},
                                freq={"unit": "MHz", "values": np.linspace(1000, 1005, 6)}
                                )


    qdd = QickDataDict(ro_chs, inner_sweeps, outer_sweeps)
    # qdd.add_data(x_pts, avgi, avgq)
    qdd.add_data(inner_sweeps, avgi, avgq, bufi, bufq, amp=1, freq=3)
    qdd.add_data(inner_sweeps, avgi, avgq, bufi, bufq, amp=1, freq=4)
    qdd.add_data(inner_sweeps, avgi, avgq, bufi, bufq, amp=2, freq=3)
    qdd.add_data(inner_sweeps, avgi, avgq, bufi, bufq, amp=2, freq=4)

    qddm = datadict_to_meshgrid(qdd, (2,2,3,2,11,2), [*list(outer_sweeps.keys())[::-1], "reps", *list(inner_sweeps.keys())[::-1], "msmts"])

    ap = autoplot(qddm)

