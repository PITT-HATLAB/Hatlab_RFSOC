from typing import List, Tuple, Dict, Sequence, Union, Any, Iterator, Optional, TypeVar

import numpy as np

from plottr.data.datadict import DataDict, DataDictBase


class QickDataDict(DataDict):
    """
    Subclass of plottr.DataDict class for keeping data from "QickProgram"s
    """
    def __init__(self, ro_chs, inner_sweeps:DataDictBase=None, outer_sweeps:DataDictBase=None):
        """
        initialize the DataDict class with sweep axes.
        :param ro_chs:
        :param inner_sweeps: DataDict that contains the axes of inner sweeps. In the initialization, the DataDict can
            just provide the names of inner sweeps axes and their units, the axes values can be empty. If the inner
            sweep value does not change for each outer sweep, the values of each inner sweep axes can also be provided,
            in which case the order of items in the dict has to follow first->last : innermost_sweep-> outermost_sweep.
        :param outer_sweeps: ataDict that contains the axes of outer sweeps. Again, axes values can be empty.
        """
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
            dd[k] = {"unit": v.get("unit"), "__list__": v.get("values")} # metadata __list__ for easier access to the sweep variables

        for k, v in inner_sweeps.items():
            dd[k] = {"unit": v.get("unit"), "__list__": v.get("values")} # metadata __list__ for easier access to the sweep variables

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

    def add_data(self, avg_i, avg_q, buf_i=None, buf_q=None,
                 inner_sweeps:Union[DataDict, DataDictBase, Dict]=None, **outer_sweeps) -> None:
        """
        Function for adding data to DataDict after each qick tproc inner sweep.

        :param avg_i: averaged I data returned from qick.RAveragerProgram.acquire()
            (or other QickPrograms that uses the same data shape: (ro_ch, msmts, expts))
        :param avg_q: averaged Q data returned from qick.RAveragerProgram.acquire()
            (or other QickPrograms that uses the same data shape: (ro_ch, msmts, expts)
        :param buf_i: all the I data points measured in qick run.
            shape: (n_ro, tot_reps, msmts_per_rep), where the order of points in the last dimension follows:(m0_exp1, m1_exp1, m0_exp2...)
        :param buf_q: all the Q data points measured in qick run.
            shape: (n_ro, tot_reps, msmts_per_rep), where the order of points in the last dimension follows:(m0_exp1, m1_exp1, m0_exp2...)
        :param inner_sweeps: Dict or DataDict that contains the keys and values of each qick inner sweep. The order has
            to be first->last : innermost_sweep-> outermost_sweep. When the inner sweep values change for each new outer
            sweep value, the inner sweep values can be re-specified when each time we add data, otherwise, the values
            provided in initialize will be used.
        :param outer_sweeps: kwargs for the new outer sweep values used in this data acquisition run.

        :return:
        """
        new_data = {}
        msmt_per_exp = avg_i.shape[-2]
        reps = 1 if buf_i is None else buf_i.shape[-2]
        if inner_sweeps is None:
            inner_sweeps = self.inner_sweeps
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
    from plottr.apps.autoplot import autoplot

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


    outer_sweeps = DataDictBase(amp={"unit": "dBm", "values": np.linspace(-20,-5, 16)},
                                freq={"unit": "MHz", "values": np.linspace(1000, 1005, 6)}
                                )

    qdd = QickDataDict(ro_chs, inner_sweeps, outer_sweeps)
    # qdd.add_data(x_pts, avgi, avgq)
    qdd.add_data(inner_sweeps, avgi, avgq, bufi, bufq, amp=1, freq=3)
    qdd.add_data(inner_sweeps, avgi, avgq, bufi, bufq, amp=1, freq=4)
    qdd.add_data(inner_sweeps, avgi, avgq, bufi, bufq, amp=2, freq=3)
    qdd.add_data(inner_sweeps, avgi, avgq, bufi, bufq, amp=2, freq=4)

    # the automatic griding in plottr doesn't work well in this complicated multidimensional sweep data.
    # We have to manually set the grid in the app.
    ap = autoplot(qdd)

