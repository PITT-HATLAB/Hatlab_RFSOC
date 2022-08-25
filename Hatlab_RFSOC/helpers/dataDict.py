from typing import List, Dict, Union
import numpy as np

from plottr.data.datadict import DataDict, DataDictBase
from plottr.data.datadict_storage import datadict_from_hdf5


class QickDataDict(DataDict):
    """
    Subclass of plottr.DataDict class for keeping data from "QickProgram"s
    """

    def __init__(self, ro_chs, inner_sweeps: DataDictBase = None, outer_sweeps: DataDictBase = None):
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
            dd[k] = {"unit": v.get("unit"),
                     "__list__": v.get("values")}  # metadata __list__ for easier access to the sweep variables

        for k, v in inner_sweeps.items():
            dd[k] = {"unit": v.get("unit"),
                     "__list__": v.get("values")}  # metadata __list__ for easier access to the sweep variables

        for ch in ro_chs:
            dd[f"avg_iq_{ch}"] = {
                "axes": [*list(outer_sweeps.keys())[::-1], "reps", *list(inner_sweeps.keys())[::-1], "msmts"],
                # in the order of outer to inner axes
                "unit": "a.u."
            }
            dd[f"buf_iq_{ch}"] = {
                "axes": [*list(outer_sweeps.keys())[::-1], "reps", *list(inner_sweeps.keys())[::-1], "msmts"],
                "unit": "a.u.",
            }

        super().__init__(**dd)

    def add_data(self, avg_i, avg_q, buf_i=None, buf_q=None,
                 inner_sweeps: Union[DataDict, DataDictBase, Dict] = None, **outer_sweeps) -> None:
        """
        Function for adding data to DataDict after each qick tproc inner sweep.

        :param avg_i: averaged I data returned from qick.RAveragerProgram.acquire()
            (or other QickPrograms that uses the same data shape: (ro_ch, msmts, expts))
        :param avg_q: averaged Q data returned from qick.RAveragerProgram.acquire()
            (or other QickPrograms that uses the same data shape: (ro_ch, msmts, expts)
        :param buf_i: all the I data points measured in qick run.
            shape: (ro_ch, tot_reps, msmts_per_rep), where the order of points in the last dimension follows:(m0_exp1, m1_exp1, m0_exp2...)
        :param buf_q: all the Q data points measured in qick run.
            shape: (ro_ch, tot_reps, msmts_per_rep), where the order of points in the last dimension follows:(m0_exp1, m1_exp1, m0_exp2...)
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
        flatten_inner = flattenSweepDict(inner_sweeps)  # assume inner sweeps have a square shape
        expts = len(list(flatten_inner.values())[0])  # total inner sweep points

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


def flattenSweepDict(sweeps: Union[DataDictBase, Dict]):
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


class DataFromQDDH5:
    def __init__(self, ddh5_path):
        self.datadict = datadict_from_hdf5(ddh5_path)
        self.avg_iq = {}
        self.buf_iq = {}
        self.axes = {}
        self.ro_chs = []
        self.reps = len(set(self.datadict["reps"]["values"]))
        self.axes_names = []
        self.datashape = []

        for k, v in self.datadict.items():
            if "avg_iq" in k:
                rch = k.replace("avg_iq_", "")
                self.avg_iq[rch] = self._reshape_original_data(v)
                self.ro_chs.append(rch)
            if "buf_iq" in k:
                rch = k.replace("buf_iq_", "")
                self.buf_iq[rch] = self._reshape_original_data(v)

        self._get_axes_values()

        print("data_shape: ", self.datashape)
        print("axes: ", self.axes_names)

    def _reshape_original_data(self, data):
        rep_idx = data["axes"].index("reps")
        data_shape = []
        if self.axes_names == []:
            self.axes_names = data["axes"]
        for ax in data["axes"]:
            try:  # assume all the sweep axes have metadata "__list__"
                ax_val = self.datadict.meta_val("list", ax)
                data_shape.append(len(ax_val))
            except KeyError:
                pass
        data_shape.insert(rep_idx, self.reps)

        data_r = np.array(data["values"]).reshape(*data_shape, -1)
        self.datashape = list(data_r.shape)

        return data_r

    def _get_axes_values(self):
        for ax in self.axes_names:
            if ax == "reps":
                self.axes[ax] = {"unit": "n", "value": np.arange(self.reps)}
            elif ax == "msmts":
                self.axes[ax] = {"unit": "n", "value": np.arange(self.datashape[-1])}
            else:  # assume all the sweep axes have metadata "__list__"
                ax_val = self.datadict.meta_val("list", ax)
                self.axes[ax] = {"unit": self.datadict[ax].get("unit"), "value": ax_val}

    def reorder_data(self, axis_order: List[str] = None, flatten_sweep=False, mute=False):
        if axis_order is None:
            an_ = self.axes_names.copy()
            an_.insert(0, an_.pop(an_.index("reps")))
            axis_order = an_

        new_idx_order = list(map(self.axes_names.index, axis_order))

        self.axes_names = axis_order
        axes_ = {k: self.axes[k] for k in axis_order}
        self.axes = axes_
        ds_ = np.array(self.datashape)[new_idx_order].tolist()
        self.datashape = ds_

        rep_idx = self.axes_names.index("reps")

        def reshape_(data):
            d = data.transpose(*new_idx_order)
            if flatten_sweep:
                d = d.reshape(*self.datashape[:rep_idx + 1], -1)
            return d

        for k, v in self.avg_iq.items():
            self.avg_iq[k] = reshape_(v)
        for k, v in self.buf_iq.items():
            self.buf_iq[k] = reshape_(v)

        if not mute:
            print("data_shape: ", self.datashape)
            print("axes: ", self.axes_names)


if __name__ == "__main__":
    from plottr.apps.autoplot import autoplot

    ro_chs = [0, 1]
    n_msmts = 2
    reps = 3

    # make fake data
    inner_sweeps = DataDictBase(length={"unit": "ns", "values": np.linspace(0, 100, 11)},
                                phase={"unit": "deg", "values": np.linspace(0, 90, 2)}
                                )

    x1_pts = inner_sweeps["length"]["values"]
    x2_pts = inner_sweeps["phase"]["values"]

    avgi = np.zeros((len(ro_chs), n_msmts, len(x1_pts) * len(x2_pts)))
    avgq = np.zeros((len(ro_chs), n_msmts, len(x1_pts) * len(x2_pts)))
    bufi = np.zeros((len(ro_chs), reps, n_msmts * len(x1_pts) * len(x2_pts)))
    bufq = np.zeros((len(ro_chs), reps, n_msmts * len(x1_pts) * len(x2_pts)))

    for i, ch in enumerate(ro_chs):
        for m in range(n_msmts):
            avgi[i, m] = (flattenSweepDict(inner_sweeps)["length"] + flattenSweepDict(inner_sweeps)["phase"]) * (
                        m + 1) + i
            avgq[i, m] = -(flattenSweepDict(inner_sweeps)["length"] + flattenSweepDict(inner_sweeps)["phase"]) * (
                        m + 1) + i

        bufi[i] = avgi[i].transpose().flatten() + (np.random.rand(reps, n_msmts * len(x1_pts) * len(x2_pts)) - 0.5) * 10
        bufq[i] = avgq[i].transpose().flatten() + (np.random.rand(reps, n_msmts * len(x1_pts) * len(x2_pts)) - 0.5) * 10

    outer_sweeps = DataDictBase(amp={"unit": "dBm", "values": np.linspace(-20, -5, 16)},
                                freq={"unit": "MHz", "values": np.linspace(1000, 1005, 6)}
                                )

    qdd = QickDataDict(ro_chs, inner_sweeps, outer_sweeps)
    # qdd.add_data(x_pts, avgi, avgq)
    qdd.add_data(avgi, avgq, bufi, bufq, inner_sweeps, amp=1, freq=3)
    qdd.add_data(avgi, avgq, bufi, bufq, inner_sweeps, amp=1, freq=4)
    qdd.add_data(avgi, avgq, bufi, bufq, inner_sweeps, amp=2, freq=3)
    qdd.add_data(avgi, avgq, bufi, bufq, inner_sweeps, amp=2, freq=4)

    # the automatic griding in plottr doesn't work well in this complicated multidimensional sweep data.
    # We have to manually set the grid in the app.
    ap = autoplot(qdd)
