import warnings
from typing import List, Dict, Union
import tempfile
import numpy as np
import h5py
import yaml

from plottr.data.datadict import DataDict, DataDictBase
from plottr.data.datadict_storage import set_attr as dd_set_attr
from Hatlab_DataProcessing.data_saving import datadict_from_hdf5, HatDDH5Writer
from Hatlab_RFSOC.helpers.yaml_editor import to_yaml_friendly

def add_axis_meta(dd:Union[DataDictBase, Dict], ax_name: str, ax_value):
    """
    add the values of a sweep axis as metadata (hdf5 header attribute) for easier access in data loading
    :param dd: datadict
    :param ax_name: name of the sweep axis
    :param ax_value: values of the sweep axis
    :return:
    """
    # check if the header attribute is oversize or not
    tf = tempfile.TemporaryFile()
    try:
        dd_set_attr(h5py.File(tf, "w"), ax_name, ax_value)
        header_settable = True
    except RuntimeError:
        header_settable = False
    finally:
        tf.close()

    # if oversize, try to rewrite the value array as an eval np.arange string
    if not header_settable:
        # check if the value can be represented as a np.arange expression
        steps = ax_value[1:] - ax_value[:-1]
        if len(set(steps)) > 1:
            if (np.max(steps) - np.min(steps))/np.mean(steps) < 1e-8:
                warnings.warn(f"there seems to be a non-unique step in the sweep array, but the difference seems to be"
                              f"small. The averaged step value {np.average(steps)} is used for variable {ax_name}.")
                step = np.average(steps)
            else:
                raise NotImplementedError(f"can't write {ax_name} with value {ax_value} as a np.arrange expression")
        else:
            step = steps[0]
        newval = f"eval__np.arange({ax_value[0]}, {ax_value[-1] + step }, {step})" # we will decode this in data loading
    else:
        newval = ax_value

    dd[f"__val_{ax_name}__"] = newval


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

        dd = {"msmts": {}}
        add_axis_meta(dd, "msmts", None)

        for k, v in outer_sweeps.items():
            dd[k] = {"unit": v.get("unit")}
            add_axis_meta(dd, k, v.get("values"))

        dd["reps"] = {}
        add_axis_meta(dd, "reps", None)

        for k, v in inner_sweeps.items():
            dd[k] = {"unit": v.get("unit")}
            add_axis_meta(dd, k, v.get("values"))

        dd["soft_reps"] = {}
        add_axis_meta(dd, "soft_reps", None)

        for ch in ro_chs:
            dd[f"avg_iq_{ch}"] = {
                "axes": ["soft_reps", *list(outer_sweeps.keys())[::-1], "reps", *list(inner_sweeps.keys())[::-1],
                         "msmts"],
                # in the order of outer to inner axes
                "unit": "a.u.",
                "__isdata__": True
            }
            dd[f"buf_iq_{ch}"] = {
                "axes": ["soft_reps", *list(outer_sweeps.keys())[::-1], "reps", *list(inner_sweeps.keys())[::-1],
                         "msmts"],
                # in the order of outer to inner axes
                "unit": "a.u.",
                "__isdata__": True
            }
        super().__init__(**dd)

    def add_data(self, avg_i, avg_q, buf_i=None, buf_q=None,
                 inner_sweeps: Union[DataDict, DataDictBase, Dict] = None, soft_rep=0, **outer_sweeps) -> None:
        """
        Function for adding data to DataDict after each qick tproc inner sweep.

        At this point, there no requirement on the order of soft_rep and outer sweeps, e.g. you can sweep over all
        outer sweeps in whatever order, then do soft repeat (repeat in python), or, you can do soft repeat of the qick
        inner sweeps first, then sweep the outer parameters. The data-axes mapping relation will always be correct.

        BUT, to make the data extraction methods in"DataFromQDDH5" work correctly, it is recommended to add data in
        the order of outer_sweeps dict first (first key->last key), then add soft repeats.

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
        :param soft_rep: soft repeat index
        :param outer_sweeps: kwargs for the new outer sweep values used in this data acquisition run.

        :return:
        """
        new_data = {}
        msmt_per_exp = avg_i.shape[-2]
        reps = 1 if buf_i is None else buf_i.shape[-2]
        if inner_sweeps is None:
            inner_sweeps = self.inner_sweeps
        flatten_inner = flatten_sweep_dict(inner_sweeps)  # assume inner sweeps have a square shape

        expts = len(list(flatten_inner.values())[0]) if flatten_inner != {} else 1  # total inner sweep points

        # add msmt index data
        new_data["msmts"] = np.tile(range(msmt_per_exp), expts * reps)
        add_axis_meta(self, "msmts", np.arange(msmt_per_exp))


        # add iq data
        for i, ch in enumerate(self.ro_chs):
            new_data[f"avg_iq_{ch}"] = np.tile((avg_i[i] + 1j * avg_q[i]).transpose().flatten(), reps)
            if buf_i is not None:
                new_data[f"buf_iq_{ch}"] = (buf_i[i] + 1j * buf_q[i]).flatten()
            else:
                new_data[f"buf_iq_{ch}"] = np.zeros(msmt_per_exp * expts)

        # add qick repeat index data
        new_data["reps"] = np.repeat(np.arange(reps), msmt_per_exp * expts)
        add_axis_meta(self, "reps", np.arange(reps))

        # add qick inner sweep data
        # for k, v in flatten_inner.items():
        #     new_data[k] = np.tile(np.repeat(v, msmt_per_exp), reps)

        for ki, vi in flatten_inner.items():
            new_data[ki] = np.tile(np.repeat(vi, msmt_per_exp), reps)
            for ko, vo in outer_sweeps.items():
                new_data[ki] = np.tile(new_data[ki], len(vo))

        # add outer sweep data
        for k, v in outer_sweeps.items():
            new_data[k] = np.repeat([v], msmt_per_exp * expts * reps)

        # add soft repeat index data
        new_data["soft_reps"] = np.repeat([soft_rep], msmt_per_exp * expts * reps)
        add_axis_meta(self, "soft_reps",  np.arange(soft_rep + 1))

        super().add_data(**new_data)


def flatten_sweep_dict(sweeps: Union[DataDictBase, Dict]):
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

def dict_to_datadict(d:Dict):
    """
    converts a normal python dict to Data dict with "values" keys,
    (for compatibility with functions that require a DataDict input)
    :param d:
    :return:
    """
    dd = DataDict()
    for k, v in d.items():
        dd[k] = {"values":v}
    return dd


def quick_save(filepath, filename, avg_i, avg_q, buf_i=None, buf_q=None, config=None, **sweep_params):
    """
    simply saves iq data points acquired from qick averager programs. Unlike the more complete data saving methods in
    "Experiment" class, here we only save the qick inner sweep axes without units.
    :param filepath: data file directory
    :param filename: data file name
    :param avg_i: avg_i data from qick averager programs
    :param avg_q: avg_q data from qick averager programs
    :param buf_i: buf_i data from qick averager programs
    :param buf_q: buf_q data from qick averager programs
    :param config: config dict
    :param sweep_params: dict of qick sweep parameters
    :return:
    """
    inner_sweeps = dict_to_datadict(sweep_params)
    qdd = QickDataDict(config["ro_chs"], inner_sweeps)
    ddw = HatDDH5Writer(qdd, filepath, filename=filename)

    with ddw as dw:
        if config is not None:
            dw.save_config(config)
        dw.add_data(inner_sweeps=inner_sweeps, avg_i=avg_i, avg_q=avg_q, buf_i=buf_i, buf_q=buf_q)


def _get_eval_meta(qdd, nema):
    val = qdd.meta_val(f"val_{nema}")
    if (type(val) == str) and (val[:6] == "eval__"):
        return eval(val[6:])
    else:
        return val


def get_config_info(datapath: str):
    """
    get the config and info dict from the data path
    :param datapath: 
    :return: 
    """
    filename = datapath.split("\\")[-1][:-5]
    filepath = "\\".join(datapath.split("\\")[:-1]) + "\\"
    config = yaml.safe_load(open(datapath[:-5] + "_cfg.yaml"))["config"]
    info = yaml.safe_load(open(datapath[:-5] + "_cfg.yaml"))["info"]

    return config, info 

class DataFromQDDH5:
    def __init__(self, ddh5_path, merge_reps=True, progress=False, fast_load=True):
        """
        load data from a DDH5 file that was created from a QickDataDict object. Adds the loaded data to dictionaries
        that are easy to use (avg_iq, buf_iq, axes). To ensure the correct order of axis values, the original data must
        be created in the order of (outer->inner): (soft_rep, outer_sweeps, reps, inner_sweeps, msmts)

        :param ddh5_path: path to the ddh5 file.
        :param merge_reps: when True, the soft_reps and reps (qick inner reps) will be merged into one axes. For avg_iq,
            the data from different soft repeat cycles will be averaged.
        :param progress: when True, show a progress bar for data loading.
        :param fast_load: when True, load the experiment data (avg_iq, buf_iq) only, and axes values will be loaded from
            metadata.
        """
        self.datadict = datadict_from_hdf5(ddh5_path, progress=progress, data_only=fast_load)
        self.avg_iq = {}
        self.buf_iq = {}
        self.axes = {}
        self.ro_chs = []
        self.reps = _get_eval_meta(self.datadict, "reps")[-1] + 1
        self.soft_reps = _get_eval_meta(self.datadict, "soft_reps")[-1] + 1
        self.total_reps = self.reps * self.soft_reps
        self.axes_names = []
        self.datashape = []

        # reshape original data based on the size of each sweep axes (including reps and msmts)
        for k, v in self.datadict.items():
            if "avg_iq" in k:
                # rch = k.replace("avg_iq_", "")
                rch=0
                self.avg_iq[rch] = self._reshape_original_data(v)
                self.ro_chs.append(rch)
            if "buf_iq" in k:
                # rch = k.replace("buf_iq_", "")
                self.buf_iq[rch] = self._reshape_original_data(v)

        if merge_reps:
            self._merge_reps()
        else:
            rep_idx = self.axes_names.index("reps")
            for k, v in self.avg_iq.items():
                self.avg_iq[k] = np.moveaxis(v, rep_idx, 0)[0]

        print("buffer data shape: ", self.datashape)
        print("buffer data axes: ", self.axes_names)

    def _reshape_original_data(self, data):
        """
        reshape original data based on the size of each sweep axes (including reps and msmts), and get the values of
        each sweep axes.

        :param data:
        :return:
        """
        data_shape = []
        if self.axes_names == []:
            self.axes_names = data["axes"]
        for ax in data["axes"][::-1]: 
            try:  
                # get values of sweep axes from metadata if saved in metadata
                ax_val = _get_eval_meta(self.datadict, ax)
                ax_dim = len(ax_val)

                if not isinstance(ax_val, np.ndarray):
                    if ax_val == "None": # sweep axis not saved in meta
                        ax_dim = -1
                        ax_val = None
                #     # assume the data were added in order of sweep axes, this should give us the right axes values
                #     if ax == "timestamp":
                #         print(self.datadict[ax]["values"], data_shape)
                #     last_layer_sweep_runs = np.product(data_shape)
                #     vals = np.array(self.datadict[ax]["values"])[:last_layer_sweep_runs] #TODO: wrong here
                #     ax_val = vals.reshape(-1, last_layer_sweep_runs)[:, 0]
                if ax not in self.axes:  # only need to add once
                    self.axes[ax] = {"unit": self.datadict[ax].get("unit"), "values": ax_val}
                data_shape.insert(0, ax_dim)
            except KeyError:
                pass

        data_r = np.array(data["values"]).reshape(*data_shape)
        self.datashape = list(data_r.shape)

        return data_r

    def _merge_reps(self):
        """
        merge the software repeats and the qick inner repeats into one axes.
        :return:
        """
        rep_idx = self.axes_names.index("reps")
        for k, v in self.avg_iq.items():
            v = np.moveaxis(v, rep_idx, 1)
            self.avg_iq[k] = np.average(v.reshape(-1, *v.shape[2:]), axis=0)
        for k, v in self.buf_iq.items():
            v = np.moveaxis(v, rep_idx, 1)
            self.buf_iq[k] = v.reshape(-1, *v.shape[2:])
        self.datashape = list(self.buf_iq[k].shape)

        self.axes_names.pop(rep_idx)
        self.axes_names[0] = "reps"

        _new_axes = {"reps": np.arange(self.total_reps)}
        for k in self.axes_names[1:]:
            _new_axes[k] = self.axes[k]
        self.axes = _new_axes


def save_state_pct(filepath, filename, state_pct, **sweep_params):
    data = {"state_pct": {"unit": "a.u."}}
    for k in sweep_params.keys():
        data[k] = {"axes": []}
    data["state_pct"]["axes"] = list(sweep_params.keys())

    dd = DataDict(**data)
    ddw = HatDDH5Writer(dd, filepath, filename=filename)
    with ddw as dw:
        dw.add_data(state_pct=state_pct, **sweep_params)


def save_data_raw(filepath, filename, avg_i, avg_q, buf_i=None, buf_q=None, config=None, info=None,
                  inner_sweeps: Union[DataDictBase, Dict] = None, outer_sweeps: Union[DataDictBase, Dict] = None):
    """
    simply saves iq data points acquired from qick averager programs. Unlike the more complete data saving methods in
    "Experiment" class, here we only save the qick inner sweep axes without units.
    :param filepath: data file directory
    :param filename: data file name
    :param avg_i: avg_i data from qick averager programs
    :param avg_q: avg_q data from qick averager programs
    :param buf_i: buf_i data from qick averager programs
    :param buf_q: buf_q data from qick averager programs
    :param config: config dict
    :param inner_sweep: DataDictBase or dict of qick sweep parameters
    :param outer_sweep: var of outer sweep parameters
    :return:
    """
    qdd = QickDataDict(config["ro_chs"], inner_sweeps, outer_sweeps)
    ddw = HatDDH5Writer(qdd, filepath, filename=filename)
    with ddw as dw:
        if config is not None:
            dw.save_config(to_yaml_friendly({"config": config, "info": info}))
        avg_i, avg_q = np.asarray(avg_i), np.asarray(avg_q)
        buf_i, buf_q = np.asarray(buf_i), np.asarray(buf_q)
        if outer_sweeps is not None:
            outer_dict = outer_sweeps.to_dict()
            avg_i = np.moveaxis(avg_i, len(outer_dict), 0)
            avg_q = np.moveaxis(avg_q, len(outer_dict), 0)
            buf_i = np.moveaxis(buf_i, len(outer_dict), 0)
            buf_q = np.moveaxis(buf_q, len(outer_dict), 0)
            dw.add_data(inner_sweeps=inner_sweeps, avg_i=avg_i, avg_q=avg_q, buf_i=buf_i, buf_q=buf_q, **outer_dict)
        else:
            dw.add_data(inner_sweeps=inner_sweeps, avg_i=avg_i, avg_q=avg_q, buf_i=buf_i, buf_q=buf_q)


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
            avgi[i, m] = (flatten_sweep_dict(inner_sweeps)["length"] + flatten_sweep_dict(inner_sweeps)["phase"]) * (
                        m + 1) + i
            avgq[i, m] = -(flatten_sweep_dict(inner_sweeps)["length"] + flatten_sweep_dict(inner_sweeps)["phase"]) * (
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
