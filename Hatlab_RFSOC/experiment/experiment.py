from typing import Dict, List, Union, Callable, Type

from qick.qick_asm import QickProgram

from Hatlab_RFSOC.proxy import getSocProxy
from Hatlab_RFSOC.data import QickDataDict
from Hatlab_RFSOC.helpers.yaml_editor import to_yaml_friendly
from Hatlab_DataProcessing.data_saving import HatDDH5Writer, DummyWriter

from plottr.data.datadict import DataDict, DataDictBase


class Experiment():
    def __init__(self, program: Type[QickProgram], cfg: Dict, info: Dict,
                 inner_sweeps: DataDictBase = None, outer_sweeps: DataDictBase = None,
                 data_base_dir=None, data_folder_name: str = None, data_file_name: str = None):
        """
        a general manager class for running experiments with qick and other instruments. Includes data saving methods
        that save data to "DDH5" files used in plottr.

        :param program: QickProgram class
        :param cfg: config dict
        :param info: info dict, must contain "PyroServer" (name of the pyro server)
        :param inner_sweeps: DataDict of qick inner sweeps. To ensure the correct data order, the user need to make sure
            that the qick program always iterate from the first item to the last item this dictionary.
        :param outer_sweeps: DataDict of outer sweeps, the values are usually parameters of instruments that not
            controlled by qick or parameters in qick config dict that are not convenient to be swept in the inner loop.
            To ensure the correct data order, the user need to make sure that
            the outer sweep always iterate from the first item to the last item in this dictionary.
        :param data_base_dir: base directory for saving data files
        :param data_folder_name: data folder name in the base_dir, by default, uses the current date as folder name
        :param data_file_name: data file name.
        :return:
        """
        self.PyroServer = info["PyroServer"]
        self.info = info
        self.soc, self.soccfg = getSocProxy(self.PyroServer)
        self.program = program
        self.inner_sweeps = inner_sweeps
        self.outer_sweeps = outer_sweeps
        self.cfg = cfg

        self.qdd = QickDataDict(self.cfg["ro_chs"], self.inner_sweeps, self.outer_sweeps)
        self.data_base_dir = data_base_dir
        self.data_folder_name = data_folder_name
        self.data_file_name = data_file_name
        self.ddw = None

    def run(self, save_data=True, save_buf=False, readouts_per_experiment=None, save_experiments: List = None,
            new_inner: Union[DataDictBase, Dict] = None, soft_rep=0, inner_progress=True, **outer_vals):
        """
        run qick program and save data. By default, after each run, the new data will be appended to the same data file.

        :param save_data: if true, save experiment data to DDH5
        :param save_buf: if true, save the IQ buffer data (all data points)
        :param readouts_per_experiment: by default, this value is automatically in prog.measure()
        :param save_experiments: by default save all measurements.
        :param new_inner: the inner sweep dictionary can be updated in each run.
        :param soft_rep: index of soft repeat (average loop done in python)
        :param inner_progress: when True, show the progress bar fo the qick inner sweep
        :param outer_vals: the values of the outer sweep used in this run
        :return:
        """
        if new_inner is None:
            inner_sweeps = self.inner_sweeps
        else:
            inner_sweeps = new_inner

        if save_data:
            if self.ddw is None:
                self.ddw = HatDDH5Writer(self.qdd, self.data_base_dir, self.data_folder_name, self.data_file_name)
            ddw = self.ddw
        else:
            ddw = DummyWriter()

        self.prog = self.program(self.soccfg, self.cfg)
        x_pts, avgi, avgq = self.prog.acquire(self.soc, load_pulses=True, progress=inner_progress, debug=False,
                                              readouts_per_experiment=readouts_per_experiment,
                                              save_experiments=save_experiments)

        ## run program (and save data)
        if save_data:
            if ddw.inserted_rows == 0:
                ddw.__enter__()

            ddw.save_config(to_yaml_friendly({"config":self.cfg, "info": self.info}))

            if save_buf:
                ddw.add_data(avg_i=avgi, avg_q=avgq, buf_i=self.prog.di_buf_p, buf_q=self.prog.dq_buf_p,
                             inner_sweeps=inner_sweeps, soft_rep=soft_rep, **outer_vals)
                return x_pts, avgi, avgq, self.prog.di_buf_p, self.prog.dq_buf_p
            else:
                ddw.add_data(avg_i=avgi, avg_q=avgq, inner_sweeps=inner_sweeps, soft_rep=soft_rep, **outer_vals)
                return x_pts, avgi, avgq

        return x_pts, avgi, avgq, self.prog.di_buf_p, self.prog.dq_buf_p

    def change_data_file(self, data_folder_name: str = None, data_file_name: str = None):
        """
        change target file for data saving.

        :param data_folder_name:
        :param data_file_name:
        :return:
        """
        self.qdd = QickDataDict(self.cfg["ro_chs"], self.inner_sweeps, self.outer_sweeps)
        self.ddw = HatDDH5Writer(self.qdd, self.data_base_dir, data_folder_name, data_file_name)
