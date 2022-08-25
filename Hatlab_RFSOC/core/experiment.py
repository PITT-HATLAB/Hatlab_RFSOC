from typing import Dict, List, Union, Callable, Type

from plottr.data.datadict import DataDict, DataDictBase
from qick.qick_asm import QickProgram

from Hatlab_RFSOC.proxy import getSocProxy
from Hatlab_RFSOC.helpers import QickDataDict
from Hatlab_DataProcessing.data_saving import HatDDH5Writer, DummyWriter


class Experiment():
    def __init__(self, program: Type[QickProgram], cfg: Dict, info: Dict,
                 inner_sweeps: DataDictBase = None, outer_sweeps: DataDictBase = None,
                 data_base_dir=None, data_folder_ame: str = None, data_file_name: str = None):
        """

        :param program:
        :param cfg:
        :param info:
        :param inner_sweeps: DataDict of inner sweeps. By default, always iterate from the first item to the last item.
        :param outer_sweeps: DataDict of outer sweeps. By default, always iterate from the first item to the last item.
        :param data_base_dir:
        :param data_folder_ame:
        :param data_file_name:
        :return:
        """
        self.PyroServer = info["PyroServer"]
        self.soc, self.soccfg = getSocProxy(self.PyroServer)
        self.program = program
        self.inner_sweeps = inner_sweeps
        self.outer_sweeps = outer_sweeps
        self.cfg = cfg


        self.qdd = QickDataDict(self.cfg["ro_chs"], self.inner_sweeps, self.outer_sweeps)
        self.data_base_dir = data_base_dir
        self.data_folder_name = data_folder_ame
        self.data_file_name = data_file_name
        self.ddw = None

    def run(self, save_data=True, save_buf=False, readouts_per_experiment=1, save_experiments=None,
            new_inner: Union[DataDictBase, Dict] = None, **outer_vals):
        """

        :param save_data:
        :param save_buf: if true, save the IQ buffer data (all data points)
        :param readouts_per_experiment:
        :param save_experiments:
        :param new_inner:
        :param outer_vals:
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

        # run program (and save data)
        with ddw as dw:
            dw.save_config(self.cfg)
            self.prog = self.program(self.soccfg, self.cfg)
            x_pts, avgi, avgq = self.prog.acquire(self.soc, load_pulses=True, progress=True, debug=False,
                                             readouts_per_experiment=readouts_per_experiment,
                                             save_experiments=save_experiments)
            if save_buf:
                dw.add_data(avg_i=avgi, avg_q=avgq, buf_i=self.prog.di_buf_p, buf_q=self.prog.dq_buf_p,
                            inner_sweeps=inner_sweeps, **outer_vals)
                return x_pts, avgi, avgq, self.prog.di_buf_p, self.prog.dq_buf_p
            else:
                dw.add_data(avg_i=avgi, avg_q=avgq, inner_sweeps=inner_sweeps, **outer_vals)
                return x_pts, avgi, avgq


    def change_data_file(self, data_folder_name: str = None, data_file_name: str = None):
        self.ddw = HatDDH5Writer(self.qdd, self.data_base_dir, data_folder_name, data_file_name)
