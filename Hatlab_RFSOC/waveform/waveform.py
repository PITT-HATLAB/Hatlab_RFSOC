import warnings
from typing import List, Union, Type, Callable
import numpy as np
from qick.qick_asm import QickProgram
import matplotlib.pyplot as plt


NumType = Union[int, float]


class WaveformRegistry:
    _registry = {}

    @classmethod
    def register(cls, shape: str, waveform_cls: Type['Waveform']):
        cls._registry[shape] = waveform_cls

    @classmethod
    def create(cls, shape: str, *args, **kwargs) -> 'Waveform':
        for wave in cls._registry:
            if shape.lower() == wave.lower():
                shape = wave
        if shape not in cls._registry:
            raise ValueError(f"Waveform '{shape}' is not registered.")
        return cls._registry[shape](*args, **kwargs)

    @classmethod
    def available_waveforms(cls):
        return list(cls._registry.keys())


class Waveform:
    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        WaveformRegistry.register(cls.__name__, cls)

    def __init__(self, prog: QickProgram, gen_ch: Union[int, str], phase, maxv):
        self._set_channel_cfg(prog, gen_ch)
        self.maxv = self.soc_gencfg['maxv'] * self.soc_gencfg['maxv_scale'] if maxv is None else maxv
        self.phase = phase
        self.waveform = None

    @staticmethod
    def core(*args, **kwargs) -> np.ndarray:
        pass

    def _generate_waveform(self, *args, **kwargs):
        """
        Generates waveform based on the core function.
        Subclasses should implement this method to generate the waveform.
        """
        raise NotImplementedError("Subclasses must implement _generate_waveform().")

    def add_waveform(self, prog: QickProgram, name):
        idata = self._pad_waveform(np.real(self.waveform))
        qdata = self._pad_waveform(np.imag(self.waveform))

        if np.max(np.abs(idata)) > 32766 or np.max(np.abs(qdata)) > 32766:
            i_max, q_max = np.max(np.abs(idata)), np.max(np.abs(qdata))
            k = 32766 / np.max((i_max, q_max))
            idata *= k
            qdata *= k
            # warnings.warn("pulse amplitude exceeded maxv")
            print(f"pulse '{name}' amplitude exceeded maxv by {np.max((i_max, q_max)) - 32766}")

        prog.add_pulse(self.gen_ch, name, idata=idata.astype(int), qdata=qdata.astype(int))

    def _set_channel_cfg(self, prog: QickProgram, gen_ch: Union[int, str]):
        self.gen_ch = prog.cfg["gen_chs"][gen_ch]["ch"] if isinstance(gen_ch, str) else gen_ch
        self.soc_gencfg = prog.soccfg['gens'][self.gen_ch]
        self.samps_per_clk = self.soc_gencfg['samps_per_clk']
        self.fclk = self.soc_gencfg['f_fabric']
        self.sampling_rate = self.samps_per_clk * self.fclk

    def us_to_samps(self, length):
        """Convert length in physical units to register units."""
        return length * self.sampling_rate

    def _pad_waveform(self, waveform: np.ndarray) -> np.ndarray:
        """Pad waveform with zeros so that it's length is a multiple of samps_per_clk."""
        pad_len = (-len(waveform)) % self.samps_per_clk
        return np.pad(waveform, (0, pad_len))

    def _apply_padding(self, data: np.ndarray, padding: Union[float, List[float], None]) -> np.ndarray:
        """Pad waveform with zeros before and/or after."""
        if padding is None:
            padding = [0, 0]
        elif isinstance(padding, (int, float)):
            padding = [0, padding]

        padding_reg = np.ceil(self.us_to_samps(np.array(padding))).astype(int)
        return np.pad(data, (padding_reg[0], padding_reg[1]))

    def plot_waveform(self, ax=None, clock_cycle=False):
        """Plots the waveform."""
        fig, ax = plt.subplots() if ax is None else (ax.get_figure(), ax)
        if clock_cycle:
            t_list = np.arange(0, len(self.waveform)) / self.samps_per_clk
            ax.set_xlabel("gen_ch clock cycle")
        else:
            t_list = np.arange(0, len(self.waveform)) / (self.fclk * self.samps_per_clk)
            ax.set_xlabel("Time (us)")
        ax.set_ylabel("Amplitude")
        ax.grid()
        ax.plot(t_list, np.real(self.waveform), label="I")
        ax.plot(t_list, np.imag(self.waveform), label="Q")
        ax.plot(t_list, np.abs(self.waveform), label="mag", linestyle="dashed")
        ax.legend()


class Gaussian(Waveform):
    def __init__(self, prog, gen_ch, length, sigma, phase=0, maxv=None,
                 padding: Union[float, List[float], None] = None):
        super().__init__(prog, gen_ch, phase=phase, maxv=maxv)
        self.sigma_samps = self.us_to_samps(sigma)
        self.length_samps = self.us_to_samps(length)
        self.padding = padding
        self.waveform = self._generate_waveform(self.length_samps, self.sigma_samps)

    @staticmethod
    def core(length, sigma):
        """the definetion of Gaussian"""
        t = np.arange(length)
        y = np.exp(-(t - length / 2) ** 2 / sigma ** 2)
        return y - np.min(y)

    def _generate_waveform(self, *args, **kwargs):
        """
        apply the necessary modificaiton to the core function,
        generate in-phase (I) and quadrature (Q) components
        """
        waveform = self.maxv * self.core(*args, **kwargs)
        waveform_padded = self._apply_padding(waveform, self.padding)
        waveform_wphase = np.exp(1j * np.deg2rad(self.phase)) * waveform_padded
        return waveform_wphase


class TanhBox(Waveform):
    def __init__(self, prog, gen_ch, length, ramp_width, cut_offset=0.01, phase=0, maxv=None,
                 padding: Union[float, List[float], None] = None):
        super().__init__(prog, gen_ch, phase=phase, maxv=maxv)
        self.ramp_samps = self.us_to_samps(ramp_width)
        self.length_samps = self.us_to_samps(length)
        self.cut_offset = cut_offset
        self.padding = padding
        self.waveform = self._generate_waveform(self.length_samps, self.ramp_samps, self.cut_offset)

    @staticmethod
    def core(length, ramp_width, cut_offset):
        """
        Create a numpy array containing a smooth box pulse made of two tanh functions subtract from each other.

        :param length: number of points of the pulse
        :param ramp_width: number of points from cutOffset to 0.95 amplitude
        :param cut_offset: the initial offset to cut on the tanh Function
        :return:
        """
        t = np.arange(length)
        c0_, c1_ = np.arctanh(2 * cut_offset - 1), np.arctanh(2 * 0.95 - 1)
        k_ = (c1_ - c0_) / ramp_width
        y = (0.5 * (np.tanh(k_ * t + c0_) - np.tanh(k_ * (t - length) - c0_)) - cut_offset) / (1 - cut_offset)
        return y - np.min(y)

    def _generate_waveform(self, *args, **kwargs):
        """
        apply the necessary modificaiton to the core function,
        generate in-phase (I) and quadrature (Q) components
        """
        waveform = self.maxv * self.core(*args, **kwargs)
        waveform_padded = self._apply_padding(waveform, self.padding)
        waveform_wphase = np.exp(1j * np.deg2rad(self.phase)) * waveform_padded
        return waveform_wphase


class FileDefined(Waveform):
    def __init__(self, prog, gen_ch, filepath, phase=0, maxv=None, drag_coeff=0,
                 padding: Union[float, List[float], None] = None):
        super().__init__(prog, gen_ch, phase=phase, maxv=maxv)
        self.filepath = filepath
        self.padding = padding
        self.drag_coeff = drag_coeff
        self.waveform = self._generate_waveform(filepath=self.filepath)

    @staticmethod
    def core(filepath, **kwargs):
        """
        Reads waveform data (I, Q) from the file.

        param filepath: the filepath of the waveform
        return:
        """
        # todo: deal with file formats
        filetype = filepath.split(".")[-1]
        if filetype == "npy":
            data = np.load(filepath, **kwargs)
            idata = data[:, 0]  # First column: I data
            qdata = data[:, 1]  # Second column: Q data
            return idata + 1j * qdata
        elif filetype == "csv":
            data = np.loadtxt(filepath, **kwargs)  # Assuming the file contains a two-column format (I, Q)
            idata = data[:, 0]  # First column: I data
            qdata = data[:, 1]  # Second column: Q data
            return idata + 1j * qdata
        else:
            try:
                data = np.loadtxt(filepath, **kwargs)  # Assuming the file contains a two-column format (I, Q)
                idata = data[:, 0]  # First column: I data
                qdata = data[:, 1]  # Second column: Q data
                return idata + 1j * qdata
            except Exception as e:
                raise ValueError(f"Error reading file {filepath}. Exception {e}")

    def _generate_waveform(self, *args, **kwargs):
        """
        apply the necessary modificaiton to the core function,
        generate in-phase (I) and quadrature (Q) components
        """
        waveform = self.core(*args, **kwargs)
        waveform_padded = self._apply_padding(waveform, self.padding)
        waveform_wphase = np.exp(1j * np.deg2rad(self.phase)) * waveform_padded
        # waveform_dragged = self.apply_drag_modulation(waveform_wphase, drag_coeff=self.drag_coeff)
        return waveform_wphase


class GaussianModulated(Waveform):
    def __init__(self, prog, gen_ch, length, sigma, phase=0, maxv=None, modulations: list = (),
                 padding: Union[float, List[float], None] = None, shape=None):
        super().__init__(prog, gen_ch, phase=phase, maxv=maxv)
        self.sigma_samps = self.us_to_samps(sigma)
        self.length_samps = self.us_to_samps(length)
        self.padding = padding
        self.modulations = modulations
        self.waveform = self._generate_waveform(self.length_samps, self.sigma_samps)
        shape = shape if shape is not None else self.__class__.__name__
        WaveformRegistry.register(shape, self.__class__)

    @staticmethod
    def core(length, sigma):
        """the definetion of Gaussian"""
        t = np.arange(length)
        y = np.exp(-(t - length / 2) ** 2 / sigma ** 2)
        return y - np.min(y)

    def _generate_waveform(self, *args, **kwargs):
        """
        apply the necessary modificaiton to the core function,
        generate in-phase (I) and quadrature (Q) components
        """
        waveform = self.maxv * self.core(*args, **kwargs)
        waveform = self._apply_padding(waveform, self.padding)
        waveform = np.exp(1j * np.deg2rad(self.phase)) * waveform
        for mod in self.modulations:
            waveform = mod.apply_modulation(waveform, self.sampling_rate)
        return waveform


class TanhBoxModulated(Waveform):
    def __init__(self, prog, gen_ch, length, ramp_width, cut_offset=0.01, phase=0, maxv=None, modulations: list = (),
                 padding: Union[float, List[float], None] = None, shape=None):
        super().__init__(prog, gen_ch, phase=phase, maxv=maxv)
        self.ramp_samps = self.us_to_samps(ramp_width)
        self.length_samps = self.us_to_samps(length)
        self.cut_offset = cut_offset
        self.padding = padding
        self.modulations = modulations
        self.waveform = self._generate_waveform(self.length_samps, self.ramp_samps, self.cut_offset)
        shape = shape if shape is not None else self.__class__.__name__
        WaveformRegistry.register(shape, self.__class__)

    @staticmethod
    def core(length, ramp_width, cut_offset):
        """
        Create a numpy array containing a smooth box pulse made of two tanh functions subtract from each other.

        :param length: number of points of the pulse
        :param ramp_width: number of points from cutOffset to 0.95 amplitude
        :param cut_offset: the initial offset to cut on the tanh Function
        :return:
        """
        t = np.arange(length)
        c0_, c1_ = np.arctanh(2 * cut_offset - 1), np.arctanh(2 * 0.95 - 1)
        k_ = (c1_ - c0_) / ramp_width
        y = (0.5 * (np.tanh(k_ * t + c0_) - np.tanh(k_ * (t - length) - c0_)) - cut_offset) / (1 - cut_offset)
        return y - np.min(y)

    def _generate_waveform(self, *args, **kwargs):
        """
        apply the necessary modificaiton to the core function,
        generate in-phase (I) and quadrature (Q) components
        """
        waveform = self.maxv * self.core(*args, **kwargs)
        waveform = self._apply_padding(waveform, self.padding)
        waveform = np.exp(1j * np.deg2rad(self.phase)) * waveform
        for mod in self.modulations:
            waveform = mod.apply_modulation(waveform, self.sampling_rate)
        return waveform


class ConcatenateWaveform(Waveform):
    def __init__(self, prog, gen_ch, waveforms: List[Waveform], phase=0, maxv=None, shape=None):
        super().__init__(prog, gen_ch, phase, maxv)
        self.wavefrom_list = waveforms
        self.waveform = self._generate_waveform()
        shape = shape if shape is not None else self.__class__.__name__
        WaveformRegistry.register(shape, self.__class__)

    def _generate_waveform(self):
        return np.concatenate([w.waveform for w in self.wavefrom_list])


def add_waveform(prog: QickProgram, gen_ch, name, shape, **kwargs):
    """Adds a waveform to the DAC channel, using physical parameters of the pulse.
    The pulse will peak at length/2.

    Parameters
    ----------
    prog: QickProgram
        The experiment QickProgram
    gen_ch : str
        name of the generator channel
    name : str
        Name of the pulse
    shape : str
        shape/type of the pulse, e.g. Gaussian, TanhBoxChirp
    """
    if shape.lower() in (wave.lower() for wave in WaveformRegistry.available_waveforms()):
        pulse = WaveformRegistry.create(shape=shape, prog=prog, gen_ch=gen_ch, **kwargs)
        # pulse.plot_waveform()
        pulse.add_waveform(prog, name=name)
    else:
        raise NameError(f"Unsupported pulse shape {shape}."
                        f"Choose from available shapes: {WaveformRegistry.available_waveforms()},"
                        f"or define new waveforms.")


def add_waveform_concatenate(prog: QickProgram, gen_ch: str | int, name, gatelist, maxv=None):
    pass


if __name__ == "__main__":
    from Hatlab_RFSOC.core.averager_program import NDAveragerProgram, QubitMsmtMixin
    from Hatlab_RFSOC.proxy import getSocProxy
    from Hatlab_RFSOC.waveform import modulation
    import yaml

    # --------------------- initialize a qick program ----------------------------------------------
    def get_cfg_info(cfgFilePath):
        yml = yaml.safe_load(open(cfgFilePath))
        config, info = yml["config"], yml["info"]
        return config, info

    class Program(QubitMsmtMixin, NDAveragerProgram):
        def initialize(self):
            self.sync_all(self.us2cycles(1))  # give processor some time to configure pulses

        def body(self):
            pass

    cfgFilePath = r"W:\code\SubHarmonic_20250307\RFSOC_phaseReset\config_files\20250307_SubHarmonic_Q1_amp_bigenv2.yml"
    config, info = get_cfg_info(cfgFilePath)
    soc, soccfg = getSocProxy(info["PyroServer"])
    expt_cfg = {"reps":  1, "relax_delay": 20}
    config.update(expt_cfg)
    prog = Program(soccfg, config)

    # --------------------- generate waveforms ----------------------------------------------------
    # wf = Gaussian(prog, 0, length=0.05, sigma=0.01, phase=0, padding=[0.05, 0.05])
    wf = TanhBox(prog, 0, length=0.1, ramp_width=0.01, phase=0, padding=[0.015, 0.015])
    wf.plot_waveform()

    # --------------------- generate modulated waveforms ----------------------------------------------------
    # define modulations
    def cfunc(amp, maxf, maxv):
        return maxf * (amp/maxv)**2
    cm = modulation.ChirpModulation(chirp_func=cfunc, maxf=-50, maxv=30000)
    dm = modulation.DragModulation(0.003)

    wf2 = GaussianModulated(prog=prog, gen_ch=0, length=0.05, sigma=0.01, phase=0, padding=[0.015, 0.015],
                            modulations=[dm, cm], shape="GaussianChirpDrag")
    wf2 = TanhBoxModulated(prog=prog, gen_ch=0, length=0.05, ramp_width=0.01, phase=0, padding=[0.015, 0.015],
                           modulations=[dm, cm], shape="TanhboxChirpDrag")
    wf2.plot_waveform()

    wfc = ConcatenateWaveform(prog=prog, gen_ch=0, waveforms=[wf, wf2], phase=0, shape="concatenated_pulse_1")
    wfc.plot_waveform()

    corrFile = r"W:\data\SubHarmonic\WileE_20250326\Q1\calibration\\" \
               r"Q1_DAC0_Line1_Subh1000-1300MHz_5000DAC_Q3700-3720MHz_8000DAC-2_epsilon_smoothed(2).csv"
    wcorr = modulation.WaveformCorrection(filepath=corrFile, freq=1100, scale="linear", max_scale=0.5)
    wfcorr = TanhBoxModulated(prog=prog, gen_ch=0, length=0.1, ramp_width=0.01, phase=0, padding=[0.015, 0.015],
                           modulations=[cm, wcorr], shape="TanhboxCorrected")
    wfcorr = GaussianModulated(prog=prog, gen_ch=0, length=0.1, sigma=0.02, phase=0, padding=[0.015, 0.015],
                           modulations=[wcorr], shape="GaussianCorrected")
    wfcorr.plot_waveform()
    plt.ylim((-30000, 33000))

    wf_fft = wcorr.compute_fourier_transform(wfcorr.waveform, wfcorr.sampling_rate)
    plt.figure()
    plt.plot(wf_fft[0], wf_fft[1])

    t_list = np.linspace(0, (len(wf.waveform) - 1) / wf.sampling_rate, len(wf.waveform))
    signal = wf.waveform
    signal_wc = wcorr.apply_modulation(signal, wf.sampling_rate)
    signal_recv = wcorr.recover_modulation(signal_wc, wf.sampling_rate)

    plt.figure()
    plt.plot(t_list, np.abs(signal), label="original")
    plt.plot(t_list, np.abs(signal_wc), label="corrected")
    plt.plot(t_list, np.abs(signal_recv), label="undo")
    plt.legend()


