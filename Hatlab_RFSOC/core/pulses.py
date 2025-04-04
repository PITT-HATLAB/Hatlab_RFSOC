from typing import List, Union, Type
import numpy as np
from qick.asm_v1 import QickProgram
import matplotlib.pyplot as plt

NumType = Union[int, float]


def tanh_box(length: int, ramp_width: int, cut_offset=0.01, maxv=30000):
    """
    Create a numpy array containing a smooth box pulse made of two tanh functions subtract from each other.

    :param length: Length of array (in points)
    :param ramp_width: number of points from cutOffset to 0.95 amplitude
    :param cut_offset: the initial offset to cut on the tanh Function
    :param maxv: the max value of the waveform
    :return:
    """
    x = np.arange(0, length)
    c0_ = np.arctanh(2 * cut_offset - 1)
    c1_ = np.arctanh(2 * 0.95 - 1)
    k_ = (c1_ - c0_) / ramp_width
    y = (0.5 * (np.tanh(k_ * x + c0_) - np.tanh(k_ * (x - length) - c0_)) - cut_offset) / (
            1 - cut_offset) * maxv
    return y - np.min(y)


def gaussian(sigma: int, length: int, maxv=30000):
    """
    Create a numpy array containing a Gaussian function.

    :param sigma: sigma (standard deviation) of Gaussian
    :param length: total number of points of gaussian pulse
    :param maxv: the max value of the waveform
    :return:
    """
    x = np.arange(0, length)
    y = maxv * np.exp(-(x - length / 2) ** 2 / sigma ** 2)
    y = y - np.min(y)
    return y


def tanh_box_fm(freq: float, length: int, ramp_width: int, cut_offset=0.01, maxv=30000):
    x = np.arange(0, length)
    y = tanh_box(length, ramp_width, cut_offset, maxv) * np.cos(2*np.pi * freq * x)
    return y


def tanh_box_IQ(freq: float, length: int, ramp_width: int, cut_offset=0.01, maxv=30000):
    x = np.arange(0, length)
    i = tanh_box(length, ramp_width, cut_offset, maxv) * np.cos(2*np.pi * freq * x)
    q = tanh_box(length, ramp_width, cut_offset, maxv) * np.sin(2*np.pi * freq * x)
    return [i, q]


def gaussian_fm(freq: float, sigma: int, length: int, maxv=30000):
    x = np.arange(0, length)
    y = gaussian(sigma, length, maxv) * np.cos(2*np.pi * freq * x)
    return y


def add_padding(data, soc_gencfg, padding):
    """
    pad some zeros before and/or after the waveform data
    :param data:
    :param soc_gencfg: gen_ch config
    :param padding: the length of padding in us
    :return:
    """
    samps_per_clk = soc_gencfg['samps_per_clk']
    fclk = soc_gencfg['f_fabric']

    if isinstance(padding, int | float):
        padding = np.array([0, padding])
    else:
        padding = np.array(padding)
    padding_samp = np.ceil(padding * fclk * samps_per_clk)
    data = np.concatenate((np.zeros(int(padding_samp[0])), data, np.zeros(int(padding_samp[1]))))

    return data


def add_tanh(prog: QickProgram, gen_ch, name, length: float, ramp_width: float, cut_offset: float = 0.01,
             phase: float = 0, maxv=None, padding: Union[NumType, List[NumType]] = None, drag: float = 0):
    """
    Adds a smooth box pulse made of two tanh functions to the waveform library, using physical parameters of the pulse.
    The pulse will peak at length/2.

    Parameters
    ----------
    gen_ch : str
        name of the DAC channel defined in the YAML
    name : str
        Name of the pulse
    length : float
        Total pulse length (in units of us)
    ramp_width : float
        ramping time from cut_offset to 0.95 amplitude (in units of us)
    cut_offset: float
        the initial offset to cut on the tanh Function (in unit of unit-height pulse)
    phase: float
        the phase of the waveform in degree
    maxv : float
        Value at the peak (if None, the max value for this generator will be used)
    padding: float | List[float]
        padding zeros in front of and at the end of the pulse

    """

    gen_ch = prog.cfg["gen_chs"][gen_ch]["ch"]
    soc_gencfg = prog.soccfg['gens'][gen_ch]
    if maxv is None:
        maxv = soc_gencfg['maxv'] * soc_gencfg['maxv_scale']
    samps_per_clk = soc_gencfg['samps_per_clk']
    fclk = soc_gencfg['f_fabric']

    # length_cyc = prog.us2cycles(length, gen_ch=gen_ch)
    length_cyc = length * fclk
    length_reg = length_cyc * samps_per_clk
    # ramp_reg = np.int64(np.round(ramp_width*fclk*samps_per_clk))
    ramp_reg = ramp_width * fclk * samps_per_clk

    wf = tanh_box(length_reg, ramp_reg, cut_offset, maxv=maxv)
    if padding is not None:
        wf = add_padding(wf, soc_gencfg, padding)
    zero_padding = np.zeros((16 - len(wf)) % 16)
    wf_padded = np.concatenate((wf, zero_padding))
    drag_padded = -np.gradient(wf_padded) * drag

    wf_idata = np.cos(np.pi / 180 * phase) * wf_padded - np.sin(np.pi / 180 * phase) * drag_padded
    wf_qdata = np.sin(np.pi / 180 * phase) * wf_padded + np.cos(np.pi / 180 * phase) * drag_padded

    # prog.add_pulse(gen_ch, name, idata=wf_padded)
    prog.add_pulse(gen_ch, name, idata=wf_idata, qdata=wf_qdata)


def add_gaussian(prog: QickProgram, gen_ch: str, name, sigma: float, length: float, phase: float = 0,
                 maxv=None, padding: Union[NumType, List[NumType]] = None, drag: float = 0):
    """Adds a gaussian pulse to the waveform library, using physical parameters of the pulse.
    The pulse will peak at length/2.

    Parameters
    ----------
    gen_ch : str
        name of the generator channel
    name : str
        Name of the pulse
    sigma : float
        sigma of gaussian (in units of us)
    length : float
        Total pulse length (in units of us)
    maxv : float
        Value at the peak (if None, the max value for this generator will be used)

    """

    gen_ch = prog.cfg["gen_chs"][gen_ch]["ch"]
    soc_gencfg = prog.soccfg['gens'][gen_ch]
    if maxv is None:
        maxv = soc_gencfg['maxv'] * soc_gencfg['maxv_scale']
    samps_per_clk = soc_gencfg['samps_per_clk']
    fclk = soc_gencfg['f_fabric']

    # length_cyc = prog.us2cycles(length, gen_ch=gen_ch)
    length_cyc = length * fclk
    length_reg = length_cyc * samps_per_clk
    sigma_reg = sigma * fclk * samps_per_clk

    wf = gaussian(sigma_reg, length_reg, maxv=maxv)
    if padding is not None:
        wf = add_padding(wf, soc_gencfg, padding)
    zero_padding = np.zeros((16 - len(wf)) % 16)
    wf_padded = np.concatenate((wf, zero_padding))
    drag_padded = -np.gradient(wf_padded) * drag

    wf_idata = np.cos(np.pi / 180 * phase) * wf_padded - np.sin(np.pi / 180 * phase) * drag_padded
    wf_qdata = np.sin(np.pi / 180 * phase) * wf_padded + np.cos(np.pi / 180 * phase) * drag_padded

    # prog.add_pulse(gen_ch, name, idata=wf_padded)
    prog.add_pulse(gen_ch, name, idata=wf_idata, qdata=wf_qdata)


def add_arbitrary(prog: QickProgram, gen_ch: str, name, envelope, phase: float = 0,
                  maxv=None, padding: Union[NumType, List[NumType]] = None, drag: float = 0):
    """Adds an arbitrary pulse to the waveform library, using physical parameters of the pulse.
    The pulse will peak at length/2.

    Parameters
    ----------
    gen_ch : str
        name of the generator channel
    name : str
        Name of the pulse
    envelope : float
        the envelope of the waveform
    maxv : float
        Value at the peak (if None, the max value for this generator will be used)

    """    
    gen_ch = prog.cfg["gen_chs"][gen_ch]["ch"]
    soc_gencfg = prog.soccfg['gens'][gen_ch]

    wf = envelope

    if padding is not None:
        wf = add_padding(wf, soc_gencfg, padding)
    zero_padding = np.zeros((16 - len(wf)) % 16)
    wf_padded = np.concatenate((wf, zero_padding))
    drag_padded = -np.gradient(wf_padded) * drag

    wf_idata = np.cos(np.pi / 180 * phase) * drag_padded
    wf_qdata = np.sin(np.pi / 180 * phase) * drag_padded

    # prog.add_pulse(gen_ch, name, idata=wf_padded)
    prog.add_pulse(gen_ch, name, idata=wf_idata, qdata=wf_qdata)


def add_pulse_concatenate(prog: QickProgram, gen_ch: str | int, name, gatelist, maxv=None):
    def get_gain_max(gatelist):
        gmax = 0
        for gate in gatelist:
            gmax = gate['gain'] if gate['gain'] > gmax else gmax
        return gmax
    gen_ch = prog.cfg["gen_chs"][gen_ch]["ch"] if type(gen_ch) == str else gen_ch
    soc_gencfg = prog.soccfg['gens'][gen_ch]
    if maxv is None:
        maxv = soc_gencfg['maxv'] * soc_gencfg['maxv_scale']
    samps_per_clk = soc_gencfg['samps_per_clk']
    fclk = soc_gencfg['f_fabric']

    wfdata_i = []
    wfdata_q = []
    wf_len_list = []
    gmax = get_gain_max(gatelist)
    for gate in gatelist:
        maxv_p = gate.get('maxv', maxv)
        if gate['shape'] == 'gaussian':
            length_reg = gate['length'] * fclk * samps_per_clk
            sigma_reg = gate['sigma'] * fclk * samps_per_clk
            pulsedata = gate['gain'] / gmax * gaussian(sigma_reg, length_reg, maxv=maxv_p)

        elif gate['shape'] == 'tanh_box':
            length_reg = gate['length'] * fclk * samps_per_clk
            ramp_reg = gate['ramp_width'] * fclk * samps_per_clk
            pulsedata = gate['gain'] / gmax * tanh_box(length_reg, ramp_reg, maxv=maxv_p)

        else:
            raise NameError(f"unsupported pulse shape {gate['shape']}")

        padding = gate.get('padding')
        if gate['padding'] is not None:
            pulsedata = add_padding(pulsedata, soc_gencfg, padding)

        wfdata_i = np.concatenate((wfdata_i, pulsedata * np.cos(gate['phase'] / 360 * 2 * np.pi)))
        wfdata_q = np.concatenate((wfdata_q, pulsedata * np.sin(gate['phase'] / 360 * 2 * np.pi)))
        zero_padding = np.zeros((16 - len(wfdata_i)) % 16)
        wfdata_i = np.concatenate((wfdata_i, zero_padding))
        wfdata_q = np.concatenate((wfdata_q, zero_padding))
        # print("gate phase: ", gate['phase'])
        # wf_len_list.append(len(wfdata_i) / 16.0)

    # print(wf_len_list)
    if len(wfdata_i) == 0:
        prog.add_pulse(gen_ch, name, idata=3 * [0] * samps_per_clk, qdata=3 * [0] * samps_per_clk)
    else:
        prog.add_pulse(gen_ch, name, idata=wfdata_i, qdata=wfdata_q)


class WaveformRegistry:
    _registry = {}

    @classmethod
    def register(cls, name: str, waveform_cls: Type['Waveform']):
        cls._registry[name] = waveform_cls

    @classmethod
    def create(cls, name: str, *args, **kwargs) -> 'Waveform':
        if name not in cls._registry:
            raise ValueError(f"Waveform '{name}' is not registered.")
        return cls._registry[name](*args, **kwargs)

    @classmethod
    def available_waveforms(cls):
        return list(cls._registry.keys())


class Waveform:
    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        WaveformRegistry.register(cls.__name__, cls)

    def __init__(self, prog: QickProgram, gen_ch: Union[int, str], phase, maxv):
        self.maxv = maxv
        self.phase = phase
        self.waveform = None
        self._set_channel_cfg(prog, gen_ch)

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
        prog.add_pulse(self.gen_ch, name, idata=idata, qdata=qdata)

    def _set_channel_cfg(self, prog: QickProgram, gen_ch: Union[int, str]):
        self.gen_ch = prog.cfg["gen_chs"][gen_ch]["ch"] if isinstance(gen_ch, str) else gen_ch
        soc_gencfg = prog.soccfg['gens'][gen_ch]
        self.samps_per_clk = soc_gencfg['samps_per_clk']
        self.fclk = soc_gencfg['f_fabric']
        self.sampling_rate = self.samps_per_clk * self.fclk

    def us_to_samps(self, length):
        """Convert length in physical units to register units."""
        return length * self.sampling_rate

    def _pad_waveform(self, waveform: np.ndarray) -> np.ndarray:
        """Pads waveform to be a multiple of samps_per_clk."""
        pad_len = (-len(waveform)) % self.samps_per_clk
        return np.pad(waveform, (0, pad_len))

    def _apply_padding(self, data: np.ndarray, padding: Union[float, List[float], None]) -> np.ndarray:
        """Pads waveform with zeros before and/or after."""
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


class DragModulationMixin:
    @staticmethod
    def _drag_func(waveform, drag_coeff: float = 0, sampling_rate=1):
        """drag correction for resonant driving"""
        dt = 1/sampling_rate
        return -drag_coeff * np.exp(1j * np.pi/2) * np.gradient(waveform, dt)

    def apply_drag_modulation(self, waveform, drag_func=None, drag_coeff: float = 0, sampling_rate=1):
        """
        Apply a drag modulation to the input waveform.

        Parameters:
        - waveform: Input waveform array.
        - drag_func: A callable accepting (waveform, drag_coeff, sampling_rate) that returns the drag correction.
        - drag_coeff: A coefficient for the drag correction amplitude
        """
        if drag_func is None:
            drag_func = self._drag_func
        wf_drag = drag_func(waveform, drag_coeff, sampling_rate)
        return waveform + wf_drag


class ChirpModulationMixin:
    @staticmethod
    def _instant_frequency(chirp_func, waveform, maxf, maxv):
        return chirp_func(np.abs(waveform), maxf, maxv)

    @staticmethod
    def _chirp_phase(instant_freq, sampling_rate):
        phase = np.zeros(len(instant_freq))
        phi0 = 0
        for i in range(len(instant_freq) - 1):
            phase[i] = phi0
            phi0 += np.pi * (instant_freq[i] + instant_freq[i + 1]) / sampling_rate
        phase[-1] = phi0
        return phase

    def apply_chirp_modulation(self, waveform, chirp_func, sampling_rate, maxf=0, maxv=30000):
        """
        Apply a chirp modulation to the input waveform.

        Parameters:
        - waveform: Input waveform array.
        - chirp_func: A callable accepting (amp, maxf, maxv) that returns the chirp frequency at the given amplitude.
        - sampling_rate: sampling_rate of the waveform
        - maxf: Maximum chirp instanteous frequency in MHz. Default value is 0
        - maxv: Maximum amplitude of the given waveform. Default value is 30000
        """
        chirp_freq = self._instant_frequency(chirp_func, waveform, maxf, maxv)
        chirp_phase = self._chirp_phase(chirp_freq, sampling_rate)
        wf_chirp = waveform * np.exp(1j * chirp_phase)

        return wf_chirp


class WaveformCorrectionMixin:
    @staticmethod
    def compute_fourier_transform(signal, sampling_rate):
        """
        Compute the Fourier Transform of a signal.

        Parameters:
        - signal: numpy array of the waveform data.
        - sampling_rate: sampling rate in Hz.

        Returns:
        - frequencies: numpy array of frequency bins (MHz).
        - magnitudes: numpy array of corresponding magnitude spectrum.
        """
        N = len(signal)
        fft_vals = np.fft.fft(signal)
        fft_freq = np.fft.fftfreq(N, d=1 / sampling_rate)

        return fft_freq, fft_vals

    @staticmethod
    def compute_inverse_fourier_transform(fft_signal):
        """
        Compute the Inverse Fourier Transform of a complex frequency-domain signal.

        Parameters:
          fft_signal: numpy array of complex Fourier coefficients (full spectrum).

        Returns:
          time_signal: numpy array representing the recovered complex time-domain signal.
        """
        time_signal = np.fft.ifft(fft_signal)
        return time_signal

    @staticmethod
    def calibrate_waveform_in_frequency_domain(signal, calibration_func, sampling_rate):
        """
        Modify a waveform in the frequency domain and return the modified time-domain waveform.

        Parameters:
            - signal (np.array): Input time-domain signal (can be complex or real).
            - calibration_func (callable): A function that takes (fft_freq, fft_values) as inputs
                                          and returns modified fft_values.
            - sampling_rate (float): Sampling rate of the signal in Hz.

        Returns:
            np.array: The modified time-domain signal (complex if the input was complex).
        """
        N = len(signal)
        # Compute the Fourier Transform and associated frequency bins.
        fft_values = np.fft.fft(signal)
        fft_freq = np.fft.fftfreq(N, d=1 / sampling_rate)

        # Apply the provided modification function to the FFT coefficients.
        modified_fft_values = calibration_func(fft_freq) * fft_values

        # Compute the inverse FFT to get back the modified time-domain waveform.
        modified_waveform = np.fft.ifft(modified_fft_values)
        return modified_waveform

    @staticmethod
    def _get_calib_data(calibFilepath):
        return np.loadtxt(calibFilepath, delimiter=",")

    @staticmethod
    def get_calibration_func(calibration_data, dac_ref, freq_ref, attenuation):
        from scipy.interpolate import CubicSpline
        freq_MHz = calibration_data[0]
        S21_dbm = calibration_data[1]
        S21_interpolate = CubicSpline(freq_MHz, S21_dbm + attenuation)
        interpolation = CubicSpline(freq_MHz, 10**(-(-S21_dbm + S21_interpolate(freq_ref))/10))
        def calib_func(val):
            f_min = np.min(freq_MHz)
            f_max = np.max(freq_MHz)
            val_array = np.atleast_1d(val)  # Ensure val is treated as a numpy array.
            mask = (val_array >= f_min) & (val_array <= f_max)  # Create a mask for values within the range.
            # For values within the calibrated range, use the interpolation function. Otherwise, return 1
            result = np.where(mask, interpolation(val_array), 1)
            if result.size == 1:
                return result.item()  # Return a scalar if the input was a scalar.
            return result

        return calib_func

    @staticmethod
    def get_calibration_func2(calibration_data, dac_ref, freq_ref, attenuation):
        from scipy.interpolate import CubicSpline
        freq_MHz = calibration_data[0]
        S21_dbm = calibration_data[1]
        S21_interpolate = CubicSpline(freq_MHz, S21_dbm + attenuation)
        interpolation = CubicSpline(freq_MHz, 10 ** ((-S21_dbm + S21_interpolate(freq_ref)) / 10))

        def calib_func(val):
            f_min = np.min(freq_MHz)
            f_max = np.max(freq_MHz)
            val_array = np.atleast_1d(val)  # Ensure val is treated as a numpy array.
            mask = (val_array >= f_min) & (val_array <= f_max)  # Create a mask for values within the range.
            # For values within the calibrated range, use the interpolation function. Otherwise, return 1
            result = np.where(mask, interpolation(val_array), 1)
            if result.size == 1:
                return result.item()  # Return a scalar if the input was a scalar.
            return result

        return calib_func


class Gaussian(Waveform, DragModulationMixin):
    def __init__(self, prog, gen_ch, length, sigma, phase=0, maxv=30000, drag_coeff=0,
                 padding: Union[float, List[float], None] = None):
        super().__init__(prog, gen_ch, phase=phase, maxv=maxv)
        self.sigma_samps = self.us_to_samps(sigma)
        self.length_samps = self.us_to_samps(length)
        self.padding = padding
        self.drag_coeff = drag_coeff
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
        waveform_drag = self.apply_drag_modulation(waveform_wphase, drag_coeff=self.drag_coeff,
                                                   sampling_rate=self.sampling_rate)
        return waveform_drag


class TanhBox(Waveform, DragModulationMixin):
    def __init__(self, prog, gen_ch, length, ramp_width, cut_offset=0.01, phase=0, maxv=30000, drag_coeff=0.0,
                 padding: Union[float, List[float], None] = None):
        super().__init__(prog, gen_ch, phase=phase, maxv=maxv)
        self.ramp_samps = self.us_to_samps(ramp_width)
        self.length_samps = self.us_to_samps(length)
        self.cut_offset = cut_offset
        self.padding = padding
        self.drag_coeff = drag_coeff
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
        waveform_drag = self.apply_drag_modulation(waveform_wphase, drag_coeff=self.drag_coeff,
                                                   sampling_rate=self.sampling_rate)
        return waveform_drag


class FileDefined(Waveform, DragModulationMixin):
    def __init__(self, prog, gen_ch, filepath, phase=0, maxv=30000, drag_coeff=0,
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
        try:
            data = np.loadtxt(filepath, **kwargs)  # Assuming the file contains a two-column format (I, Q)
            idata = data[:, 0]  # First column: I data
            qdata = data[:, 1]  # Second column: Q data
        except Exception as e:
            raise ValueError(f"Error reading file {filepath}: {e}")

        return idata + 1j*qdata

    def _generate_waveform(self, *args, **kwargs):
        """
        apply the necessary modificaiton to the core function,
        generate in-phase (I) and quadrature (Q) components
        """
        waveform = self.maxv * self.core(*args, **kwargs)
        waveform_padded = self._apply_padding(waveform, self.padding)
        waveform_wphase = np.exp(1j * np.deg2rad(self.phase)) * waveform_padded
        waveform_dragged = self.apply_drag_modulation(waveform_wphase, drag_coeff=self.drag_coeff)
        return waveform_dragged


class GaussianChirped(Gaussian, ChirpModulationMixin):
    def __init__(self, chirp_func, maxf, maxv=30000, drag_coeff=0, **kwargs):
        self.chirp_func = chirp_func
        self.maxf = maxf
        self.maxv = maxv
        super().__init__(maxv=maxv, drag_coeff=drag_coeff, **kwargs)

    def _generate_waveform(self, *args, **kwargs):
        waveform = super()._generate_waveform(*args, **kwargs)
        waveform_chirped = self.apply_chirp_modulation(waveform, self.chirp_func, self.sampling_rate, self.maxf, self.maxv)
        return waveform_chirped


class TanhBoxChirped(TanhBox, ChirpModulationMixin):
    def __init__(self, chirp_func, maxf, maxv=30000, drag_coeff=0, **kwargs):
        self.chirp_func = chirp_func
        self.maxf = maxf
        self.maxv = maxv
        super().__init__(maxv=maxv, drag_coeff=drag_coeff, **kwargs)

    def _generate_waveform(self, *args, **kwargs):
        waveform = super()._generate_waveform(*args, **kwargs)
        waveform_chirped = self.apply_chirp_modulation(waveform, self.chirp_func, self.sampling_rate, self.maxf, self.maxv)
        return waveform_chirped


if __name__ == "__main__":
    from Hatlab_RFSOC.core.averager_program import NDAveragerProgram, QubitMsmtMixin
    from Hatlab_RFSOC.proxy import getSocProxy
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
    expt_cfg = {"reps":  5000, "relax_delay": 20}
    config.update(expt_cfg)
    prog = Program(soccfg, config)

    # --------------------- generate waveforms ----------------------------------------------------
    # wf = Gaussian(prog, 0, length=0.05, sigma=0.01, phase=0, drag_coeff=0, padding=[0.05, 0.05])
    # wf = TanhBox(prog, 0, length=0.05, ramp_width=0.01, phase=0, drag_coeff=0.003183, padding=[0.015, 0.015])

    # --------------------- generate chriped waveforms ----------------------------------------------------
    def cfunc(amp, maxf, maxv):
        return maxf * (amp/maxv)**2
    wf = TanhBoxChirped(prog=prog, gen_ch=0, length=0.05, ramp_width=0.01, phase=0, drag_coeff=0.003183,
                        chirp_func=cfunc, maxf=-50, padding=[0.03, 0.03])

    wf.plot_waveform()
    wf.add_waveform(prog, "pulseTest")

