import warnings
from typing import List, Union, Type, Callable
import numpy as np
from qick.asm_v1 import QickProgram
import matplotlib.pyplot as plt

NumType = Union[int, float]


class DragModulation:
    def __init__(self, drag_factor, drag_func: Callable = None):
        """
        Apply a drag modulation to the input waveform.

        Parameters:
        - drag_factor: A coefficient for the drag correction amplitude
        - drag_func: A callable accepting (waveform, drag_factor, sampling_rate) that returns the drag correction.
        """
        self.drag_factor = drag_factor
        self.drag_func = drag_func if drag_func is not None else self._drag_func

    @staticmethod
    def _drag_func(waveform, sampling_rate=1):
        """drag correction for resonant driving"""
        dt = 1/sampling_rate
        return -1 * np.exp(1j * np.pi/2) * np.gradient(waveform, dt)

    def apply_modulation(self, waveform, sampling_rate=1):
        wf_drag = self.drag_factor * self.drag_func(waveform, sampling_rate)
        return waveform + wf_drag


class ChirpModulation:
    def __init__(self, chirp_func, maxf, maxv=None):
        """
        Apply a chirp modulation to the input waveform.

        Parameters:
        - chirp_func: A callable accepting (amp, maxf, maxv) that returns the chirp frequency at the given amplitude.
        - maxf: Maximum chirp instanteous frequency in MHz. Default value is 0
        - maxv: Maximum amplitude of the given waveform. Default value is 30000
        """
        self.maxf = maxf
        self.maxv = maxv if maxv is not None else 32766
        self.chirp_func = chirp_func

    def _instant_frequency(self, waveform):
        max_i = np.max(np.abs(np.real(waveform)))
        max_q = np.max(np.abs(np.imag(waveform)))
        # return self.chirp_func(np.abs(waveform), self.maxf, np.max((max_i, max_q)))
        return self.chirp_func(np.abs(waveform), self.maxf, np.max(np.abs(waveform)))
        # return self.chirp_func(np.abs(waveform), self.maxf, self.maxv)

    @staticmethod
    def _chirp_phase(instant_freq, sampling_rate):
        phase = np.zeros(len(instant_freq))
        phi0 = 0
        for i in range(len(instant_freq) - 1):
            phase[i] = phi0
            phi0 += np.pi * (instant_freq[i] + instant_freq[i + 1]) / sampling_rate
        phase[-1] = phi0
        return phase

    def apply_modulation(self, waveform, sampling_rate):
        """
        Apply a chirp modulation to the input waveform.

        Parameters:
        - waveform: Input waveform array.
        - sampling_rate: sampling_rate of the waveform
        """
        chirp_freq = self._instant_frequency(waveform)
        chirp_phase = self._chirp_phase(chirp_freq, sampling_rate)
        wf_chirp = waveform * np.exp(1j * chirp_phase)

        return wf_chirp


class WaveformCorrection:
    def __init__(self, filepath, freq, scale: str = "linear", max_scale=0.5):
        self.calibration_data = self.get_calib_data(filepath)
        self.freq = freq
        self.scale = scale
        self.max_scale = max_scale
        if scale.lower() in ["db", "dbm", "log"]:
            self.freq_ref = self.calibration_data[0][
                np.argmin(np.abs(self.calibration_data[1] - (np.max(self.calibration_data[1]) - 3)))]
        elif scale.lower() == "linear":
            self.freq_ref = self.calibration_data[0][
                np.argmin(np.abs(self.calibration_data[1] - max_scale*np.max(self.calibration_data[1])))]

        self.calibration_func = self.get_calibration_func(freq_ref=self.freq_ref, attenuation=0)
        self.recover_func = self.get_recover_func(freq_ref=self.freq_ref, attenuation=0)

    @staticmethod
    def get_calib_data(filepath):
        data = np.loadtxt(filepath, delimiter=",")
        return data

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

    def apply_modulation(self, waveform, sampling_rate):
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
        N = len(waveform)
        t_list = np.arange(0, N) / sampling_rate
        signal = waveform * np.exp(1j * 2 * np.pi * self.freq * t_list)

        # N = len(waveform)
        # signal = waveform

        # Compute the Fourier Transform and associated frequency bins.
        fft_values = np.fft.fft(signal)
        fft_freq = np.fft.fftfreq(N, d=1 / sampling_rate)

        # limit pulse bandwidth by cutting off the component smaller than 0.01%
        fft_max = np.max(np.abs(fft_values))
        mask = (np.abs(fft_values)/fft_max >= 1e-4)
        fft_values = np.where(mask, fft_values, 0)

        # Apply the provided modification function to the FFT coefficients.
        modified_fft_values = self.calibration_func(fft_freq) * fft_values

        # Compute the inverse FFT to get back the modified time-domain waveform.
        modified_signal = np.fft.ifft(modified_fft_values)
        modified_waveform = modified_signal * np.exp(-1j * 2 * np.pi * self.freq * t_list)

        return modified_waveform

    def recover_modulation(self, waveform, sampling_rate):
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
        N = len(waveform)
        t_list = np.arange(0, N) / sampling_rate
        signal = waveform * np.exp(1j * 2 * np.pi * self.freq * t_list)

        # Compute the Fourier Transform and associated frequency bins.
        fft_values = np.fft.fft(signal)
        fft_freq = np.fft.fftfreq(N, d=1 / sampling_rate)

        # Apply the provided modification function to the FFT coefficients.
        modified_fft_values = self.recover_func(fft_freq) * fft_values

        # Compute the inverse FFT to get back the modified time-domain waveform.
        modified_signal = np.fft.ifft(modified_fft_values)
        # if np.max(modified_signal) > 2**15:
        #     modified_signal *= 2**15 / np.max(modified_signal)
        #     warnings.warn("pulse amplitude exceeded maxv")
        modified_waveform = modified_signal * np.exp(-1j * 2 * np.pi * self.freq * t_list)
        return modified_waveform

    def get_calibration_func(self, freq_ref, attenuation):
        from scipy.interpolate import CubicSpline
        if self.scale.lower() in ["db", "dbm", "log"]:
            freq_MHz = self.calibration_data[0]
            S21_dbm = self.calibration_data[1]
            S21_interpolate = CubicSpline(freq_MHz, S21_dbm + attenuation)
            interpolation = CubicSpline(freq_MHz, 10 ** ((-S21_dbm + S21_interpolate(freq_ref)) / 10))
        elif self.scale.lower() == "linear":
            freq_MHz = self.calibration_data[0]
            S21 = self.calibration_data[1]
            S21_interpolate = CubicSpline(freq_MHz, S21 + attenuation)
            interpolation = CubicSpline(freq_MHz, + S21_interpolate(freq_ref)/S21)
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

    def get_recover_func(self, freq_ref, attenuation):
        from scipy.interpolate import CubicSpline
        if self.scale.lower() in ["db", "dbm"]:
            freq_MHz = self.calibration_data[0]
            S21_dbm = self.calibration_data[1]
            S21_interpolate = CubicSpline(freq_MHz, S21_dbm + attenuation)
            interpolation = CubicSpline(freq_MHz, 10 ** ((S21_dbm - S21_interpolate(freq_ref)) / 10))
        elif self.scale.lower() == "linear":
            freq_MHz = self.calibration_data[0]
            S21 = self.calibration_data[1]
            S21_interpolate = CubicSpline(freq_MHz, S21 + attenuation)
            interpolation = CubicSpline(freq_MHz, + S21/S21_interpolate(freq_ref))

        def calib_func(val):
            f_min = np.min(freq_MHz)
            f_max = np.max(freq_MHz)
            val_array = np.atleast_1d(val)  # Ensure val is treated as a numpy array.
            mask = (val_array >= f_min) & (val_array <= f_max)  # Create a mask for values within the range.
            result = np.where(mask, interpolation(val_array), 1)  # For values within the calibrated range, use the interpolation function. Otherwise, return 1
            if result.size == 1:
                return result.item()  # Return a scalar if the input was a scalar.
            return result
        return calib_func

    def plot_calibration(self, ax=None):
        if ax is None:
            fig, ax = plt.subplots(1, 1)
        else:
            fig = ax.get_figure()
        ax.set_title("calibration data")
        ax.plot(self.calibration_data[0], self.calibration_data[1])
        ax.set_xlabel("Frequency (MHz)")
        return fig, ax



