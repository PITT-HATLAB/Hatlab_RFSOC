from typing import List, Literal

import numpy as np
from qick.qick_asm import QickProgram

def tanh_box(length: int, ramp_width: int, cut_offset=0.01, maxv=30000):
    """
    Create a numpy array containing a smooth box pulse made of two tanh functions subtract from each other.

    :param length: Length of array (in points)
    :param ramp_width: number of points from cutOffset to 0.95 amplitude
    :param cut_offset: the initial offset to cut on the tanh Function
    :return:
    """
    x = np.arange(0, length)
    c0_ = np.arctanh(2 * cut_offset - 1)
    c1_ = np.arctanh(2 * 0.95 - 1)
    k_ = (c1_ - c0_) / ramp_width
    y = (0.5 * (np.tanh(k_ * x + c0_) - np.tanh(k_ * (x - length) - c0_)) - cut_offset) / (
                1 - cut_offset) * maxv
    return y



def gaussian(sigma: int, length: int, maxv=30000):
    """
    Create a numpy array containing a Gaussian function.

    :param sigma: igma (standard deviation) of Gaussian
    :param length: total number of points of gaussian pulse
    :return:
    """
    x = np.arange(0, length)
    y = maxv * np.exp(-(x - length/2) ** 2 / sigma ** 2)
    return y


def add_tanh(prog: QickProgram, gen_ch, name, length:float, ramp_width:float, cut_offset:float=0.01, maxv=None):
    """Adds a smooth box pulse made of two tanh functions to the waveform library, using physical parameters of the pulse.
    The pulse will peak at length/2.

    Parameters
    ----------
    ch : int
        DAC channel (index in 'gens' list)
    name : str
        Name of the pulse
    length : float
        Total pulse length (in units of us)
    ramp_width : float
        ramping time from cut_offset to 0.95 amplitude (in units of us)
    cut_offset: float
        the initial offset to cut on the tanh Function (in unit of unit-height pulse)
    maxv : float
        Value at the peak (if None, the max value for this generator will be used)

    """

    gen_ch = prog.cfg["gen_chs"][gen_ch]["ch"]
    soc_gencfg = prog.soccfg['gens'][gen_ch]
    if maxv is None: maxv = soc_gencfg['maxv'] * soc_gencfg['maxv_scale']
    samps_per_clk = soc_gencfg['samps_per_clk']
    fclk = soc_gencfg['f_fabric']

    # length_cyc = prog.us2cycles(length, gen_ch=gen_ch)
    length_cyc = length * fclk
    length_reg = length_cyc * samps_per_clk
    # ramp_reg = np.int64(np.round(ramp_width*fclk*samps_per_clk))
    ramp_reg = ramp_width * fclk * samps_per_clk

    wf = tanh_box(length_reg, ramp_reg, cut_offset, maxv=maxv)
    zero_padding = np.zeros((16-len(wf))%16)
    wf_padded = np.concatenate((wf, zero_padding))

    prog.add_pulse(gen_ch, name, idata=wf_padded)



def add_gaussian(prog: QickProgram, gen_ch:str, name, sigma:float, length:float, maxv=None):
    """Adds a gaussian pulse to the waveform library, using physical parameters of the pulse.
    The pulse will peak at length/2.

    Parameters
    ----------
    ch : str
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
    if maxv is None: maxv = soc_gencfg['maxv'] * soc_gencfg['maxv_scale']
    samps_per_clk = soc_gencfg['samps_per_clk']
    fclk = soc_gencfg['f_fabric']

    # length_cyc = prog.us2cycles(length, gen_ch=gen_ch)
    length_cyc = length * fclk
    length_reg = length_cyc * samps_per_clk
    sigma_reg = sigma * fclk * samps_per_clk

    wf = gaussian(sigma_reg, length_reg, maxv=maxv)
    zero_padding = np.zeros((16-len(wf))%16)
    wf_padded = np.concatenate((wf, zero_padding))

    prog.add_pulse(gen_ch, name, idata=wf_padded)



