from typing import List, Literal, Union

import numpy as np
from qick.qick_asm import QickProgram

NumType = Union[int, float]

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
    return y - np.min(y)



def gaussian(sigma: int, length: int, maxv=30000):
    """
    Create a numpy array containing a Gaussian function.

    :param sigma: sigma (standard deviation) of Gaussian
    :param length: total number of points of gaussian pulse
    :return:
    """
    x = np.arange(0, length)
    y = maxv * np.exp(-(x - length/2) ** 2 / sigma ** 2)
    y = y - np.min(y)
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
    
    if isinstance(padding, int|float):
        padding = np.array([0, padding])
    else:
        padding = np.array(padding)
    padding_samp = np.ceil(padding * fclk * samps_per_clk)
    data = np.concatenate((np.zeros(int(padding_samp[0])), data, np.zeros(int(padding_samp[1]))))

    return data

def add_tanh(prog: QickProgram, gen_ch, name, length:float, ramp_width:float, cut_offset:float=0.01, phase: float=0,
             maxv=None, padding: Union[NumType, List[NumType]]=None, drag: float=0):
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
    padding: float | List[float]
        padding zeros in front of and at the end of the pulse

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
    if padding is not None:
        wf = add_padding(wf, soc_gencfg, padding)
    zero_padding = np.zeros((16-len(wf))%16)
    wf_padded = np.concatenate((wf, zero_padding))
    drag_padded = -np.gradient(wf_padded) * drag

    wf_idata = np.cos(np.pi / 180 * phase) * wf_padded - np.sin(np.pi / 180 * phase) * drag_padded
    wf_qdata = np.sin(np.pi / 180 * phase) * wf_padded + np.cos(np.pi / 180 * phase) * drag_padded

    # prog.add_pulse(gen_ch, name, idata=wf_padded)
    prog.add_pulse(gen_ch, name, idata=wf_idata, qdata=wf_qdata)



def add_gaussian(prog: QickProgram, gen_ch:str, name, sigma:float, length:float, phase: float=0,
                 maxv=None, padding: Union[NumType, List[NumType]]=None, drag: float=0):
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
    if padding is not None:
        wf = add_padding(wf, soc_gencfg, padding)
    zero_padding = np.zeros((16-len(wf))%16)
    wf_padded = np.concatenate((wf, zero_padding))
    drag_padded = -np.gradient(wf_padded) * drag

    wf_idata = np.cos(np.pi / 180 * phase) * wf_padded - np.sin(np.pi / 180 * phase) * drag_padded
    wf_qdata = np.sin(np.pi / 180 * phase) * wf_padded + np.cos(np.pi / 180 * phase) * drag_padded

    # prog.add_pulse(gen_ch, name, idata=wf_padded)
    prog.add_pulse(gen_ch, name, idata=wf_idata, qdata=wf_qdata)

def add_pulse_concatenate(prog: QickProgram, gen_ch: str|int, name, gatelist, maxv=None):
    gen_ch = prog.cfg["gen_chs"][gen_ch]["ch"] if type(gen_ch)==str else gen_ch
    soc_gencfg = prog.soccfg['gens'][gen_ch]
    if maxv is None: maxv = soc_gencfg['maxv'] * soc_gencfg['maxv_scale']
    samps_per_clk = soc_gencfg['samps_per_clk']
    fclk = soc_gencfg['f_fabric']
    
    wfdata_i = []
    wfdata_q = []
    wf_len_list = []
    for gate in gatelist:
        maxv_p = gate.get('maxv', maxv)
        if gate['shape'] == 'gaussian':
            length_reg = gate['length'] * fclk * samps_per_clk
            sigma_reg = gate['sigma'] * fclk * samps_per_clk
            pulsedata = gaussian(sigma_reg, length_reg, maxv=maxv_p)

        elif gate['shape'] == 'tanh_box':
            length_reg = gate['length'] * fclk * samps_per_clk
            ramp_reg = gate['ramp_width'] * fclk * samps_per_clk
            pulsedata = tanh_box(length_reg, ramp_reg, maxv=maxv_p)

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
        # wf_len_list.append(len(wfdata_i) / 16.0)
        
    # print(wf_len_list)
    prog.add_pulse(gen_ch, name, idata=wfdata_i, qdata=wfdata_q)
    
