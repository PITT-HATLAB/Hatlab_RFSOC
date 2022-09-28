import numpy as np
from qick.qick_asm import QickProgram
from typing import List, Literal
import warnings


def tanh_box(length: int, ramp_width: int, cut_offset=0.01, maxv=30000):
    """
    Create a numpy array containing a smooth box pulse made of two tanh functions subtract from each other.

    :param length: Length of array (in points)
    :param ramp_width: number of points from cutOffset to 0.95 amplitude
    :param cut_offset: the initial offset to cut on the tanh Function
    :return:
    """
    warnings.warn(DeprecationWarning("This function will be removed later, use core.pulses.tanh_box instead"))
    x = np.arange(0, length)
    c0_ = np.arctanh(2 * cut_offset - 1)
    c1_ = np.arctanh(2 * 0.95 - 1)
    k_ = (c1_ - c0_) / ramp_width
    y = (0.5 * (np.tanh(k_ * x + c0_) - np.tanh(k_ * (x - length) - c0_)) - cut_offset) / (1 - cut_offset) * maxv
    return y


def add_tanh(prog: QickProgram, ch, name, length, ramp_width, cut_offset=0.01, maxv=None):
    """Adds a smooth box pulse made of two tanh functions to the waveform library.
    The pulse will peak at length/2.

    Parameters
    ----------
    ch : int
        DAC channel (index in 'gens' list)
    name : str
        Name of the pulse
    length : int
        Total pulse length (in units of fabric clocks)
    ramp_width : int
        Number of points from cut_offset to 0.95 amplitude (in units of fabric clocks)
    cut_offset: float
        the initial offset to cut on the tanh Function (in unit of unit-height pulse)
    maxv : float
        Value at the peak (if None, the max value for this generator will be used)

    """
    warnings.warn(DeprecationWarning("This function will be removed later, use core.pulses.add_tanh instead"))
    gencfg = prog.soccfg['gens'][ch]
    if maxv is None: maxv = gencfg['maxv'] * gencfg['maxv_scale']
    samps_per_clk = gencfg['samps_per_clk']

    length = np.round(length) * samps_per_clk
    ramp_width *= samps_per_clk

    prog.add_pulse(ch, name, idata=tanh_box(length, ramp_width, cut_offset, maxv=maxv))


def set_pulse_registers_IQ(prog: QickProgram, ch_I, ch_Q, skewPhase, IQScale, **kwargs):
    """ set the pulse register for two DAC channels that are going to be sent to a IQ mixer.
    :param prog: qick program for which the pulses will be added
    :param ch_I: DAC channel for I
    :param ch_Q: DAC channel for Q
    :param skewPhase: pretuned skewPhase value for the IQ mixer (deg)
    :param IQScale: pretuned IQ scale value for the IQ mixer
    :param kwargs: kwargs for "set_pulse_registers"
    :return:
    """

    # pop the two keys in ro_chs[channel_name] that will not be used here, so that we can pass the whole channel config
    # dict as keyword arguments, i.e. set_pulse_registers_IQ(**ro_chs[channel_name], **kwargs)
    kwargs.pop("nqz", None)
    kwargs.pop("ro_ch", None)

    gain_I = kwargs.pop("gain", None)
    gain_Q = int(gain_I * IQScale)
    phase_I = kwargs.pop("phase", None)
    phase_Q = prog.deg2reg(prog.reg2deg(phase_I, ch_I) + skewPhase, ch_Q)

    prog.set_pulse_registers(ch=ch_I, phase=phase_I, gain=gain_I, **kwargs)
    prog.set_pulse_registers(ch=ch_Q, phase=phase_Q, gain=gain_Q, **kwargs)


def declareMuxedGenAndReadout(prog: QickProgram, res_ch: int, res_nqz: Literal[1, 2], res_mixer_freq: float,
                              res_freqs: List[float], res_gains: List[float], ro_chs: List[int], readout_length: int):
    """ declare muxed generator and readout channels
    :param prog: qick program for which the channels will be declared
    :param res_ch: DAC channel for resonator
    :param res_nqz: resonator DAC nyquist zone, should consider mixer_freq+res_freqs
    :param res_mixer_freq: LO freq for digital up conversion (DUC) of the DAC channel
    :param res_freqs: DAC waveform frequencies for each muxed channel (IF of DUC)
    :param res_gains: gains of each muxed channel, float numbers between [-1, 1]
    :param ro_chs: ADC channels for readout
    :param readout_length: ADC readout length. In clock cycles

    :return:
    """

    # configure DACs
    prog.declare_gen(ch=res_ch, nqz=res_nqz, mixer_freq=res_mixer_freq,
                     mux_freqs=res_freqs,
                     ro_ch=ro_chs[0], mux_gains=res_gains)

    # configure the readout lengths and downconversion frequencies
    for iCh, ch in enumerate(ro_chs):
        prog.declare_readout(ch=ch, freq=res_freqs[iCh], length=readout_length,
                             gen_ch=res_ch)


def add_prepare_msmt(prog: QickProgram, q_drive_ch:str, q_pulse_cfg:dict, res_ch:str, syncdelay:float,
                     prepare_q_gain:int=None):
    """
    add a state preparation measurement to the qick asm program.

    :param prog:
    :param q_drive_ch:
    :param q_pulse_cfg:
    :param res_ch:
    :param syncdelay: time to wait after msmt, in us
    :param prepare_q_gain: q drive gain for the prepare pulse
    :return:
    """
    # todo: maybe move this to somewhere else

    if prepare_q_gain is None:
        prepare_q_gain = q_pulse_cfg["pi2_gain"]

    # play ~pi/n pulse to ensure ~50% selection rate.
    prog.set_pulse_params(q_drive_ch, style="arb", waveform=q_pulse_cfg["waveform"], phase=q_pulse_cfg.get("phase", 0),
                          freq=q_pulse_cfg["ge_freq"], gain=prepare_q_gain)
    prog.pulse(ch=prog.cfg["gen_chs"][q_drive_ch]["ch"])  # play gaussian pulse

    prog.sync_all(prog.us2cycles(0.05))  # align channels and wait 50ns

    # add measurement
    prog.measure(pulse_ch=prog.cfg["gen_chs"][res_ch]["ch"],
                 adcs=prog.ro_chs,
                 pins=[0],
                 adc_trig_offset=prog.cfg["adc_trig_offset"],
                 wait=True,
                 syncdelay=prog.us2cycles(syncdelay))
