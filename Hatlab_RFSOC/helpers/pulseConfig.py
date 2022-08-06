from qick.qick_asm import QickProgram
from typing import List, Literal


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
    gain_I = kwargs.pop("gain", None)
    gain_Q = int(gain_I * IQScale)
    phase_I = kwargs.pop("phase", None)
    phase_Q = prog.deg2reg(prog.reg2deg(phase_I, ch_I) + skewPhase, ch_Q)

    prog.set_pulse_registers(ch=ch_I, phase=phase_I, gain=gain_I, **kwargs)
    prog.set_pulse_registers(ch=ch_Q, phase=phase_Q, gain=gain_Q, **kwargs)


def declareMuxedGenAndReadout(prog: QickProgram, res_ch: int, res_nqz: Literal[1, 2], res_mixer_freq: float,
                              res_freqs: float, res_gains: List[float], ro_chs: List[int], readout_length: int):
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

