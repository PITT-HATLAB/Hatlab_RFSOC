from qick.qick_asm import QickProgram


def set_pulse_registers_IQ(prog:QickProgram, ch_I, ch_Q, skewPhase, IQScale, **kwargs):
    """ set the pulse register for two DAC channels that are going to be sent to a IQ mixer.
    TODO: this should be moved to QickProgram
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
