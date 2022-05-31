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



def measure_with_IQch(prog:QickProgram, adcs, pulse_ch_I, pulse_ch_Q, pins=None,
                      adc_trig_offset=270, t='auto', wait=False, syncdelay=None):
    """ same as the measure method in qick_asm, but uses IQ channel for resonator drive
        TODO: we probably don't need this since the original measure method already support multiple DAC channels
    :param prog: qick program for which the pulses will be added
    :type prog: QickProgram
    :param pulse_ch_I: DAC channel
    :type pulse_ch_I: int or list
    :param pins: List of marker pins to pulse.
    :type pins: list
    :param adc_trig_offset: Offset time at which the ADC is triggered (in clock ticks)
    :type adc_trig_offset: int
    :param t: The number of clock ticks at which point the pulse starts
    :type t: int
    :param wait: Pause tProc execution until the end of the ADC readout window
    :type wait: bool
    :param syncdelay: The number of additional clock ticks to delay in the sync_all.
    :type syncdelay: int
    """
    prog.trigger(adcs, pins=pins, adc_trig_offset=adc_trig_offset)  # trigger the adc acquisition
    prog.pulse(ch=pulse_ch_I, t=0)
    prog.pulse(ch=pulse_ch_Q, t=0)
    if wait:
        prog.wait_all()
    if syncdelay is not None:
        prog.sync_all(syncdelay)
