from typing import Dict, List, Union, Callable, Literal, Tuple
import warnings

from tqdm import tqdm
import numpy as np

from qick.qick_asm import QickProgram, FullSpeedGenManager, QickRegister, QickRegisterManagerMixin
from qick.averager_program import AbsQickSweep, QickSweep, NDAveragerProgram

from .pulses import add_gaussian, add_tanh

RegisterTypes = Literal["freq", "time", "phase", "adc_freq"]

class FlatTopLengthSweep(QickSweep):
    """
    Currently, the register that controls the flat part length of a flat_top pulse is packed in the last 16 bit of
    "mode" register. So we need some additional treatment to the initial and step values here.
    """

    def __init__(self, prog: QickProgram, mode_reg: QickRegister, start, stop, expts: int,
                 t_wait_reg: QickRegister = None, label=None):
        """
        initialize a QickSweep object for sweeping the flat part length of a flat_top pulse. The register used for this
        sweep needs to the "mode" register of a gen channel. If t_wait_reg is provided, it's value will also be updated/
        reset when each time the pulse flat part length is updated/reset.

        :param prog: QickProgram in which the sweep happens.
        :param mode_reg: mode register associated to the generator channel in which the pulse length will be swept.
        :param start: start value of the flat part length, in us.
        :param stop: stop value of the flat part length, in us.
        :param expts: number of experiment points between start and stop value.
        :param t_wait_reg: QickRegister object associated to t_proc, which is used for waiting for the amount of time
            that equals to flat part length of the pulse.
        :param label: label to be used for the loop tag in qick asm program.
        """
        super().__init__(prog, mode_reg, start, stop, expts, label)
        self.t_wait_reg = t_wait_reg
        # check length validity
        min_l, max_l = prog.cycles2us(2, self.reg.gen_ch), prog.cycles2us(2 ** 16, self.reg.gen_ch)
        for pl in [start, stop]:
            if pl >= max_l or pl <= min_l:
                raise RuntimeError(f"flat part length must be longer then {min_l} us, and shorter than {max_l} us")

        # overwrite the initial and step value in base class
        self.step_val = (stop - start) / (expts - 1)
        self.reg_step = prog.us2cycles(self.step_val, self.reg.gen_ch)
        if self.reg_step == 0:
            warnings.warn(RuntimeWarning(f"sweep step for register {self.reg.name} is 0"))

        reg_start = prog.us2cycles(start, self.reg.gen_ch)
        gen_mgr = prog._gen_mgrs[self.reg.gen_ch]
        self.reg.init_val = gen_mgr.get_mode_code(length=reg_start, mode="oneshot", outsel="dds")

    def update(self):
        self.prog.mathi(self.reg.page, self.reg.addr, self.reg.addr, '+', self.reg_step)
        # also sweep the wait time register in t_proc if provided
        if self.t_wait_reg is not None:
            self.t_wait_reg.set_to(self.t_wait_reg, '+', self.step_val)

    def reset(self):
        self.reg.reset()
        if self.t_wait_reg is not None:
            self.t_wait_reg.reset()


class FlatTopGainSweep(QickSweep):
    """
    Currently, the gain of the flat part of the flat_top pulse is controlled by a different register "gain2". So when we
    sweep the gain of a flat_top pulse, gain2 needs to be swept at the same time.
    """

    def __init__(self, prog: QickProgram, gen_ch: str, start, stop, expts: int, label=None):
        """
        initialize a QickSweep object for sweeping the overall gain of a flat_top pulse. For convenience, generator
        channel is passed so that both "gain" and "gain2" registers will be found automatically.

        :param prog: QickProgram in which the sweep happens.
        :param gen_ch: generator channel name for which the flat_top pulse gain will be swept
        :param start: start value of the flat part length, in us.
        :param stop: stop value of the flat part length, in us.
        :param expts: number of experiment points between start and stop value.
        :param label: label to be used for the loop tag in qick asm program.
        """
        self.gain_reg = prog.get_gen_reg(gen_ch, "gain")
        super().__init__(prog, self.gain_reg, start, stop, expts, label)

        # flat part gain
        self.gain2_reg = prog.get_gen_reg(gen_ch, "gain2")
        self.gain2_reg.init_val = start // 2
        self.gain2_step = self.step_val // 2

        if type(prog._gen_mgrs[self.gain_reg.gen_ch]) != FullSpeedGenManager:
            raise NotImplementedError("gain sweep for flat top pulse of non-FullSpeedGen is not implemented yet")

    def update(self):
        # update both gain and gain2
        # self.prog.mathi(self.gain2_reg.page, self.gain2_reg.addr, self.gain2_reg.addr, '+', self.gain2_step)
        self.gain2_reg.set_to(self.gain2_reg, '+', self.gain2_step)
        super().update()

    def reset(self):
        # reset both gain and gain2
        self.gain2_reg.reset()
        super().reset()


class APAveragerProgram(QickRegisterManagerMixin, QickProgram):
    """
    APAveragerProgram class. "A" and "P" stands for "Automatic" and "Physical". This class automatically declares the
    generator and readout channels using the parameters provided in cfg["gen_chs"] and cfg["ro_chs"], and contains
    functions that hopefully can make it easier to program pulse sequences with parameters in their physical units
    (so that we don't have to constantly call "_2reg"/"_2cycles" functions).
    The "acquire" methods are copied from qick.AveragerProgram.

    config requirements:
    "gen_chs" = dictionary that contains the configuration of each generator channel;
        The format should be: {"gen_name": {**kwargs_of_declare_gen}}
    "ro_chs" = dictionary that contains the configuration of each readout channel;
        The format should be: {"ro_name": {**kwargs_of_declare_readout}}
    "waveforms"(optional) =  dictionary that contains some waveforms and their parameters (in physical units)
        The format should be: {"waveform_name": {**kwargs_of_add_waveform}}
    """

    def __init__(self, soccfg, cfg):
        """
        Initialize the QickProgram and automatically declares all generator and readout channels in cfg["gen_chs"] and
        cfg["ro_chs"]
        """
        super().__init__(soccfg)
        self.cfg = cfg
        self.reps = cfg["reps"]
        if "soft_avgs" in cfg:
            self.rounds = cfg['soft_avgs']
        if "rounds" in cfg:
            self.rounds = cfg['rounds']
        self.expts = None  # abstract variable for total number of experiments in each repetition.
        self.readout_per_exp = None  # software counter for number of readouts per experiment.
        self.declare_all_gens()
        self.declare_all_readouts()

    def declare_all_gens(self):
        """
        Declare all generators in the config dict based on the items specified in cfg["gen_chs"].

        :return:
        """
        for gen_ch, kws in self.cfg["gen_chs"].items():
            try:
                chs = [int(kws["ch"])]
            except TypeError:
                chs = kws["ch"]
            declare_kws = {}
            exclude_args = ["ch", "skew_phase", "IQ_scale"] # for cases when two DACs are used as IQ channels on a mixer
            for arg, v in kws.items():
                if not (arg in exclude_args):
                    declare_kws[arg] = v
            for ch in chs:
                self.declare_gen(ch, **declare_kws) # todo: all the other functions doesn't support IQ channel gen yet.. e.g. set_pulse_params, get_reg, etc

    def declare_all_readouts(self):
        """
        Declare all readout channels in the config dict based on the items specified in cfg["ro_chs"].

        :return:
        """
        ro_chs = self.cfg.get("ro_chs", {})
        for ro_ch, kws in ro_chs.items():
            # self.declare_readout(**kws)
            ch = int(kws["ch"])
            if self.soccfg['readouts'][ch].get('tproc_ctrl') is None:
                self.declare_readout(**kws)
            else:
                self.declare_readout(ch=ch, length=kws["length"])
                freq_ro = self.freq2reg_adc(kws["freq"], ro_ch=ch, gen_ch=kws["gen_ch"])
                self.set_readout_registers(ch=ch, freq=freq_ro, length=kws["length"], # The length here actually doesn't matter
                                           mode='oneshot', outsel='product', phrst=kws.get("phrst",0))


    def get_gen_reg(self, gen_ch: Union[str, int], name: str) -> QickRegister:
        """
        Gets tProc register page and address associated with gen_ch and register name. Creates a QickRegister object for
        return.

        :param gen_ch: name of the generator channel, as in cfg["gen_chs"]. Or generator channel number
        :param name:  name of the qick register, as in QickProgram.pulse_registers
        :return: QickRegister
        """
        if type(gen_ch) == str:
            ch_num = self.cfg["gen_chs"][gen_ch]["ch"]
        elif type(gen_ch) == int:
            ch_num = gen_ch
        return super().get_gen_reg(ch_num, name)

    def new_gen_reg(self, gen_ch: Union[str, int], name: str = None, init_val=None, reg_type: RegisterTypes = None,
                tproc_reg=False) -> QickRegister:
        """
        Declare a new register in the generator register page. Address automatically adds 1 one when each time a new
        register in the same page is declared.

        :param gen_ch: name of the generator channel, as in cfg["gen_chs"]. Or generator channel number
        :param name: name of the new register. Optional.
        :param init_val: initial value for the register, when reg_type is provided, the reg_val should be in the unit of
            the corresponding type.
        :param reg_type: type of the register, e.g. freq, time, phase.
        :param tproc_reg: if true, the new register created will not be associated to a specific generator or readout
            channel. It will still be on the same page as the gen_ch for math calculations. This is usually used for a
            time register in t_processor, where we want to calculate "us2cycles" with the t_proc fabric clock rate
            instead of the generator clock rate.
        :return: QickRegister
        """
        if type(gen_ch) == str:
            ch_num = self.cfg["gen_chs"][gen_ch]["ch"]
        elif type(gen_ch) == int:
            ch_num = gen_ch
        return super().new_gen_reg(ch_num, name, init_val, reg_type, tproc_reg)

    def pulse_param_to_reg(self, gen_ch, gen_ro_ch=None, **pulse_param):
        """
        converts some pulse parameters from physical values to regs
        :param gen_cfg: generator config
        :param pulse_param: kwargs in set_pulse_params
        :return:
        """
        pulse_reg = pulse_param.copy()
        if "freq" in pulse_param:
            pulse_reg["freq"] = self.soccfg.freq2reg(pulse_param["freq"], gen_ch, gen_ro_ch)
        if "phase" in pulse_param:
            pulse_reg["phase"] = self.soccfg.deg2reg(pulse_param["phase"], gen_ch)
        if "length" in pulse_param:
            pulse_reg["length"] = self.soccfg.us2cycles(pulse_param["length"], gen_ch)
        return pulse_reg

    def set_pulse_params(self, gen_ch: str, **kwargs):
        """
        This is a wrapper of the QickProgram.set_pulse_registers. Instead of taking register values, this function takes
        the physical values of the pulse parameters. E.g. freq in MHz, length in us, phase in degree.

        Parameters
        ----------
        gen_ch : str
            name of the DAC channel
        style : str
            Pulse style ("const", "arb", "flat_top")
        freq : float
            Frequency (MHz)
        phase : float
            Phase (deg)
        gain : int
            Gain (DAC units)
        phrst : int
            If 1, it resets the phase coherent accumulator
        stdysel : str
            Selects what value is output continuously by the signal generator after the generation of a pulse. If "last", it is the last calculated sample of the pulse. If "zero", it is a zero value.
        mode : str
            Selects whether the output is "oneshot" or "periodic"
        outsel : str
            Selects the output source. The output is complex. Tables define envelopes for I and Q. If "product", the output is the product of table and DDS. If "dds", the output is the DDS only. If "input", the output is from the table for the real part, and zeros for the imaginary part. If "zero", the output is always zero.
        length : float
            length of the constant pulse or the flat part of the flat_top pulse, in us
        waveform : str
            Name of the envelope waveform loaded with add_pulse(), used for "arb" and "flat_top" styles
        mask : list of int
            for a muxed signal generator, the list of tones to enable for this pulse
        """
        gen_cfg = self.cfg["gen_chs"][gen_ch]
        kw_reg = self.pulse_param_to_reg(gen_cfg["ch"], gen_cfg.get("ro_ch"), **kwargs)
        self.set_pulse_registers(gen_cfg["ch"], **kw_reg)

    def add_waveform(self, gen_ch, name, shape, **kwargs):
        """
        Add waveform to a generator channel based on parameters specified in cfg["waveforms"]

        :param gen_ch: name of the generator channel, as in cfg["gen_chs"]
        :param name: name of the waveform.
        :param shape: shape of the waveform, should be one of the waveforms that are available in pulses.py
        :param kwargs: kwargs for the pulse

        :return:
        """

        # todo: the parser can be better... instead of using if commands to select from only two possible waveforms here
        #   We should have a abstract waveform class, each new waveform should be written as a waveform class instance,
        #   and the parser should search for waveform in pulses.py (rename that to waveforms.py)

        if shape == "gaussian":
            add_gaussian(self, gen_ch, name, **kwargs)
        elif shape == "tanh_box":
            add_tanh(self, gen_ch, name, **kwargs)
        else:
            raise NameError(f"unsupported pulse shape {shape}")

    def add_waveform_from_cfg(self, gen_ch: str, name: str):
        """
        Add waveform to a generator channel based on parameters specified in cfg["waveforms"]

        :param gen_ch: name of the generator channel, as in cfg["gen_chs"]
        :param name: name of the waveform, as in cfg["waveforms"]
        :return:
        """
        pulse_params = self.cfg["waveforms"][name]
        self.add_waveform(gen_ch, name, **pulse_params)

    def get_expt_pts(self):
        """
        Abstract method for calculating total experiment points in each repetition based on the config.
        """
        pass

    def acquire_decimated(self, soc, load_pulses=True, readouts_per_experiment=1, start_src="internal", progress=True, debug=False):
        """
        Copied from qick.AveragerProgram
        """
        buf = super().acquire_decimated(soc, reads_per_rep=readouts_per_experiment, load_pulses=load_pulses, start_src=start_src, progress=progress, debug=debug)
        # move the I/Q axis from last to second-last
        return [np.moveaxis(d, -1, -2) for d in buf]

    def measure(self, adcs, pulse_ch=None, pins=None, adc_trig_offset=270, t='auto', wait=False, syncdelay=None,
                add_count=True):
        """Wrapper method that combines an ADC trigger, a pulse, and (optionally) the appropriate wait and a sync_all.
        You must have already run set_pulse_registers for this channel. the readout count automatically adds one on each
        time "measure" is called.

        If you use wait=True, it's recommended to also specify a nonzero syncdelay.

        Parameters
        ----------
        adcs : list of int
            ADC channels (index in 'readouts' list)
        pulse_ch : int or list of int
            DAC channel(s) (index in 'gens' list)
        pins : list of int, optional
            refer to trigger()
        adc_trig_offset : int, optional
            refer to trigger()
        t : int, optional
            refer to pulse()
        wait : bool, optional
            Pause tProc execution until the end of the ADC readout window
        syncdelay : int, optional
            The number of additional tProc cycles to delay in the sync_all
        add_count : bool, optional
            when true, the readout counter register adds one after the readout
        """
        super().measure(adcs, pulse_ch, pins, adc_trig_offset, t, wait, syncdelay)
        # automatically adds one to the readout count register.
        if add_count:
            if self.readout_per_exp is None:
                self.readout_per_exp = 1
            else:
                self.readout_per_exp += 1
    
    def reset_ts(self):
        """
        Reset the soft accumulated dac and adc timestamps.
        This is usually automatically done in sync_all(). However, when the pulse_length/trigger_time is controlled by 
        FPGA registers, the soft counted timestamp will not work, and we have to call this manually.
        :return: 
        """
        # Timestamps, for keeping track of pulse and readout end times.
        self._dac_ts = [0] * len(self._dac_ts)
        self._adc_ts = [0] * len(self._adc_ts)


class NDAveragerProgram(APAveragerProgram):
    """
    NDAveragerProgram class, for experiments that sweep over multiple variables in qick. The order of experiment runs
    follow outer->inner: reps, sweep_n,... sweep_0.

    :param cfg: Configuration dictionary
    :type cfg: dict
    """

    def __init__(self, soccfg, cfg):
        """
        Constructor for the NDAveragerProgram. Make the ND sweep asm commands.
        """
        super().__init__(soccfg, cfg)
        self.qick_sweeps: List[AbsQickSweep] = []
        self.expts = 1
        self.sweep_axes = []
        self.make_program()

    def initialize(self):
        """
        Abstract method for initializing the program. Should include the instructions that will be executed once at the
        beginning of the qick program.
        """
        pass

    def body(self):
        """
        Abstract method for the body of the program.
        """
        pass

    def add_sweep(self, sweep: AbsQickSweep):
        """
        Add a layer of register sweep to the qick asm program. The order of sweeping will follow first added first sweep.
        :param sweep:
        :return:
        """
        self.qick_sweeps.append(sweep)
        self.expts *= sweep.expts
        self.sweep_axes.append(sweep.expts)

    def make_program(self):
        """
        Make the N dimensional sweep program. The program will run initialize once at the beginning, then iterate over
        all the sweep parameters and run the body. The whole sweep will repeat for cfg["reps"] number of times.
        """
        p = self

        p.initialize()  # initialize only run once at the very beginning

        rcount = 13  # total run counter
        rep_count = 14  # repetition counter

        n_sweeps = len(self.qick_sweeps)
        if n_sweeps > 7:  # to be safe, only register 15-21 in page 0 can be used as sweep counters
            raise OverflowError(f"too many qick inner loops ({n_sweeps}), run out of counter registers")
        counter_regs = (np.arange(n_sweeps) + 15).tolist()  # not sure why this has to be a list (np.array doesn't work)

        p.regwi(0, rcount, 0)  # reset total run count

        # set repetition counter and tag
        p.regwi(0, rep_count, self.cfg["reps"] - 1)
        p.label("LOOP_rep")

        # add reset and start tags for each sweep
        for creg, swp in zip(counter_regs[::-1], self.qick_sweeps[::-1]):
            swp.reset()
            p.regwi(0, creg, swp.expts - 1)
            p.label(f"LOOP_{swp.label if swp.label is not None else creg}")

        # run body and total_run_counter++
        p.body()
        p.mathi(0, rcount, rcount, "+", 1)
        p.memwi(0, rcount, 1)

        # add update and stop condition for each sweep
        for creg, swp in zip(counter_regs, self.qick_sweeps):
            swp.update()
            p.loopnz(0, creg, f"LOOP_{swp.label if swp.label is not None else creg}")

        # stop condition for repetition
        p.loopnz(0, rep_count, 'LOOP_rep')

        p.end()

    def get_expt_pts(self):
        """
        :return:
        """
        sweep_pts = []
        for swp in self.qick_sweeps:
            sweep_pts.append(swp.get_sweep_pts())
        return sweep_pts

    def acquire(self, soc, threshold: int = None, angle: List = None, load_pulses=True, readouts_per_experiment=None,
                save_experiments: List = None, start_src: str = "internal", progress=False, debug=False):
        """
        This method optionally loads pulses on to the SoC, configures the ADC readouts, loads the machine code
        representation of the AveragerProgram onto the SoC, starts the program and streams the data into the Python,
        returning it as a set of numpy arrays.
        Note here the buf data has "reps" as the outermost axis, and the first swept parameter corresponds to the
        innermost axis.

        config requirements:
        "reps" = number of repetitions;

        :param soc: Qick object
        :param threshold: threshold
        :param angle: rotation angle
        :param readouts_per_experiment: readouts per experiment
        :param save_experiments: saved readouts (by default, save all readouts)
        :param load_pulses: If true, loads pulses into the tProc
        :param start_src: "internal" (tProc starts immediately) or "external" (each round waits for an external trigger)
        :param progress: If true, displays progress bar
        :param debug: If true, displays assembly code for tProc program
        :returns:
            - expt_pts (:py:class:`list`) - list of experiment points
            - avg_di (:py:class:`list`) - list of lists of averaged accumulated I data for ADCs 0 and 1
            - avg_dq (:py:class:`list`) - list of lists of averaged accumulated Q data for ADCs 0 and 1
        """

        self.shot_angle = angle
        self.shot_threshold = threshold

        if readouts_per_experiment is None:
            readouts_per_experiment = self.readout_per_exp

        if save_experiments is None:
            save_experiments = range(readouts_per_experiment)

        # avg_d calculated in QickProgram.acquire() assumes a different data shape, here we will recalculate based on
        # the d_buf returned.
        d_buf, avg_d, shots = super().acquire(soc, reads_per_rep=readouts_per_experiment, load_pulses=load_pulses,
                                              start_src=start_src, progress=progress, debug=debug)

        # reformat the data into separate I and Q arrays
        # save results to class in case you want to look at it later or for analysis
        self.di_buf = d_buf[:, :, 0]
        self.dq_buf = d_buf[:, :, 1]

        if threshold is not None:
            self.shots = shots

        expt_pts = self.get_expt_pts()

        n_ro = len(self.ro_chs)
        avg_di = np.zeros((n_ro, len(save_experiments), self.expts))
        avg_dq = np.zeros((n_ro, len(save_experiments), self.expts))

        for nn, ii in enumerate(save_experiments):
            for i_ch, (ch, ro) in enumerate(self.ro_chs.items()):
                avg_di[i_ch][nn] = avg_d[i_ch, ii, ..., 0]
                avg_dq[i_ch][nn] = avg_d[i_ch, ii, ..., 1]

        self.di_buf_p = self.di_buf.reshape(n_ro, self.reps, -1)
        self.dq_buf_p = self.dq_buf.reshape(n_ro, self.reps, -1)

        return expt_pts, avg_di, avg_dq

    def _average_buf(self, d_reps, reads_per_rep: int):
        """
        overwrites the default _average_buf method in QickProgram. Here "reps" is the outermost axis, and we reshape
        avg_d to the shape of the sweep axes.
        :param d_reps:
        :param reads_per_rep:
        :return:
        """
        avg_d = np.zeros((len(self.ro_chs), reads_per_rep, self.expts, 2))
        for ii in range(reads_per_rep):
            for i_ch, (ch, ro) in enumerate(self.ro_chs.items()):
                avg_d[i_ch][ii] = np.sum(d_reps[i_ch][ii::reads_per_rep, :].reshape((self.reps, self.expts, 2)),
                                         axis=0) / (self.reps * ro['length'])
        return avg_d


class QubitMsmtMixin:
    def set_pulse_params_IQ(self: QickProgram, gen_ch:str, skew_phase, IQ_scale, **kwargs):
        """ set the pulse register for two DAC channels that are going to be sent to a IQ mixer.
        :param self: qick program for which the pulses will be added
        :param gen_ch: IQ generator channel name
        :param skew_phase: pre-tuned skewPhase value for the IQ mixer (deg)
        :param IQ_scale: pre-tuned IQ scale value for the IQ mixer
        :param kwargs: kwargs for "set_pulse_params"
        :return:
        """
        ch_I, ch_Q = self.cfg["gen_chs"][gen_ch]["ch"]
        gain_I = kwargs.pop("gain", None)
        gain_Q = int(gain_I * IQ_scale)
        phase_I = kwargs.pop("phase", None)
        phase_Q = phase_I + skew_phase

        gen_ro_ch = self.cfg["gen_chs"][gen_ch].get("ro_ch")
        I_regs = self.pulse_param_to_reg(ch_I, gen_ro_ch, phase=phase_I, gain=gain_I, **kwargs)
        Q_regs = self.pulse_param_to_reg(ch_Q, gen_ro_ch, phase=phase_Q, gain=gain_Q, **kwargs)

        self.set_pulse_registers(ch=ch_I, **I_regs)
        self.set_pulse_registers(ch=ch_Q, **Q_regs)

    def set_pulse_params_auto_gen_type(self:QickProgram, gen_ch:str, **pulse_args):
        """
        set pulse params based on the generator type. feed "skew_phase" and "IQ_scale" for IQ channels; auto add mask
        for muxed channels
        :param gen_ch: generator channel name
        :param pulse_args: pulse params
        :return:
        """
        gen_params = self.cfg["gen_chs"][gen_ch]
        # set readout pulse registers
        if "skew_phase" in gen_params: # IQ channel
            pulse_args["skew_phase"] = gen_params["skew_phase"]
            pulse_args["IQ_scale"] = gen_params["IQ_scale"]
            self.set_pulse_params_IQ(gen_ch, **pulse_args)
        else:
            if ("mask" in self._gen_mgrs[gen_params["ch"]].PARAMS_REQUIRED["const"]) and ("mask" not in gen_params):
                pulse_args["mask"] = [0, 1, 2, 3]
            self.set_pulse_params(gen_ch, **pulse_args)


    def add_prepare_msmt(self:QickProgram, q_drive_ch: str, q_pulse_cfg: dict, res_ch: str, syncdelay: float,
                         prepare_q_gain: int = None, adcs=None):
        """
        add a state preparation measurement to the qick asm program.

        :param self:
        :param q_drive_ch: Qubit drive channel name
        :param q_pulse_cfg: Qubit drive pulse_cfg, should be the "q_pulse_cfg" in the yml file, which contains the
            "ge_freq".
        :param res_ch: Resonator drive channel name
        :param syncdelay: time to wait after msmt, in us
        :param prepare_q_gain: q drive gain for the prepare pulse
        :return:
        """
        if prepare_q_gain is None:
            prepare_q_gain = int(q_pulse_cfg["pi2_gain"] * 0.75)
            
        q_pulse_cfg_ = dict(style="arb", waveform=q_pulse_cfg["waveform"],
                              phase=q_pulse_cfg.get("phase", 0), freq=q_pulse_cfg["ge_freq"], gain=prepare_q_gain)

        self.add_prepare_msmt_general(q_drive_ch, q_pulse_cfg_, res_ch, syncdelay, adcs)



    def add_prepare_msmt_general(self:QickProgram, q_drive_ch: str, q_pulse_cfg: dict, res_ch: str, syncdelay: float,
                         adcs=None):
        """
        add a state preparation measurement to the qick asm program.

        :param self:
        :param q_drive_ch: Qubit drive channel name
        :param q_pulse_cfg: Qubit drive pulse_cfg
        :param res_ch: Resonator drive channel name
        :param syncdelay: time to wait after msmt, in us
        :param prepare_q_gain: q drive gain for the prepare pulse
        :return:
        """
        
        # play ~pi/n pulse to ensure ~50% selection rate.
        self.set_pulse_params(q_drive_ch, **q_pulse_cfg)
        self.pulse(ch=self.cfg["gen_chs"][q_drive_ch]["ch"])  # play gaussian pulse

        self.sync_all(self.us2cycles(0.05))  # align channels and wait 50ns

        # add measurement
        self.measure(pulse_ch=self.cfg["gen_chs"][res_ch]["ch"],
                     adcs=adcs if adcs is not None else self.ro_chs,
                     pins=[0],
                     adc_trig_offset=self.cfg["adc_trig_offset"],
                     wait=True,
                     syncdelay=self.us2cycles(syncdelay))

    def add_prepare_msmt_with_amps(self: QickProgram, q_drive_ch: str, q_pulse_cfg: dict, res_chs: list[int], syncdelay: float,
                         prepare_q_gain: int = None, adcs=None):
        """
        add a state preparation measurement to the qick asm program.

        :param self:
        :param q_drive_ch: Qubit drive channel name
        :param q_pulse_cfg: Qubit drive pulse_cfg, should be the "q_pulse_cfg" in the yml file, which contains the
            "ge_freq".
        :param res_chs: Resonator drive channels in a list. This supports turning on all the amp channels as well
        :param syncdelay: time to wait after msmt, in us
        :param prepare_q_gain: q drive gain for the prepare pulse
        :return:
        """
        if prepare_q_gain is None:
            prepare_q_gain = q_pulse_cfg["pi2_gain"]

        q_pulse_cfg_ = dict(style="arb", waveform=q_pulse_cfg["waveform"],
                            phase=q_pulse_cfg.get("phase", 0), freq=q_pulse_cfg["ge_freq"], gain=prepare_q_gain)

        self.add_prepare_msmt_general_with_amps(q_drive_ch, q_pulse_cfg_, res_chs, syncdelay, adcs)

    def add_prepare_msmt_general_with_amps(self: QickProgram, q_drive_ch: str, q_pulse_cfg: dict, res_chs: list[int], syncdelay: float,
                                 adcs=None):
        """
        add a state preparation measurement to the qick asm program.

        :param self:
        :param q_drive_ch: Qubit drive channel name
        :param q_pulse_cfg: Qubit drive pulse_cfg
        :param res_chs: Resonator drive channels dictionary
        :param syncdelay: time to wait after msmt, in us
        :param prepare_q_gain: q drive gain for the prepare pulse
        :return:
        """

        # play ~pi/n pulse to ensure ~50% selection rate.
        self.set_pulse_params(q_drive_ch, **q_pulse_cfg)
        self.pulse(ch=self.cfg["gen_chs"][q_drive_ch]["ch"])  # play gaussian pulse

        self.sync_all(self.us2cycles(0.05))  # align channels and wait 50ns

        # add measurement
        self.measure(pulse_ch=res_chs,
                     adcs=adcs if adcs is not None else self.ro_chs,
                     pins=[0],
                     adc_trig_offset=self.cfg["adc_trig_offset"],
                     wait=True,
                     syncdelay=self.us2cycles(syncdelay))


    def add_tomo(self:QickProgram, core:Callable, q_drive_ch: str, q_pulse_cfg: dict, res_ch: str, syncdelay: float,
                 ro_ch=None, phase_off=0):
        """
        add qubit tomography msmts after the core experiment

        :param core: core part of the experiment
        :param q_drive_ch: Qubit drive channel name
        :param q_pulse_cfg: Qubit drive pulse_cfg
        :param res_ch: Resonator drive channel name
        :param syncdelay: time to wait after msmt, in us
        :param ro_ch: readout channel. By default, all readout channels will be trigger.
        :param phase_off: phase offset for the tomography, in deg
        :return:
        """
        pi2_gain = q_pulse_cfg["pi2_gain"]
        if ro_ch is None:
            ro_ch = self.ro_chs
        tomo_phases = phase_off + np.array([-90, 0, 0])
        for phase_t, gain_t in zip(tomo_phases, [pi2_gain, pi2_gain, 0]):
            # perform core experiment
            core()

            # play tomo pulse
            self.set_pulse_params(q_drive_ch, style="arb", waveform=q_pulse_cfg["waveform"],
                                  phase=phase_t, freq=q_pulse_cfg["ge_freq"], gain=gain_t)

            self.pulse(ch=self.cfg["gen_chs"][q_drive_ch]["ch"])
            self.sync_all(0.05)  # align channels and wait

            # add measurement
            self.measure(pulse_ch=self.cfg["gen_chs"][res_ch]["ch"],
                         adcs=ro_ch,
                         pins=[0],
                         adc_trig_offset=self.cfg["adc_trig_offset"],
                         wait=True,
                         syncdelay=self.us2cycles(syncdelay))