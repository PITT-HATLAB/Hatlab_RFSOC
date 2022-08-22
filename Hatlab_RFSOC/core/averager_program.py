from typing import Dict, List, Union, Callable, Literal, Tuple

from tqdm import tqdm
import numpy as np
from qick.qick_asm import QickProgram

from Hatlab_RFSOC.core.pulses import add_gaussian, add_tanh

RegisterTypes= Literal["freq", "time", "phase", "adc_freq"]

class QickRegister():
    def __init__(self, prog, page:int, addr:int, reg_type:RegisterTypes=None, gen_ch=None, ro_ch=None, init_val=None):
        self.prog = prog
        self.page = page
        self.addr = addr
        self.type = reg_type
        self.gen_ch = gen_ch
        self.ro_ch = ro_ch
        self.init_val = init_val
        if init_val is not None:
            self.reset()

    def val2reg(self, val):
        if self.type == "freq":
            return self.prog.freq2reg(val, self.gen_ch, self.ro_ch)
        elif self.type == "time":
            return self.prog.us2cycles(val, self.gen_ch, self.ro_ch)
        elif self.type == "phase":
            return self.prog.deg2reg(val, self.gen_ch)
        elif self.type == "adc_freq":
            return self.prog.freq2reg_adc(val, self.ro_ch, self.gen_ch)
        else:
            return val

    def reg2val(self, reg):
        if self.type == "freq":
            return self.prog.reg2freq(reg, self.gen_ch)
        elif self.type == "time":
            return self.prog.cycles2us(reg, self.gen_ch, self.ro_ch)
        elif self.type == "phase":
            return self.prog.reg2deg(reg, self.gen_ch)
        elif self.type == "adc_freq":
            return self.prog.reg2freq_adc(reg, self.ro_ch)
        else:
            return reg

    def reset(self):
        self.prog.safe_regwi(self.page, self.addr, self.val2reg(self.init_val))



class PAveragerProgram(QickProgram):
    """
    NDAveragerProgram class, for qubit experiments that sweep over multiple variables.

    :param cfg: Configuration dictionary
    :type cfg: dict
    """

    def __init__(self, soccfg, cfg):
        """
        Constructor for the RAveragerProgram, calls make program at the end so for classes that inherit from this if you want it to do something before the program is made and compiled either do it before calling this __init__ or put it in the initialize method.
        """
        super().__init__(soccfg)
        self.cfg = cfg
        self.user_reg_dict={} # look up dict for registers defined for each channel
        self._user_regs = [] # all user defined registers
        self.declare_all_gens()
        self.declare_all_readouts()

        self.make_program()


    def declare_all_gens(self):
        """
        initialize all generators in the config dict
        :return:
        """
        for gen_ch, kws in self.cfg["gen_chs"].items():
            if ("ch_I" in kws) and ("ch_Q" in kws):
                ch_i = kws.pop("ch_I")
                ch_q = kws.pop("ch_Q")
                for arg in ["skew_phase","IQ_scale"]:
                    try:
                        kws.pop(arg)
                    except AttributeError:
                        pass
                self.declare_gen(ch_i, **kws)
                self.declare_gen(ch_q, **kws) # todo: all the other functions doesn't support IQ channel gen yet...
            else:
                self.declare_gen(**kws)
            self.user_reg_dict[gen_ch] = {}


    def declare_all_readouts(self):
        for ro_ch, kws in self.cfg["ro_chs"].items():
            self.declare_readout(**kws)



    # def initialize_gen(self, gen_ch:str): #todo: what am I doing here???
    #     """
    #     initialize a generator using the parameters in the config dict
    #     :param gen_ch: name of generator channel
    #     :return:
    #     """
    #     gen_cgf = self.cfg["gen_chs"][gen_ch]
    #     mixer_freq = gen_cgf.get("mixer_freq", 0)
    #     mux_freqs = gen_cgf.get("mux_freqs")
    #     mux_gains = gen_cgf.get("mux_gains")
    #     ro_ch = gen_cgf.get("ro_ch")
    #     self.declare_gen(gen_cgf["ch"], gen_cgf["nqz"], mixer_freq, mux_freqs, mux_gains, ro_ch)


    def get_reg(self, gen_ch:str, name:str) -> QickRegister:
        """
        get a qick generator register page and its address
        :param gen_ch:
        :param name:  name of the qick register. as in QickProgram.pulse_registers
        :return: QickRegister
        """
        gen_cgf = self.cfg["gen_chs"][gen_ch]
        page = self.ch_page(gen_cgf["ch"])
        addr = self.sreg(gen_cgf["ch"], name)
        reg_type = name if name in RegisterTypes.__args__ else None
        reg = QickRegister(self, page, addr, reg_type, gen_cgf["ch"], gen_cgf.get("ro_ch"))
        return reg

    def new_reg(self, gen_ch:str, name:str=None, init_val=None, reg_type:RegisterTypes=None) -> QickRegister:
        """
        declare a new register in the generator register page. address automatically adds 1 one when each time a new
        register in the same page is declared.
        :param gen_ch: generator channel
        :param name: name of the new register, only for keeping records in self.user_reg_dict
        :param init_val: initial value for the register, when reg_type is provided, the reg_val should be in the unit
            of the corresponding type.
        "param reg_type: type of the register, e.g. freq, time, phase
        :return: QickRegister
        """
        gen_cgf = self.cfg["gen_chs"][gen_ch]
        page = self.ch_page(gen_cgf["ch"])
        addr = 1
        while (page, addr) in self._user_regs:
            addr += 1
        if addr > 12:
            raise ValueError(f"registers in page {page} ({gen_ch}) is full.")
        self._user_regs.append((page, addr))

        if name is None:
            name = f"reg_{addr}"
        if name in self.user_reg_dict[gen_ch].keys():
            raise KeyError(f"register name '{name}' already exists for channel {gen_ch}")

        reg = QickRegister(self, page, addr, reg_type, gen_cgf["ch"], gen_cgf.get("ro_ch"), init_val)
        self.user_reg_dict[gen_ch][name] = reg

        return reg

    def set_pulse_params(self, gen_ch:str, **kwargs):
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
        kw_reg = kwargs.copy()
        gen_cgf = self.cfg["gen_chs"][gen_ch]
        if "freq" in kwargs:
            kw_reg["freq"] = self.soccfg.freq2reg(kwargs["freq"], gen_cgf["ch"], gen_cgf.get("ro_ch"))
        if "phase" in kwargs:
            kw_reg["phase"] = self.soccfg.deg2reg(kwargs["phase"], gen_cgf["ch"])
        if "length" in kwargs:
            kw_reg["length"] = self.soccfg.us2cycles(kwargs["length"], gen_cgf["ch"])
        self.set_pulse_registers(gen_cgf["ch"], **kw_reg)

    def add_waveform(self, gen_ch, name, shape, **kwargs):
        # todo: the parser can be better... instead of using if commands to select from only two possible waveforms here
        #   We should have a abstract waveform class. each new waveform should be written as a waveform class instance,
        #   and the parser should search for waveform in pulses.py

        if shape == "gaussian":
            add_gaussian(self, gen_ch, name, **kwargs)
        elif shape == "tanh_box":
            add_tanh(self, gen_ch, name, **kwargs)
        else:
            raise NameError(f"unsupported pulse shape {shape}")

    def add_waveform_from_cfg(self, gen_ch, name):
        pulse_params = self.cfg["waveforms"][name]
        self.add_waveform(gen_ch, name, **pulse_params)


    def initialize(self):
        """
        Abstract method for initializing the program and can include any instructions that are executed once at the beginning of the program.
        """
        pass

    def body(self):
        """
        Abstract method for the body of the program
        """
        pass

    def update(self):
        """
        Abstract method for updating the program
        """
        pass

    def make_program(self):
        """
        A template program which repeats the instructions defined in the body() method the number of times specified in self.cfg["reps"].
        """
        p = self

        rcount = 13
        rii = 14
        rjj = 15


        p.regwi(0, rcount, 0)

        p.regwi(0, rjj, self.cfg["reps"]-1)
        p.label("LOOP_J")

        p.initialize()
        p.regwi(0, rii, self.cfg["expts"]-1)
        p.label("LOOP_I")

        p.body()

        p.mathi(0, rcount, rcount, "+", 1)
        p.update()

        p.memwi(0, rcount, 1)

        p.loopnz(0, rii, "LOOP_I")

        p.loopnz(0, rjj, 'LOOP_J')


        p.end()

    def get_expt_pts(self):
        """
        Method for calculating experiment points (for x-axis of plots) based on the config.

        :return: Numpy array of experiment points
        :rtype: np.array
        """
        return self.cfg["start"]+np.arange(self.cfg['expts'])*self.cfg["step"]

    def acquire_round(self, soc, threshold=None, angle=None,  readouts_per_experiment=1, save_experiments=None, load_pulses=True, start_src="internal", progress=False, debug=False):
        """
        This method optionally loads pulses on to the SoC, configures the ADC readouts, loads the machine code representation of the AveragerProgram onto the SoC, starts the program and streams the data into the Python, returning it as a set of numpy arrays.

        config requirements:
        "reps" = number of repetitions;

        :param soc: Qick object
        :type soc: Qick object
        :param threshold: threshold
        :type threshold: int
        :param angle: rotation angle
        :type angle: list
        :param readouts_per_experiment: readouts per experiment
        :type readouts_per_experiment: int
        :param save_experiments: saved experiments
        :type save_experiments: list
        :param load_pulses: If true, loads pulses into the tProc
        :type load_pulses: bool
        :param start_src: "internal" (tProc starts immediately) or "external" (waits for an external trigger)
        :type start_src: string
        :param progress: If true, displays progress bar
        :type progress: bool
        :param debug: If true, displays assembly code for tProc program
        :type debug: bool
        :returns:
            - avg_di (:py:class:`list`) - list of lists of averaged accumulated I data for ADCs 0 and 1
            - avg_dq (:py:class:`list`) - list of lists of averaged accumulated Q data for ADCs 0 and 1
        """

        if angle is None:
            angle = [0, 0]
        if save_experiments is None:
            save_experiments = [0]
        if load_pulses:
            self.load_pulses(soc)

        # Configure signal generators
        self.config_gens(soc)

        # Configure the readout down converters
        self.config_readouts(soc)
        self.config_bufs(soc, enable_avg=True, enable_buf=True)

        # load this program into the soc's tproc
        self.load_program(soc, debug=debug)

        # configure tproc for internal/external start
        soc.start_src(start_src)

        reps, expts = self.cfg['reps'], self.cfg['expts']

        count = 0
        total_count = reps*expts*readouts_per_experiment
        n_ro = len(self.ro_chs)

        d_buf = np.zeros((n_ro, 2, total_count))
        self.stats = []

        with tqdm(total=total_count, disable=not progress) as pbar:
            soc.start_readout(total_count, counter_addr=1, ch_list=list(
                self.ro_chs), reads_per_count=readouts_per_experiment)
            while count<total_count:
                new_data = soc.poll_data()
                for d, s in new_data:
                    new_points = d.shape[2]
                    d_buf[:, :, count:count+new_points] = d
                    count += new_points
                    self.stats.append(s)
                    pbar.update(new_points)

        # reformat the data into separate I and Q arrays
        di_buf = d_buf[:,0,:]
        dq_buf = d_buf[:,1,:]

        # save results to class in case you want to look at it later or for analysis
        self.di_buf = di_buf
        self.dq_buf = dq_buf

        if threshold is not None:
            self.shots = self.get_single_shots(
                di_buf, dq_buf, threshold, angle)

        expt_pts = self.get_expt_pts()

        avg_di = np.zeros((n_ro, len(save_experiments), expts))
        avg_dq = np.zeros((n_ro, len(save_experiments), expts))

        for nn, ii in enumerate(save_experiments):
            for i_ch, (ch, ro) in enumerate(self.ro_chs.items()):
                if threshold is None:
                    avg_di[i_ch][nn] = np.sum(di_buf[i_ch][ii::readouts_per_experiment].reshape(
                        (reps, expts)), 0)/(reps)/ro.length
                    avg_dq[i_ch][nn] = np.sum(dq_buf[i_ch][ii::readouts_per_experiment].reshape(
                        (reps, expts)), 0)/(reps)/ro.length
                else:
                    avg_di[i_ch][nn] = np.sum(
                        self.shots[i_ch][ii::readouts_per_experiment].reshape((reps, expts)), 0)/(reps)
                    avg_dq = np.zeros(avg_di.shape)

        return expt_pts, avg_di, avg_dq

    def get_single_shots(self, di, dq, threshold, angle=None):
        """
        This method converts the raw I/Q data to single shots according to the threshold and rotation angle

        :param di: Raw I data
        :type di: list
        :param dq: Raw Q data
        :type dq: list
        :param threshold: threshold
        :type threshold: int
        :param angle: rotation angle
        :type angle: list

        :returns:
            - single_shot_array (:py:class:`array`) - Numpy array of single shot data

        """

        if angle is None:
            angle = [0, 0]
        if type(threshold) is int:
            threshold = [threshold, threshold]
        return np.array([np.heaviside((di[i]*np.cos(angle[i]) - dq[i]*np.sin(angle[i]))/self.ro_chs[ch].length-threshold[i], 0) for i, ch in enumerate(self.ro_chs)])

    def acquire(self, soc, threshold=None, angle=None, load_pulses=True, readouts_per_experiment=1, save_experiments=None, start_src="internal", progress=False, debug=False):
        """
        This method optionally loads pulses on to the SoC, configures the ADC readouts, loads the machine code representation of the AveragerProgram onto the SoC, starts the program and streams the data into the Python, returning it as a set of numpy arrays.
        config requirements:
        "reps" = number of repetitions;

        :param soc: Qick object
        :type soc: Qick object
        :param threshold: threshold
        :type threshold: int
        :param angle: rotation angle
        :type angle: list
        :param readouts_per_experiment: readouts per experiment
        :type readouts_per_experiment: int
        :param save_experiments: saved experiments
        :type save_experiments: list
        :param load_pulses: If true, loads pulses into the tProc
        :type load_pulses: bool
        :param start_src: "internal" (tProc starts immediately) or "external" (each round waits for an external trigger)
        :type start_src: string
        :param progress: If true, displays progress bar
        :type progress: bool
        :param debug: If true, displays assembly code for tProc program
        :type debug: bool
        :returns:
            - expt_pts (:py:class:`list`) - list of experiment points
            - avg_di (:py:class:`list`) - list of lists of averaged accumulated I data for ADCs 0 and 1
            - avg_dq (:py:class:`list`) - list of lists of averaged accumulated Q data for ADCs 0 and 1
        """
        reps, expts, rounds = self.cfg['reps'], self.cfg['expts'], self.cfg.get("rounds", 1)
        msmt_per_rep = expts * readouts_per_experiment
        tot_reps = reps * rounds
        total_msmt = msmt_per_rep * tot_reps

        n_ro = len(self.ro_chs)

        self.di_buf_p = np.zeros((n_ro, tot_reps, msmt_per_rep))
        self.dq_buf_p = np.zeros((n_ro, tot_reps, msmt_per_rep))

        if angle is None:
            angle = [0, 0]
        if save_experiments is None:
            save_experiments = [0]
        if "rounds" not in self.cfg or self.cfg["rounds"] == 1:
            expt_pts, avg_di, avg_dq = self.acquire_round(soc, threshold=threshold, angle=angle, readouts_per_experiment=readouts_per_experiment, save_experiments=save_experiments, load_pulses=load_pulses, start_src=start_src, progress=progress, debug=debug)
            self.di_buf_p = self.di_buf.reshape(n_ro, reps, -1)
            self.dq_buf_p = self.dq_buf.reshape(n_ro, reps, -1)
            return expt_pts, avg_di, avg_dq

        avg_di = None
        for ii in tqdm(range(self.cfg["rounds"]), disable=not progress):
            expt_pts, avg_di0, avg_dq0 = self.acquire_round(soc, threshold=threshold, angle=angle, readouts_per_experiment=readouts_per_experiment,
                                                            save_experiments=save_experiments, load_pulses=load_pulses, start_src=start_src, progress=progress, debug=debug)

            if avg_di is None:
                avg_di, avg_dq = avg_di0, avg_dq0
            else:
                avg_di += avg_di0
                avg_dq += avg_dq0

            self.di_buf_p[:, reps * ii: reps * (ii + 1), :] = self.di_buf.reshape(n_ro, reps, -1)
            self.dq_buf_p[:, reps * ii: reps * (ii + 1), :] = self.dq_buf.reshape(n_ro, reps, -1)

        return expt_pts, avg_di/self.cfg["rounds"], avg_dq/self.cfg["rounds"]



    def declareMuxedGenAndReadout(self, res_ch: int, res_nqz: Literal[1, 2], res_mixer_freq: float,
                                  res_freqs: List[float], res_gains: List[float], ro_chs: List[int],
                                  readout_length: int):
        """ declare muxed generator and readout channels
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
        self.declare_gen(ch=res_ch, nqz=res_nqz, mixer_freq=res_mixer_freq,
                         mux_freqs=res_freqs,
                         ro_ch=ro_chs[0], mux_gains=res_gains)

        # configure the readout lengths and downconversion frequencies
        for iCh, ch in enumerate(ro_chs):
            self.declare_readout(ch=ch, freq=res_freqs[iCh], length=readout_length,
                                 gen_ch=res_ch)
