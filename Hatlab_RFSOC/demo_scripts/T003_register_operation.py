import time
from Hatlab_RFSOC.proxy import getSocProxy
from qick import *
from qick.qick_asm import QickRegister, QickRegisterManager
soc, soccfg = getSocProxy("myqick216-01")


class TestRegProgram(QickRegisterManager, AveragerProgram):  # inhered the QickRegisterManager mixin class
    def initialize(self):
        cfg = self.cfg
        self.declare_gen(ch=cfg["gen_ch"], nqz=1,
                         ro_ch=cfg["ro_ch"])  # this "ro_ch" will be automatically used for freq2reg operations later

        # get the frequency register associated with gen_ch
        self.gen_freq_reg = self.get_gen_reg(cfg["gen_ch"], "freq")

        # declare another frequency type register on the same generator page, initialize it to the integer that corresponds to frequency "f1" in MHz
        # the "gen_ch" and "ro_ch" of cfg["gen_ch"] will be automatically used when converting a physical frequency value to integers of this register
        self.gen_freq_reg_temp = self.new_gen_reg(cfg["gen_ch"], name="freq_temp", init_val=cfg["f1"], reg_type="freq")

        self.synci(200)  # give processor some time to configure pulses

    def body(self):
        cfg = self.cfg

        # set the register value to the integer that corresponds to frequency "f0" in MHz
        self.gen_freq_reg.set_to(cfg["f0"])
        # Write the result to address 123
        self.memwi(self.gen_freq_reg.page, self.gen_freq_reg.addr, 123)

        # assign the value of register "gen_freq_reg_temp" to register "gen_freq_reg", which should corresponds to "f1"
        self.gen_freq_reg.set_to(self.gen_freq_reg_temp)
        # Write the result to address 124
        self.memwi(self.gen_freq_reg.page, self.gen_freq_reg.addr, 124)

        # add 300 MHz to the frequency kept in "gen_freq_reg_temp" and assign it to "gen_freq_reg"
        self.gen_freq_reg.set_to(self.gen_freq_reg_temp, "+", 300)
        # Write the result to address 125
        self.memwi(self.gen_freq_reg.page, self.gen_freq_reg.addr, 125)

        # sum the frequencies kept in "gen_freq_reg_temp" and "gen_freq_reg" and assign it to "gen_freq_reg"
        self.gen_freq_reg.set_to(self.gen_freq_reg, "+", self.gen_freq_reg_temp)
        # Write the result to address 126
        self.memwi(self.gen_freq_reg.page, self.gen_freq_reg.addr, 126)


config={
        "gen_ch": 0,
        "ro_ch": 0,
        "reps": 1, # fixed
        "f0": 1000, #MHz
        "f1": 500 #MHz
       }

prog =TestRegProgram(soccfg, config)


soc.load_bin_program(prog.compile())
# Start tProc.
soc.tproc.start()

print(prog)

time.sleep(0.1)

result = soc.tproc.single_read(addr=123)
print(f"Result = ", result, prog.freq2reg(config["f0"], config["gen_ch"], config["ro_ch"]))

result = soc.tproc.single_read(addr=124)
print(f"Result = ", result, prog.freq2reg(config["f1"], config["gen_ch"], config["ro_ch"]))

result = soc.tproc.single_read(addr=125)
print(f"Result = ", result, prog.freq2reg(config["f1"]+300, config["gen_ch"], config["ro_ch"]))

result = soc.tproc.single_read(addr=126)
print(f"Result = ", result, prog.freq2reg(config["f1"]+300+config["f1"], config["gen_ch"], config["ro_ch"]))