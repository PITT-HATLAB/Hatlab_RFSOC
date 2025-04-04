from matplotlib import pyplot as plt
import numpy as np


def plotIQTrace(iq_list, ro_chs=None):
    n_ch = len(iq_list)
    fig, axs = plt.subplots(n_ch, 1, figsize=(8, n_ch * 2.5))
    if n_ch == 1:
        axs = [axs]
    for i, iq in enumerate(iq_list):
        plot = axs[i]
        if ro_chs is not None:
            plot.set_title("ADC %d" % (ro_chs[i]))
        plot.plot(iq_list[i][0], label="I value")
        plot.plot(iq_list[i][1], label="Q value")
        plot.plot(np.abs(iq_list[i][0] + 1j * iq_list[i][1]), label="mag")
        plot.set_ylabel("a.u.")
        # plot.set_ylim([-5000,5000])
        plot.set_xlabel("Clock ticks")
        # plot.set_title("Averages = " + str(config["soft_avgs"]))
        plot.legend()

    plt.tight_layout()


def plotAvgIQresults(xdata, avgi, avgq, windowName=None, title=None, xlabel=None, ylabel=None, sub_titles:list=None):
    n_ch = len(avgi)
    fig, axs = plt.subplots(1, n_ch, figsize=(n_ch * 6, 5), num=windowName)
    if n_ch == 1:
        axs = [axs]
    for i, iq in enumerate(zip(avgi, avgq)):
        plot = axs[i]
        if (sub_titles is None) or (len(sub_titles) == 1):
            plot.set_title(title)
        else:
            plot.set_title(sub_titles[i])
        plot.plot(xdata, avgi[i][0], label="I value")
        plot.plot(xdata, avgq[i][0], label="Q value")
        plot.set_ylabel(ylabel)
        plot.set_xlabel(xlabel)
        plot.legend()
    if n_ch > 1:
        fig.suptitle(title)
        plt.tight_layout()

def plotIQHist2dLog(di_buf, dq_buf, ro_chs=None, bins=101):
    n_ch = len(di_buf)
    fig, axs = plt.subplots(1, n_ch, figsize=(n_ch*5, 4.5))
    if n_ch == 1:
        axs = [axs]
    for i in range(n_ch):
        ax = axs[i]
        range_ = np.max(np.abs([di_buf[i], dq_buf[i]]))
        if ro_chs is not None:
            ax.set_title("ADC %d"%(ro_chs[i]))
        hist, binx, biny = np.histogram2d(di_buf[i], dq_buf[i], bins=bins, range=[[-range_, range_],[-range_, range_]])
        ax.pcolormesh(binx, biny, 10*np.log10(hist.T))
        ax.set_aspect(1)
        ax.set_ylabel("Q")
        ax.set_xlabel("I")
    
def plotIQHist2d(di_buf, dq_buf, ro_chs=None, bins=101, logPlot=False):
    n_ch = len(di_buf)
    fig, axs = plt.subplots(1, n_ch, figsize=(n_ch*5, 4.5))
    if n_ch == 1:
        axs = [axs]
    for i in range(n_ch):
        ax = axs[i]
        range_ = np.max(np.abs([di_buf[i], dq_buf[i]]))
        if ro_chs is not None:
            ax.set_title("ADC %d"%(ro_chs[i]))
        if logPlot:
            hist, x, y = np.histogram2d(di_buf[i], dq_buf[i], bins=bins, range=[[-range_, range_],[-range_, range_]])
            ax.pcolor(x,y,np.log(hist))
        else:
            ax.hist2d(di_buf[i], dq_buf[i], bins=bins, range=[[-range_, range_],[-range_, range_]])
        ax.set_aspect(1)
        ax.set_ylabel("Q")
        ax.set_xlabel("I")

    plt.tight_layout()
    
def plotIQpcolormesh(xdata, ydata, idata, qdata, title=None):
    fig, axs = plt.subplots(1,2,figsize=(8,5))
    fig.suptitle(title)
    im = axs[0].pcolormesh(xdata, ydata, idata.T, shading="auto")
    plt.colorbar(im, ax=axs[0])
    im = axs[1].pcolormesh(xdata, ydata, qdata.T, shading="auto")
    plt.colorbar(im, ax=axs[1])

def plotWaveform(prog, ch: int, waveform: str, phy_unit=True, **kwargs):
    pulse_data = prog.pulses[ch][waveform]['data']
    f_dds = prog.soccfg['gens'][ch]['fs']
    xdata = np.arange(len(pulse_data)) / f_dds if phy_unit else np.arange(len(pulse_data))
    
    plt.figure(**kwargs)
    plt.plot(xdata, pulse_data[:, 0], label="I")
    plt.plot(xdata, pulse_data[:, 1], label="Q")
    plt.plot(xdata, np.abs(pulse_data[:, 0] + 1j * pulse_data[:, 1]), label="mag")
    plt.legend()
    plt.xlabel(f"time {'(us)' if phy_unit else '(clock cycle)'}")
    plt.ylabel("DAC")

