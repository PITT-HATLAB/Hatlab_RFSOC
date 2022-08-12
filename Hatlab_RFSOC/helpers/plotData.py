from matplotlib import pyplot as plt
import numpy as np

def plotIQTrace(iq_list, ro_chs=None):
    n_ch = len(iq_list)
    fig, axs = plt.subplots(n_ch,1,figsize=(8,n_ch*2.5))

    for i, iq in enumerate(iq_list):
        plot = axs[i]
        if ro_chs is not None:
            plot.set_title("ADC %d"%(ro_chs[i]))
        plot.plot(iq_list[i][0], label="I value" )
        plot.plot(iq_list[i][1], label="Q value" )
        plot.plot(np.abs(iq_list[i][0]+1j*iq_list[i][1]), label="mag")
        plot.set_ylabel("a.u.")
        # plot.set_ylim([-5000,5000])
        plot.set_xlabel("Clock ticks")
        #plot.set_title("Averages = " + str(config["soft_avgs"]))
        plot.legend()

    plt.tight_layout()
    
    
def plotIQHist2d(di_buf, dq_buf, ro_chs=None, bins=101):
    n_ch = len(di_buf)
    fig, axs = plt.subplots(1, n_ch, figsize=(n_ch*5, 4.5))

    for i in range(n_ch):
        ax = axs[i]
        range_ = np.max(np.abs([di_buf[i], dq_buf[i]]))
        if ro_chs is not None:
            ax.set_title("ADC %d"%(ro_chs[i]))
        ax.hist2d(di_buf[i], dq_buf[i], bins=bins, range=[[-range_, range_],[-range_, range_]])
        ax.set_aspect(1)
        ax.set_ylabel("Q")
        ax.set_xlabel("I")

    plt.tight_layout()
    
def plotIQpcolormesh(xdata, ydata, idata, qdata):
    fig, axs = plt.subplots(1,2,figsize=(8,5))
    im = axs[0].pcolormesh(xdata, ydata, idata.T, shading="auto")
    plt.colorbar(im, ax=axs[0])
    im = axs[1].pcolormesh(xdata, ydata, qdata.T, shading="auto")
    plt.colorbar(im, ax=axs[1])
