
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from scipy.integrate import ode
from scipy.misc import derivative

def boxDrive(t, rampSlope, cutFactor, width):
    return 0.5 * (np.tanh((rampSlope * (t)) - cutFactor) -
            np.tanh(rampSlope * (t - width) + cutFactor))

def boxDrive1(t, rampWidth, cutOffset, width):
    """

    :param t: time list
    :param rampWidth: number of points from cutOffset to 0.95 amplitude
    :param cutOffset: the initial offset to cut on the tanh Function
    :param width:
    :return:
    """
    c0_ = np.arctanh(2*cutOffset-1)
    c1_ = np.arctanh(2*0.95-1)
    k_ = (c1_-c0_)/rampWidth
    return (0.5 * (np.tanh(k_ * t + c0_) -np.tanh(k_ * (t - width) -c0_)) - cutOffset )/(1-cutOffset)


slope = 0.5
cf = 3
wd = 300
t_list = np.linspace(0,wd,wd+1)


plt.figure()
plt.plot(t_list, boxDrive(t_list, 0.5, cf, wd))
plt.plot(t_list, boxDrive1(t_list,10, 0.01, 80))

