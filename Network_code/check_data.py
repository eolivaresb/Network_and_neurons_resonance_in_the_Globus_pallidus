import numpy as np
from scipy.fft import fft, fftfreq
import struct
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import pandas as pd
################################################################
# Define your time series
N = 1000
dt = 0.00005  # Sampling time interval
ttot = 50  ##
t = np.arange(0, ttot, dt)
######
resample = 30 #### too many points on the fft to store, with this resample we get dF = 0.1 Hz
freqs = fftfreq(len(t), dt)  # Corresponding freqs
where = np.where((freqs>0)*(freqs<210))
freqs = freqs[where][::resample]



# net = np.loadtxt('simulation_files/network_hubs.dat')
# delays =  np.loadtxt('simulation_files/delays.dat')
#
# c = np.loadtxt('c16.dat')
# ################################################################
# #### Load spikes times
# f = open('./Spikes_times.dat', mode='rb').read() ## read spikes
# format = '%dd'%(len(f)//8)
# spikes = np.array(struct.unpack(format, f))
# spkn = np.array(pd.read_csv('./Spiking_neurons.dat', sep='\s+', header=None))
# spkn = spkn.reshape(len(spkn))
# ####, 'r')
#
#
#
# for pre in net.T[0][np.where(net.T[1] == 16)]:
#     d = delays[np.where((net.T[0] == pre) * (net.T[1] == 16))]
#     print(d)
#     spks = spikes[np.where(spkn == pre)]
#     for s in spks:
#         plt.axvline(x = s + d, alpha = 0.3)
#
# plt.plot(t, c)
# plt.show()
# #
################################################################
################################################################
nconv = 50
kernel = np.ones(nconv) / nconv  # Create a moving average kernel
def smooth(x):
    return np.convolve(x, kernel, mode='same')[::resample]
################################################################
def spectrum(signal):
    # Compute FFT and power spectral density
    fft_signal = np.fft.fft(signal)
    power_spectral_density = np.abs(fft_signal)**2
    power_spectral_density /= (dt * signal.sum())
    return smooth(power_spectral_density[where])
################################################################
################################################################
#### Load spikes times
f = open('./Spikes_times.dat', mode='rb').read() ## read spikes
format = '%dd'%(len(f)//8)
spk = np.array(struct.unpack(format, f))
####
signal = np.zeros(int(ttot/dt))
for s in spk:
    signal[int(s/dt)] += 1
plt.semilogy(spectrum(signal), 'r')
################################################################

################################################################
#### Load spikes times
f = open('./Spikes_times_12.dat', mode='rb').read() ## read spikes
format = '%dd'%(len(f)//8)
spk = np.array(struct.unpack(format, f))
####
signal12 = np.zeros(int(ttot/dt))
for s in spk:
    signal12[int(s/dt)] += 1
plt.semilogy(spectrum(signal12))
################################################################

plt.show()
