import numpy as np
from scipy import interpolate
from scipy import signal
import pickle
pi = np.pi
N = 16

#################################################################
'''
Sinewave VS
'''
freqs = np.arange(1, 101)
nfreq = len(freqs)

VS = np.zeros((N, nfreq))
VA = np.zeros((N, nfreq))

for k, freq in enumerate(freqs):
    with open('simulation/data_sim/spikes_freq_%d.pkl'%(freq), 'rb') as f:  # Save
        spikes = pickle.load(f)
    ### Get VS magnitude and angle for every cell
    for c, s in enumerate(spikes):
        nspk = len(s)
        Sph = (s*freq)%(1.) # spikes phase aligned to stim freq [0, 1)
        xv, yv = np.sum(np.cos(2*pi*Sph)), np.sum(np.sin(2*pi*Sph))
        VS[c, k] = np.sqrt(xv**2 + yv**2)/nspk
        VA[c, k] = (np.arctan2(yv, xv)/(2*pi))%1
np.savetxt('VS_cells/VS_sine.txt', VS)
np.savetxt('VS_cells/VA_sine.txt', VA)
###############################################################

#################################################################
'''
Noise VS
'''
epsilon = 0.0000001
######################################
dt = 0.0005 ##2 kHz
ttot = 400
time = np.arange(0, ttot, dt)
freq = np.fft.fftfreq(len(time), dt)
######################################
samplf = np.arange(2, 211, 1)
window_width = 1.5
window = signal.windows.tukey(int(ttot*(2*window_width)+1), alpha = 0.25)
################################################################
def interp_functions(noise):
    a = np.fft.fft(noise)
    int_func = []
    for f in samplf:
        ###
        ### calculate noise component
        w = np.zeros(len(a))
        w[np.where((freq>=f-window_width)*(freq<=f+window_width))] = window
        w[np.where((freq>=-f-window_width)*(freq<=-f+window_width))] = window
        af = a*w
        ncomp = np.real(np.fft.ifft(af)) ### noise component on this window
        ###
        ### calculate zero crossings
        zcindx = np.where(np.diff(np.sign(ncomp))==2) ## zeros crossing indexes
        lcomp, rcomp = ncomp[zcindx], ncomp[1:][zcindx] ## component values around zcross
        interp_time = dt*(lcomp/(lcomp-rcomp))   ## time interpolation
        tz = time[zcindx]+interp_time              ## time on the left plus interpolation
        ###
        ### Add interpolation function
        int_vector = [] ## vector to interpolate spike phase
        int_vector.append([0, 0])
        for t in tz:
            int_vector.append([t-epsilon, 1])
            int_vector.append([t+epsilon, 0])
        int_vector.append([ttot,1])
        int_vector = np.array(int_vector).T
        int_func.append(interpolate.interp1d(int_vector[0],int_vector[1]))
    return int_func
################################################################
################################################################
def get_vs_f(Sph):
    nspk = len(Sph)
    xv, yv = np.sum(np.cos(2*pi*Sph)), np.sum(np.sin(2*pi*Sph))
    return np.sqrt(xv**2 + yv**2)/nspk, (np.arctan2(yv, xv)/(2*pi))%1
################################################################
def get_VS(int_func, spikes):
    vs, va = np.zeros(len(samplf)), np.zeros(len(samplf))
    for k, func in enumerate(int_func):
        Sph = func(spikes[1:-2]) ## eliminate first and last 2 spikes just to avoid edge problems
        vs[k], va[k] = get_vs_f(Sph)
    return vs, va
################################################################
################################################################
VS = np.zeros((N, len(samplf)))
VA = np.zeros((N, len(samplf)))
####
noise = np.loadtxt('simulation/data_sim/noise.dat')
with open('simulation/data_sim/spikes_noise.pkl', 'rb') as f:  # Save
    spk = pickle.load(f)
####
int_func = interp_functions(noise)
####
for n in range(N):
    VS[n], VA[n] = get_VS(int_func, spk[n])
np.savetxt('VS_cells/VS_noise.txt', VS)
np.savetxt('VS_cells/VA_noise.txt', VA)
