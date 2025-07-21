import numpy as np
pi = np.pi
################################################################
################################################################
N = 1000
################################################################
frequencies = np.concatenate([np.arange(1, 92), np.arange(92, 201, 4)])
nfreq = len(frequencies)
################################################################
################################################################
bins_data = 1000
time_data = np.arange(0.5/bins_data, 1, 1./bins_data)

################################################################
### All cycles will be reduced to a 40 bins cycle
bins_interp = 40
x_bins = np.linspace(0, 1, 1 + bins_interp)  ## edges of bins in [0, 1]
### Stimuli over cycles is always the same
Iamp = 20 ##pA
I_stim = Iamp * np.sin(2 * pi * np.arange(0.5/bins_interp, 1, 1./bins_interp))

################################################################

def average_cycle(data, freq):
    ''' Synaptic current was saved on a 1 second repeated window,
    this function average the current on cycles that depends on the stimulation frequency
    first, it calculate the xvalues (phase positions on the 1 sec recording data)
    then calculate the average current on each bin of x_bins'''
    xvalues = freq*(time_data%(1./freq))
    I_syn = np.zeros(bins_interp)
    for k, lm in enumerate(x_bins[:-1]):
        rm = x_bins[k+1]
        I_syn[k] = np.mean(data[np.where((xvalues>=lm)*(xvalues<rm))])
    return I_syn + I_stim

################################################################
################################################################
for delay in ['hetero', '0', '4', '8', '12']:
    P_to_P = np.zeros((N, nfreq))
    for j, freq in enumerate(frequencies):
        #### Load 1 second total current per neuron
        It = np.loadtxt('data/MeanI_delay_%s_freq_%d.dat'%(delay, freq))
        #######################
        #### Calculate mean current per stim cycle and neuron
        for n in range(N):
            meanI_cycle = average_cycle(It[n], freq)
            maximum = np.nanmax(meanI_cycle)
            minimum = np.nanmin(meanI_cycle)
            P_to_P[n, j] = maximum - minimum
    np.savetxt('data_analysis/P_to_P_%s.txt'%(delay), P_to_P, fmt = '%.5f')
