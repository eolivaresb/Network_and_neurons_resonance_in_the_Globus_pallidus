import matplotlib.pylab as plt
import matplotlib.gridspec as gridspec
import numpy as np
# Network simulation time control
N = 1000
ttot = 300  ##
################################################################
frequencies = np.arange(101)
nfreq = len(frequencies)
################################################################
#####################################################
'''
Idea: Synaptoic delay increases entreainment (VS) in the beta range

Add number of synaptic inputs, mean presynaptic IPSC rate,
PRC, phase densities, Rates and CV over frequency,
VAngle
Do figure per cell with all this info

'''

############################################
plt .close('all')
fig = plt.figure(figsize = [10, 5.2])
gm = gridspec.GridSpec(380, 280)
axes = [plt.subplot(gm[i//3 * 100: i//3*100 + 80, i%3 * 100: i%3*100 + 80]) for i in range(12)]
neurons = np.random.randint(0, N, 12)

for i, delay in enumerate(delays):
    VS = np.load('data3/VStren_%d.npy'%(delay), allow_pickle = True)
    VA = np.load('data3/VAngle_%d.npy'%(delay), allow_pickle = True)
    Rates = np.load('data3/Rates_%d.npy'%(delay), allow_pickle = True)
    for k, n in enumerate(neurons):
        axes[k].plot(frequencies[1:], VS[n], color = colors[i])
        axes[k].axvline(x = np.mean(Rates[n]), color = 'r')
for ax in axes:
    ax.set_xlim(0.01, 100)
    ax.set_ylim(0, 1)
    ax.set_xlabel('Stim freq (Hz)')
    ax.set_ylabel('VS')

plt.show()
