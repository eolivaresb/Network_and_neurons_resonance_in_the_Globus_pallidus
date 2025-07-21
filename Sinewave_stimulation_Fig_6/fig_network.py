import matplotlib.pylab as plt
import matplotlib.gridspec as gridspec
import numpy as np
################################################################
import matplotlib
matplotlib.rcParams.update({'text.usetex': False, 'font.family': 'stixgeneral', 'mathtext.fontset': 'stix',})
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42
# Network simulation time control
################################################################
N = 1000
freqs = np.concatenate([np.arange(1, 92), np.arange(92, 201, 4)])
nfreq = len(freqs)
delays = ['hetero', '0', '4', '8', '12']
################################################################
colors = ['g', '#e66101','#fdb863','#b2abd2','#5e3c99']
############################################

plt .close('all')
fig = plt.figure(figsize = [11, 5.6])
gm = gridspec.GridSpec(175, 275)
axes = [plt.subplot(gm[i//3 * 100: i//3*100 + 72, i%3 * 100: i%3*100 + 72]) for i in range(6)]

all_rates = []
for i, delay in enumerate(delays):
    plt.figtext(0.06, 0.86 - i*0.06, 'Syn Delay = %s ms'%delay, fontsize = 14, color = colors[i])
    VS = np.load('data_analysis/VStren_%s.npy'%(delay), allow_pickle = True)
    VA = np.load('data_analysis/VAngle_%s.npy'%(delay), allow_pickle = True)
    Rates = np.load('data_analysis/Rates_%s.npy'%(delay), allow_pickle = True)
    CVs = np.load('data_analysis/CVs_%s.npy'%(delay), allow_pickle = True)
    PtoP = np.loadtxt('data_analysis/P_to_P_%s.txt'%(delay))

    axes[1].plot(freqs, np.mean(Rates[:-1], axis = 0), color = colors[i], lw = 2)
    axes[2].plot(freqs, np.mean(CVs[:-1], axis = 0), color = colors[i], lw = 2)
    axes[3].plot(freqs, np.mean(VS[:-1], axis = 0), color = colors[i], lw = 2)
    axes[4].plot(freqs, np.mean(VA[:-1], axis = 0), color = colors[i], lw = 2)
    axes[5].plot(freqs, np.mean(PtoP[:-1], axis = 0), color = colors[i], lw = 2)


for ax in axes:
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
for ax in axes[1:]:
    ax.set_xlim(0.01, 180)
    ax.set_xlabel('Stim freq (Hz)', fontsize = 12)

axes[0].axis('off')

axes[1].set_ylim(20, 30)
axes[2].set_ylim(0.01, 1)

axes[3].set_ylim(0.01, 1)
axes[4].set_ylim(0.01, 0.61)

axes[5].axhline(y = 40, linestyle = '--', color = 'k')

axes[1].set_ylabel('Rate (spk/s)', fontsize = 12)
axes[2].set_ylabel('CV ISI' , fontsize = 12)
axes[3].set_ylabel('V Strength', fontsize = 12)
axes[4].set_ylabel('V Angle', fontsize = 12)
axes[5].set_ylabel('Peak to Peak I (pA)', fontsize = 12)

fig.subplots_adjust(left = 0.08, bottom = 0.08, right = 0.97, top = 0.9)
plt.savefig('network.png', dpi = 400)
