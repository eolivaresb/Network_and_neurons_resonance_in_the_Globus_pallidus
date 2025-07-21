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
#####################################################
'''
Idea: Synaptic delay increases entreainment (VS) in the beta range

Add number of synaptic inputs, mean presynaptic IPSC rate,
PRC, phase densities, Rates and CV over frequency,
VAngle
Do figure per cell with all this info
'''
colors = ['g', '#e66101','#fdb863','#b2abd2','#5e3c99']
neuron = 245
############################################
### PRCs
def comp (x, e, pth):
    return (x**e*(pth**e - x**e))/(e*pth**(1+2*e)/(1 + 3*e+2*e**2))
def Zfunc(x, c1, c2, c3, e1, e2, e3, pth):
    z = c1*comp(x, e1, pth) + c2*comp(x, e2, pth) + c3*comp(x, e3, pth)
    z[np.where(z<0)] = 0.0
    return z

Z_params = np.loadtxt('../simulation_files/diff_prc.dat')
zbins = 500
xZ = np.arange(0.5/zbins, 1, 1./zbins)


for neuron in np.random.choice(np.arange(N), size=3, replace=False):
    Z = Zfunc(xZ, *Z_params[neuron])
    w = np.loadtxt('../simulation_files/neurons.dat')[neuron, 0]

    plt .close('all')
    fig = plt.figure(figsize = [11, 5.6])
    gm = gridspec.GridSpec(175, 275)
    axes = [plt.subplot(gm[i//3 * 100: i//3*100 + 72, i%3 * 100: i%3*100 + 72]) for i in range(6)]

    all_rates = []
    for i, delay in enumerate(delays):
        plt.figtext(i*0.19 + 0.05, 0.96, 'Syn Delay = %s ms'%delay, fontsize = 14, color = colors[i])
        VS = np.load('data_analysis/VStren_%s.npy'%(delay), allow_pickle = True)[neuron]
        VA = np.load('data_analysis/VAngle_%s.npy'%(delay), allow_pickle = True)[neuron]
        Rates = np.load('data_analysis/Rates_%s.npy'%(delay), allow_pickle = True)[neuron]
        all_rates.append(Rates)
        CVs = np.load('data_analysis/CVs_%s.npy'%(delay), allow_pickle = True)[neuron]
        pdens = np.load('data_analysis/Phase_dens_%s.npy'%(delay), allow_pickle = True)[neuron]
        PtoP = np.loadtxt('data_analysis/P_to_P_%s.txt'%(delay))[neuron]
        axes[0].plot(xZ, Z, 'k')
        axes[1].plot(freqs, Rates, color = colors[i])
        axes[1].axhline(y = w, linestyle = '--', color = 'k')
        axes[2].plot(freqs, CVs, color = colors[i])
        axes[3].plot(freqs, VS, color = colors[i])
        axes[4].plot(freqs, VA, color = colors[i])
        axes[5].plot(freqs, PtoP, color = colors[i])

    mrate = np.mean(np.concatenate(all_rates))
    for ax in axes[1:]:
        ax.axvline(x = mrate, linestyle = '--', color = 'k', lw = 1, alpha = 0.6)

    for ax in axes:
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
    for ax in axes[1:]:
        ax.set_xlim(0.01, 100)
        ax.set_xlabel('Stim freq (Hz)', fontsize = 12)

    axes[0].set_xlabel('Phase', fontsize = 12)
    axes[0].set_xlim(-0.01, 1.01)

    axes[0].set_ylim(bottom = -0.01)
    axes[2].set_ylim(0.01, 1)
    axes[3].set_ylim(0.01, 1)
    axes[4].set_ylim(0.01, 0.61)

    axes[5].axhline(y = 40, linestyle = '--', color = 'k')

    axes[0].set_ylabel('Z', fontsize = 12)
    axes[1].set_ylabel('Rate (spk/s)', fontsize = 12)
    axes[2].set_ylabel('CV ISI' , fontsize = 12)
    axes[3].set_ylabel('V Strength', fontsize = 12)
    axes[4].set_ylabel('V Angle', fontsize = 12)
    axes[5].set_ylabel('Peak to Peak I (pA)', fontsize = 12)

    axes[0].set_xlim(-0.01, 1.01)
    axes[0].set_ylim(bottom = -0.01)
    axes[0].set_xlabel('Phase', fontsize = 12)

    fig.subplots_adjust(left = 0.08, bottom = 0.08, right = 0.97, top = 0.9)
    plt.savefig('figures/neuron_%d.png'%(neuron), dpi = 400)
