import numpy as np
from scipy.fft import fft, fftfreq
import pickle
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import scipy.stats as stats
################################################################
fz = 14

def plot_pearson(ax, x, y, mfreq, text):
    ax.plot(x, y, '.k', ms = 4)
    corr_coeff, p_value = stats.pearsonr(x, y)
    ax.text(0.1, 0.9, 'r = %.3f'%corr_coeff, fontsize = fz-1, transform=ax.transAxes)

    ax.set_xlabel(text + ' Noise', fontsize = fz)
    ax.set_ylabel(text + ' Sinewaves', fontsize = fz)
    return corr_coeff

rates = np.loadtxt('data/rates.txt')
pvs, pva = np.zeros(16), np.zeros(16)

VS_noise = np.loadtxt('VS_cells/VS_noise.txt')
VA_noise = np.loadtxt('VS_cells/VA_noise.txt')
VS_sine = np.loadtxt('VS_cells/VS_sine.txt')
VA_sine = np.loadtxt('VS_cells/VA_sine.txt')

data = {}
data['xrange'] = []
data['vs_noise'] = []
data['va_noise'] = []
data['vs_sine'] = []
data['va_sine'] = []

for c in range(16):
    cell = c+1
    plt .close('all')
    fig = plt.figure(figsize = [14, 9.4])
    gm = gridspec.GridSpec(175, 175)
    ax0 = plt.subplot(gm[:75, 0:75])
    ax00 = ax0.twinx()
    ax1 = plt.subplot(gm[:75, 100:175])

    ax10 = plt.subplot(gm[100:, 0:75])
    ax11 = plt.subplot(gm[100:, 100:175])

    vs_nf, va_nf = VS_noise[c], VA_noise[c]
    vs_s, va_s = VS_sine[c], VA_sine[c]
    ################# Vector angle wrap around
    va_s[np.where(va_s > 0.918)] = va_s[np.where(va_s > 0.918)] - 1
    va_nf[np.where(va_nf > 0.918)] = va_nf[np.where(va_nf > 0.918)] - 1
    #################
    rnoise, rsine = rates[c][0], rates[c][1]

    samplf = np.arange(2, 211, 1)
    mfreq = len(vs_s)
    xfreq = np.arange(1, 101)
    m_exp_freq = len(np.loadtxt('../data_experimental/data/VS_Sine_%d.txt'%(cell))[0])
    print(m_exp_freq/rsine)

    print('%.2f, %.2f, %.1f, %.1f'%(rnoise, rsine, xfreq[-1], samplf[-1]))

    # ax00.plot(samplf, vs_nf, '.-b', lw = 1.5, ms = 4)
    # ax1.plot(samplf, va_nf, '.-b', lw = 1.5, ms = 4)
    # ax0.plot(xfreq, vs_s, '.-r', lw = 1.5, ms = 4)
    # ax1.plot(xfreq, va_s, '.-r', lw = 1.5, ms = 4)

    xfreq = xfreq/rsine
    samplf = samplf/rnoise

    xlow, xhigh = np.max([xfreq[0], samplf[0]]), m_exp_freq/rsine

    nsamples = 100
    xrange = np.linspace(xlow, xhigh, nsamples)

    vs_nf = np.interp(xrange, samplf, vs_nf)
    va_nf = np.interp(xrange, samplf, va_nf)
    vs_s = np.interp(xrange, xfreq, vs_s)
    va_s = np.interp(xrange, xfreq, va_s)

    data['xrange'].append(xrange)
    data['vs_noise'].append(vs_nf)
    data['va_noise'].append(va_nf)
    data['vs_sine'].append(vs_s)
    data['va_sine'].append(va_s)

    ax00.plot(xrange, vs_nf, '.-b', lw = 1.5, ms = 4)
    ax1.plot(xrange, va_nf, '.-b', lw = 1.5, ms = 4)
    ax0.plot(xrange, vs_s, '.-r', lw = 1.5, ms = 4)
    ax1.plot(xrange, va_s, '.-r', lw = 1.5, ms = 4)

    pvs[c-1] = plot_pearson(ax10, vs_nf, vs_s, mfreq, 'VS')
    pva[c-1] = plot_pearson(ax11, va_nf, va_s, mfreq, 'VA')
#
#
    for ax in [ax0, ax1, ax10, ax11]:
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)

    for ax in [ax0, ax1]:
        ax.set_xlabel('Frequency (Hz)', fontsize = fz)
        # ax.axvline(x = rnoise, color = 'b', linestyle = '--', lw = 1.7)
        # ax.axvline(x = rsine, color = 'r', linestyle = '--', lw = 1.7)

    ax00.spines['top'].set_visible(False)

    # ax1.set_ylim(0, 0.63)

    for ax in [ax0, ax00]:
        ax.set_ylim(bottom = -0.01)

    ax0.set_ylabel('VS', fontsize = fz)

    ax1.set_ylabel('VA', fontsize = fz)

    fig.subplots_adjust(left = 0.06, bottom = 0.1, right = 0.98, top = 0.96)
    plt.savefig('figures/cell%d_norm.png'%(cell), dpi = 400)
with open('sim_data.pkl', 'wb') as f:  # Use 'wb' for binary mode
    pickle.dump(data, f)
