import numpy as np
from scipy import signal
import matplotlib.pyplot as plt
import pickle
import matplotlib.gridspec as gridspec
################################################################
import matplotlib
# matplotlib.rcParams.update({'text.usetex': False, 'font.family': 'stixgeneral', 'mathtext.fontset': 'stix',})
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42
matplotlib.rcParams["pdf.use14corefonts"] = True
matplotlib.rcParams["ps.useafm"] = True
matplotlib.rcParams['text.usetex'] = False
################################################################
################################################################
delays = np.arange(0, 32, 0.5)
ndel = len(delays)
fs, nper = 1000, 10000
################################################################
S = np.random.normal(0,1 , 20000)
freqs, x = signal.coherence(S, S, fs=fs, nperseg=nper)
################################################################
data_final = {}
################################################################
colors = ['k', 'g', 'r']
################################################################
def find_peak(f, y):
    ''' find the f value at the peak of x, not considering the freq<2 hz point in x, y
    makes sure to return the first peak'''
    pindx = np.argmax(y[2:])
    mean_second_valley = np.mean(y[int(2+0.66*pindx):int(2+int(0.75*pindx))])
    pindx2 = np.argmax(y[2:int(0.66*pindx)])  ## peak in the left part of the maximum peak
    if y[2:][pindx2] > 1.5*mean_second_valley:  ## clear peak if peak value greater than in between valley mean value
        return f[2:][pindx2]
    else:
        return f[2:][pindx]
################################################################
plt .close('all')
fig = plt.figure(figsize = [9, 6.4])
gm = gridspec.GridSpec(165, 170)
ax1 = [plt.subplot(gm[i*100:i*100 + 65, :70]) for i in range(2)]
ax2 = [plt.subplot(gm[i*100:i*100 + 65, 100:170]) for i in range(2)]
################################################################
################################################################
nindx = 0
labels = ['Small-world', 'Random', 'Hubs']
for nindx, net in enumerate(['small', 'random', 'hub']):
    for c, cond in enumerate(['', '_shared']):
        data_final[net+cond] = {}
        with open('data_analysis/PSD_%s%s.pkl'%(net, cond), 'rb') as f:  # Use 'wb' for binary mode
            data = pickle.load(f)
        ################################################################
        peaks_psd = np.zeros(ndel)
        mean_rate, std_rate = np.zeros(ndel), np.zeros(ndel)
        min_rate, max_rate = np.zeros(ndel), np.zeros(ndel)
        min_autoc, max_autoc = np.zeros(ndel), np.zeros(ndel)
        ################################################################
        for d, delay in enumerate(delays):
            peaks_psd[d] = find_peak(freqs, data['psd'][d])
            min_autoc[d], max_autoc[d] = np.min(data['autoc'][d][1:]), np.max(data['autoc'][d][1:])
            mean_rate[d], std_rate[d], min_rate[d], max_rate[d] = data['minmax'][d]
        ################################################################
        ax1[c].plot(delays, 1000./peaks_psd, '.-', color = colors[nindx], label = labels[nindx])
        ax2[c].plot(delays, (max_autoc - min_autoc)/2, '.-', color = colors[nindx], label = labels[nindx])
        ax2[c].plot(delays, (min_autoc - max_autoc)/2, '.-', color = colors[nindx])
        data_final[net+cond]['peaks_psd'] = peaks_psd
        data_final[net+cond]['min_autoc'] = min_autoc
        data_final[net+cond]['max_autoc'] = max_autoc

with open('./final_data_network.pkl', 'wb') as f:  # Use 'wb' for binary mode
    pickle.dump(data_final, f)
##### Titles
y1 = 1.18
y2 = 0.0
ax1[0].text(1.2, y1-0.03, 'Intrinsic behavior', fontsize = 14, ha = 'center', transform = ax1[0].transAxes)
ax1[1].text(1.2, y1+y2-0.03, 'Shared input', fontsize = 14, ha = 'center', transform = ax1[1].transAxes)

arrowprops=dict(arrowstyle='|-|, widthA=0.2,widthB=0.2', linewidth = 1.25, edgecolor = 'k')
ax1[0].annotate("",  xy= [0.0, y1], xycoords='axes fraction', xytext = [0.83,y1], textcoords = 'axes fraction', arrowprops=arrowprops)
ax1[0].annotate("",  xy= [1.58, y1], xycoords='axes fraction', xytext = [2.42,y1], textcoords = 'axes fraction', arrowprops=arrowprops)

ax1[1].annotate("",  xy= [0.0, y2+y1], xycoords='axes fraction', xytext = [0.85,y2+ y1], textcoords = 'axes fraction', arrowprops=arrowprops)
ax1[1].annotate("",  xy= [1.56,y2+ y1], xycoords='axes fraction', xytext = [2.42,y2+ y1], textcoords = 'axes fraction', arrowprops=arrowprops)

for ax in ax1:
    ax.set_ylabel('Period PSD peak (ms)', fontsize = 13)


for ax in ax2:
    ax.set_ylabel('Autocorrelation Amplitude\n(Spk/s)', fontsize = 13)
ax2[0].set_yticks([-1, 0, 1])
ax2[0].set_yticklabels(['-1', '0', '1'])

ax2[1].set_yticks([-6, -3, 0, 3, 6])
ax2[1].set_yticklabels(['-6', '-3', '0', '3', '6'])

for ax in ax1 + ax2:
    ax.legend(frameon = False)
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.set_xlabel('Transmission delay (ms)', fontsize = 13)
    ax.set_xlim(-0.9, 32.5)

################################################################
fig.subplots_adjust(left = 0.08, bottom = 0.1, right = 0.98, top = 0.9)
plt.savefig('delay_effect.png', dpi = 400)
plt.savefig('delay_effect.pdf')
