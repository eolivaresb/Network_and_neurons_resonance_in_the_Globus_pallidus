import numpy as np
import pickle
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import scipy.stats as stats
################################################################
import matplotlib
# matplotlib.rcParams.update({'text.usetex': False, 'font.family': 'stixgeneral', 'mathtext.fontset': 'stix',})
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42
matplotlib.rcParams["pdf.use14corefonts"] = True
matplotlib.rcParams["ps.useafm"] = True
################################################################
################################################################
fz = 14
################################################################
dt = 0.000025   ## timestep for 40 kHz
len_save_trace = int(2./dt)  #save 2 seconds of traces
time = np.arange(0, 2, dt)
tsine = np.arange(0, 0.5, 0.00005)
###################################################
######################################
samplf = [17, 35, 52] ### cell rate = 34.67
with open('data_experiments/data_figure_vs.pkl', 'rb') as f:  # Use 'wb' for binary mode
    data = pickle.load(f)

spikes = data['spikes']
spikes = spikes[np.where(spikes<2)]
filt_noise = data['filt_noise']
vtrace = data['vtrace']
noise = data['noise']
vs = data['vs']
va = data['va']
phases = data['phases']

f_sine = [15, 31, 47] ### cell rate = 34.67
with open('data_experiments/data_Sine_figure_vs.pkl', 'rb') as f:  # Use 'wb' for binary mode
    data = pickle.load(f)

rates_sine = data['rates']
vs_sine = data['vs']
va_sine = data['va']
phases_sine = data['phases']
traces_sine = data['traces']
################################################################
plt .close('all')
fig = plt.figure(figsize = [14, 8.75])
gm = gridspec.GridSpec(490, 340)
ax1 = plt.subplot(gm[:90, :100])
ax2 = plt.subplot(gm[100:190, :100])
ax3 = plt.subplot(gm[200:290, :100])
ax4 = plt.subplot(gm[300:390, :100])
ax5 = plt.subplot(gm[400:490, :100])

ax32 = plt.subplot(gm[200:290, 120:160])
ax42 = plt.subplot(gm[300:390, 120:160])
ax52 = plt.subplot(gm[400:490, 120:160])

ax33 = plt.subplot(gm[200:290, 180:220])
ax43 = plt.subplot(gm[300:390, 180:220])
ax53 = plt.subplot(gm[400:490, 180:220])

ax34 = plt.subplot(gm[200:290, 240:340])
ax44 = plt.subplot(gm[300:390, 240:340])
ax54 = plt.subplot(gm[400:490, 240:340])
ax35 = ax34.twinx()
ax45 = ax44.twinx()
ax55 = ax54.twinx()
################################################################
ax1.plot(time, 1000*vtrace, 'k')
ax2.plot(time, noise, 'k')
for k, ax in enumerate([ax3, ax4, ax5]):
    ax.plot(time, filt_noise[k], 'k')

for ax in [ax5, ax54]:
    ax.annotate("", xy=(0.8, -.12), xycoords='axes fraction', xytext=(1., -.12), textcoords = 'axes fraction', arrowprops=dict(arrowstyle='-', linewidth=2), clip_on=False)
    ax.text(0.95, -0.32, '200 ms', fontsize = 10, ha = 'center', transform = ax.transAxes)

for a in [ax2, ax3, ax4, ax5]:
    for s in spikes:
        a.axvline(x = s, color = 'r', linestyle = '--', linewidth = 1.2)

for a in [ax1, ax2, ax3, ax4, ax5]:
    a.set_xlim(1, 1.5)
    a.spines['right'].set_visible(False)
    a.spines['top'].set_visible(False)
    a.spines['bottom'].set_visible(False)
    a.set_xticks([])

for k, ax in enumerate([ax3, ax4, ax5]):
    ax.set_ylim(-4.2, 4.2)
    ax.axhline(y = 0, color = 'k')

ax1.set_ylabel('Vm (mV)', fontsize = fz)
ax2.set_ylabel('I' + r'$_{app}$' + ' (pA)', fontsize = fz)
ax3.set_ylabel('I' + r'$_{app}$' + ' (pA)\n0.5 x cell rate', fontsize = fz)
ax4.set_ylabel('I' + r'$_{app}$' + ' (pA)\ncell rate', fontsize = fz)
ax5.set_ylabel('I' + r'$_{app}$' + ' (pA)\n1.5 x cell rate', fontsize = fz)

################################################################
for k, ax in enumerate([ax32, ax42, ax52]):
    ax.hist(phases[k], density=True, bins = 20, alpha=0.87, range = [0, 1], color = 'k')
    ax.set_ylim(0, 1.35)
################################################################
for k, ax in enumerate([ax33, ax43, ax53]):
    ax.hist(phases_sine[f_sine[k]], density=True, bins = 20, alpha=0.87, range = [0, 1], color = 'k')
    ax.set_ylim(0, 4.25)
################################################################
for ax in [ax32, ax42, ax52, ax33, ax43, ax53]:
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.set_xticks([0, 0.5, 1], ['0', '0.5', '1'])
    ax.set_xlim(0, 1)
    ax.set_ylabel('PDF', fontsize = fz)
for ax in [ax32, ax42, ax33, ax43]:
    ax.set_xticklabels([])
for ax in [ax52, ax53]:
    ax.set_xlabel('Stim phase ('+ r'$\theta$' + ')', fontsize = fz)
################################################################
################################################################

for k, ax in enumerate([ax35, ax45, ax55]):
    ax.plot(tsine, 20 * np.sin(2* np.pi* tsine* (f_sine[k]+1)), 'b', lw = 1.2, alpha = 0.8)

for k, ax in enumerate([ax34, ax44, ax54]):
    ax.plot(tsine, 1000 * traces_sine[f_sine[k]], 'k')

for a in [ax34, ax35, ax44, ax45, ax54, ax55]:
    a.set_xlim(0, .5)
    a.spines['top'].set_visible(False)
    a.spines['bottom'].set_visible(False)
    a.set_xticks([])

ax2.set_yticks([-100, 0, 100], ['-100', '0', '100'])

for a in [ax3, ax4, ax5]:
    a.set_yticks([-3, 0, 3], ['-3', '0', '3'])

for a in [ax1, ax34, ax44, ax54]:
    a.set_yticks([-50, -25, 0, 25], ['-50', '-25', '0', '20'])

for a in [ax35, ax45, ax55]:
    a.spines['left'].set_visible(False)
    a.set_ylim(-36, 36)
    a.set_yticks([-20, 0, 20], ['-20', '0', '20'])


ax34.set_ylabel('Vm (mV)', fontsize = fz)
ax44.set_ylabel('Vm (mV)', fontsize = fz)
ax54.set_ylabel('Vm (mV)', fontsize = fz)

ax35.set_ylabel('I' + r'$_{app}$' + ' (pA)\n0.5 x cell rate', fontsize = fz, color = 'b', rotation = -90, labelpad = 32)
ax45.set_ylabel('I' + r'$_{app}$' + ' (pA)\ncell rate', fontsize = fz, color = 'b', rotation = -90, labelpad = 32)
ax55.set_ylabel('I' + r'$_{app}$' + ' (pA)\n1.5 x cell rate', fontsize = fz, color = 'b', rotation = -90, labelpad = 32)
################################################################

###############################################################
x1, x2, x3, x4 = 0.001, 0.33, 0.48, 0.63
y0, y1, y2, fz = 0.97, 0.78, 0.6, 26
plt.figtext(x1, y0, 'A', ha = 'left', va = 'center', fontsize = fz)
plt.figtext(x1, y1, 'B', ha = 'left', va = 'center', fontsize = fz)
plt.figtext(x1, y2, 'C', ha = 'left', va = 'center', fontsize = fz)
plt.figtext(x2, y2, 'D', ha = 'left', va = 'center', fontsize = fz)
plt.figtext(x3, y2, 'E', ha = 'left', va = 'center', fontsize = fz)
plt.figtext(x4, y2, 'F', ha = 'left', va = 'center', fontsize = fz)
################################################################
################################################################
fig.subplots_adjust(left = 0.06, bottom = 0.09, right = 0.94, top = 0.96)
plt.savefig('./VS_example.png', dpi = 400, transparent=False)
plt.savefig('./VS_example.pdf')
