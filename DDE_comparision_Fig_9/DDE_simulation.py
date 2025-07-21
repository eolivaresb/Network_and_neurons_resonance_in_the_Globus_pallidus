import numpy as np
from scipy import signal
import matplotlib.pyplot as plt
import pickle
import matplotlib.gridspec as gridspec
import scipy.stats as stats
################################################################
import matplotlib
matplotlib.rcParams.update({'text.usetex': False, 'font.family': 'stixgeneral', 'mathtext.fontset': 'stix',})
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42
matplotlib.rcParams["pdf.use14corefonts"] = True
matplotlib.rcParams["ps.useafm"] = True
################################################################
####  Load Network final data
################################################################
delays_nn = np.arange(0, 32, 0.5)
with open('./final_data_network.pkl', 'rb') as f:  # Use 'wb' for binary mode
    data_nn = pickle.load(f)
peaks_psd_nn = data_nn['hub']['peaks_psd']
mean_rate_nn = data_nn['hub']['mean_rate']
std_rate_nn = data_nn['hub']['std_rate']
min_rate_nn = data_nn['hub']['min_rate']
max_rate_nn = data_nn['hub']['max_rate']
min_autoc_nn = data_nn['hub']['min_autoc']
max_autoc_nn = data_nn['hub']['max_autoc']
################################################################
################################################################
tau = 0.002
C = 4
W = 4
Mg, Bg = 120, 30
################################################################
################################################################
def spectrum(S, dt):
    # Autocorrelation
    autocorr = signal.correlate(S, S, mode='full')
    autocorr = autocorr[len(S)-1:]  # Take only positive lags
    # Power Spectral Density
    freqs, psd = signal.welch(S, fs=fs, nperseg=nper)
    return freqs, psd, (autocorr/(S.sum()))[1:1001]
################################################################
################################################################
def find_peak(f, y):
    ''' No big second peaks'''
    if np.max(np.abs(np.diff(psd))) < 10**-1 : return 1000
    pindx = np.argmax(y[2:])
    return f[2:][pindx]

################################################################
ttot = 30
tadapt = 1
dt = 0.0001
fs = int(1./dt)
nper = 100000
################################################################
def F(x):
  """Sigmoid function."""
  return Mg / (1 + (Mg-Bg)/Bg * np.exp(-4*x / Mg))
################################################################

def simulate_dde(tau, C, W, tdel):
    t_values = np.arange(-tadapt, ttot, dt)
    G_values = np.zeros(len(t_values))
    G_values[0] = Bg
    # Initialize history buffer in Bg
    if tdel == 0:
        for i in range(1, len(t_values)):
            t = t_values[i]
            dGdt = (F(C - W * G_values[i - 1]) - G_values[i - 1]) / tau
            G_values[i] = G_values[i - 1] + dGdt * dt
    else:
        history_buffer = [(t, Bg) for t in np.arange(-tadapt - tdel, -tadapt, dt)]
        for i in range(1, len(t_values)):
            t = t_values[i]
            # Find G(t - tdel) from the history buffer
            G_delayed = np.interp(t - tdel, [item[0] for item in history_buffer], [item[1] for item in history_buffer])
            dGdt = (F(C - W * G_delayed) - G_values[i - 1]) / tau
            G_values[i] = G_values[i - 1] + dGdt * dt
            # Update history buffer
            history_buffer.append((t, G_values[i]))
            history_buffer.pop(0)  # Remove oldest value
    return t_values[int(tadapt/dt):], G_values[int(tadapt/dt):]

delays = np.arange(0, 32, 0.5)
ndel = len(delays)

# C = 109
# W = 1.4
# tau = 0.0032
################################################################
peaks_psd = np.zeros(ndel)
mean_rate, std_rate = np.zeros(ndel), np.zeros(ndel)
min_rate, max_rate = np.zeros(ndel), np.zeros(ndel)
min_autoc, max_autoc = np.zeros(ndel), np.zeros(ndel)
################################################################
plt .close('all')
fig = plt.figure(figsize = [10, 5.])
gm = gridspec.GridSpec(170, 170)
ax1 = plt.subplot(gm[:, :70])
ax2 = plt.subplot(gm[:, 100:170])
ax22 = ax2.twinx()
################################################################
################################################################
for d, tdel in enumerate(delays):
    t_values, G_values = simulate_dde(tau, C, W, tdel/1000.)
    freqs, psd, autoc = spectrum(G_values, dt)
    peaks_psd[d] = find_peak(freqs, psd)
    min_autoc[d], max_autoc[d] = np.min(autoc[1:]), np.max(autoc[1:])
    mean_rate[d], std_rate[d], min_rate[d], max_rate[d] = np.mean(G_values), np.std(G_values), np.min(G_values), np.max(G_values)
################################################################
ax1.plot(delays, 1000./peaks_psd, '.k', label = 'DDE')
ax2.plot(delays, (max_autoc - min_autoc)/2, '.k')
ax2.plot(delays, (min_autoc - max_autoc)/2, '.k')
################################################################
ax1.plot(delays_nn, 1000./peaks_psd_nn, '.r', label = 'NN')
ax22.plot(delays_nn, (max_autoc_nn - min_autoc_nn)/2, '.r')
ax22.plot(delays_nn, (min_autoc_nn - max_autoc_nn)/2, '.r')
################################################################

data_dde = {}
data_dde['peaks_psd'] = peaks_psd
data_dde['mean_rate'] = mean_rate
data_dde['std_rate'] = std_rate
data_dde['min_rate'] = min_rate
data_dde['max_rate'] = max_rate
data_dde['min_autoc'] = min_autoc
data_dde['max_autoc'] = max_autoc

with open('./final_dde.pkl', 'wb') as f:  # Use 'wb' for binary mode
    pickle.dump(data_dde, f)

ax1.legend()
for ax in [ax1, ax2]:
    ax.set_xlabel('Transmission delay (ms)', fontsize = 12)
################################################################
ax1.set_ylabel('Peak PSD period (ms)', fontsize = 12)
ax2.set_ylabel('Autoc amplitude DDE', fontsize = 12)
ax22.set_ylabel('Autoc amplitude NN', fontsize = 12, rotation = -90, labelpad = 15)
################################################################
plt.figtext(0.1, 0.97, 'DDE v/s NN      C = %.1f   W = %.1f    tau = %.1f, Mg = %.1f    Bg = %.1f'%(C, W, 1000*tau, Mg, Bg), fontsize = 13)
fig.subplots_adjust(left = 0.06, bottom = 0.12, right = 0.94, top = 0.94)
plt.savefig('delay_DDE_NN_%d.png'%indx, dpi = 400)
################################################################
