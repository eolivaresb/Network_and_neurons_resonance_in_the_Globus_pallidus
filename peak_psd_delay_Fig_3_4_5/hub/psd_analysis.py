import numpy as np
from scipy import signal
import pickle
################################################################
from mpi4py import MPI
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.size
################################################################
# Define your time series
N = 1000
dt = 0.001  # Sampling time interval
ttot = 400  ##
t = np.arange(0, ttot, dt)
######
xpsd = np.arange(0, 500.1, 0.1)
xaut = np.arange(1, 1001)
################################################################
def binned_spikes(spikes):
    ######
    len_binned_spikes = int(ttot/dt)
    nspks = np.zeros((N, len_binned_spikes))
    ######
    for n in range(N):
        spk_indx = (spikes[n]/dt).astype(int)
        for s in spk_indx:
            nspks[n, s%len_binned_spikes] += 1
    return nspks
################################################################
################################################################
def spectrum(signal_data, dt, ttot):
    # Autocorrelation
    twindow = 10 ## 10 seconds => df = 0.1 Hz
    len_window = int(10/dt)
    autocorr = signal.correlate(signal_data, signal_data, mode='full')
    autocorr = autocorr[len(signal_data)-1:]  # Take only positive lags
    # Power Spectral Density
    freqs, psd = signal.welch(signal_data, fs=1/dt, nperseg=min(10000, len(signal_data)))
    return psd/(dt * signal_data.sum()/(ttot/twindow)), (autocorr/(dt * signal_data.sum()))[1:1001]
################################################################
################################################################
PSD = {}
for k, noise in enumerate(['noiseless', 'shared']):
    PSD[noise] = {}
    ################################################
    for delay in np.concatenate([np.arange(15), [15, 20, 25]]):
        ####
        with open('./data_%s/spikes_delay_%s.pkl'%(noise, delay), 'rb') as f:  # Load
            spk = pickle.load(f)
        spk_matrix = binned_spikes(spk)
        spk_matrix_copy = np.copy(spk_matrix)
        for s in spk_matrix_copy:
            np.random.shuffle(s)
        ################################################################
        ############       Network  analysis                 ###########
        p, a = spectrum(np.mean(spk_matrix, axis = 0), dt, ttot)
        p_norm, a_norm = spectrum(np.mean(spk_matrix_copy, axis = 0), dt, ttot)
        PSD[noise][delay] = p/np.mean(p_norm)
        ################################################################
with open('./PSD.pkl', 'wb') as f:  # Use 'wb' for binary mode
    pickle.dump(PSD, f)


'''
################################################################
for k, noise in enumerate(['noiseless', 'indep', 'shared']):
    if rank == k:
        PSD = {}
        Aut = {}
        for g in ['cells', 'groups', 'net']:
            PSD[g] = {}
            Aut[g] = {}
        ################################################
        for delay in ['hetero', '0', '4', '8', '12']:
            ####
            with open('./data_%s/spikes_delay_%s.pkl'%(noise, delay), 'rb') as f:  # Load
                spk = pickle.load(f)
            spk_matrix = binned_spikes(spk)
            ################################################################
            ############   Cells analysis                 ##################
            psd = np.zeros((N, len(xpsd)))
            aut = np.zeros((N, len(xaut)))
            for n in range(N):
                psd[n], aut[n] = spectrum(spk_matrix[n], dt, ttot)
            PSD['cells'][delay] = psd
            Aut['cells'][delay] = aut
            ################################################################
            ################################################################
            ############   Small groups analysis                 ###########
            psd = np.zeros((N//10, len(xpsd)))
            aut = np.zeros((N//10, len(xaut)))
            for n in range(N//10):
                psd[n], aut[n] = spectrum(np.mean(spk_matrix[n*10:(n+1)*10], axis = 0), dt, ttot)
            PSD['groups'][delay] = psd
            Aut['groups'][delay] = aut
            ################################################################
            ############       Network  analysis                 ###########
            psd, aut = spectrum(np.mean(spk_matrix, axis = 0), dt, ttot)
            PSD['net'][delay] = psd
            Aut['net'][delay] = aut
            ################################################################
        with open('data_analysis/PSD_%s.pkl'%(noise), 'wb') as f:  # Use 'wb' for binary mode
            pickle.dump(PSD, f)
        with open('data_analysis/Aut_%s.pkl'%(noise), 'wb') as f:  # Use 'wb' for binary mode
            pickle.dump(Aut, f)

'''
