import numpy as np
pi = np.pi
import pickle
################################################################
from mpi4py import MPI
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.size

################################################################
N = 1000
dt = 0.001  # Sampling time interval
ttot = 100  ##
t = np.arange(0, ttot, dt)
################################################################
frequencies = np.concatenate([np.arange(1, 92), np.arange(92, 201, 4)])
nfreq = len(frequencies)

nbins = 40  ### phase densities histograms
def CV(x): return np.std(x)/np.mean(x)
################################################################
def analyze(spk, freq):
    isis = np.diff(spk)
    rate = 1./(np.mean(isis))
    cv = CV(isis)
    ######
    Sph = (spk*freq)%1  ## spikes phase aligned to stim freq [0, 1]
    pdens = np.histogram(Sph, range = [0, 1], bins = nbins, density=True)[0]
    ######
    nspk = len(spk)
    xv, yv = np.sum(np.cos(2*pi*Sph)), np.sum(np.sin(2*pi*Sph))
    VS = np.sqrt(xv**2 + yv**2)/nspk
    VA = (np.arctan2(yv, xv)/(2*pi))%1
    return pdens, VS, VA, rate, cv

################################################################
################################################################
################################################################
for i, delay in enumerate(['hetero', '0', '4', '8', '12']):
    if (rank==i):
        Rates = np.zeros((N, nfreq))
        CVs = np.zeros((N, nfreq))
        VStren = np.zeros((N, nfreq))
        VAngle = np.zeros((N, nfreq))
        Phase_dens = np.zeros((N, nfreq, nbins))
        for j, freq in enumerate(frequencies):
            with open('data/spikes_delay_%s_freq_%d.pkl'%(delay, freq), 'rb') as f:
                spikes = pickle.load(f)
            ####
            for k, spk in enumerate(spikes):
                pdens, VS, VA, rate, cv =  analyze(spk, freq)
                Rates[k, j] = rate
                CVs[k, j] = cv
                VStren[k, j] = VS
                VAngle[k, j] = VA
                Phase_dens[k, j] = pdens
            ####
        np.save('data_analysis/Rates_%s.npy'%(delay), Rates)
        np.save('data_analysis/CVs_%s.npy'%(delay), CVs)
        np.save('data_analysis/VStren_%s.npy'%(delay), VStren)
        np.save('data_analysis/VAngle_%s.npy'%(delay), VAngle)
        np.save('data_analysis/Phase_dens_%s.npy'%(delay), Phase_dens)
################################################################
