import numpy as np
import os
import pandas as pd
import struct
import pickle
################################################################
################################################################
N = 1000
##########################################################
ttot = 100### seconds
Iamp = 20   ### pA
##############      Simulation     #############################
################################################################
frequencies = np.concatenate([np.arange(1, 92), np.arange(92, 201, 4)])
##############      Simulation     #############################
################################################################
from mpi4py import MPI
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.size

def save_spikes(freq, delay):
    '''
    save spikes on one file,
    N elements in a list containing the spiketimes for each neuron
    name is a string containing delay file and stim freq used on simulation
    '''
    #### Load spikes times
    f = open('./Spikes_times.dat', mode='rb').read() ## read spikes
    format = '%dd'%(len(f)//8)
    spk = np.array(struct.unpack(format, f))
    #### Load spiking neurons
    spkn = np.array(pd.read_csv('./Spiking_neurons.dat', sep='\s+', header=None))
    spkn = spkn.reshape(len(spkn))
    #### add spikes
    spikes = [[] for _ in range(N)]
    for k, s in enumerate(spk):
        spikes[int(spkn[k]-1)].append(s)
    for i in range(N):
        spikes[i] = np.array(spikes[i])
    with open('../data/spikes_delay_%s_freq_%d.pkl'%(delay, freq), 'wb') as f:  # Save
        pickle.dump(spikes, f)
    # with open('data.pkl', 'rb') as f:  # Load
    #     loaded_data = pickle.load(f)
################################################################
for i in range(size):
    if (i==rank):
        os.mkdir('sim_%d'%(i))
        os.chdir('sim_%d'%(i))
        os.system('cp ../main/main ./')

k = 0
################################################################
for delay in ['hetero', '0', '4']:#, '8', '12']:
    for freq in frequencies:
    ################################################################
        if (rank==k%size):
            os.system('cp ../../simulation_files/delays_%s.dat ./delays.dat'%(delay))
            #### Run simulation
            os.system('./main %.1f %.2f'%(freq, ttot))
            os.system('mv MeanI.dat ../data/MeanI_delay_%s_freq_%d.dat'%(delay, freq))
            save_spikes(freq, delay)
            os.system('rm *dat')
        k+=1
