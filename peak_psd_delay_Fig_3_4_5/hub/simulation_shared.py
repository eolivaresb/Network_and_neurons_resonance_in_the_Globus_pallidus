import numpy as np
import time as clocktime
import os
import pandas as pd
import struct
import pickle
import time
################################################################
################################################################
N = 1000
n = 10
##########################################################
ttot = 400.0### seconds
##############      Simulation     #############################
################################################################

def save_spikes(delay):
    '''
    save spikes on one file,
    N elements in a list containing the spiketimes for each neuron
    name is a string containing delay file used on simulation
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
    with open('../data_shared/spikes_delay_%s.pkl'%(delay), 'wb') as f:  # Save
        pickle.dump(spikes, f)
    # with open('data.pkl', 'rb') as f:  # Load
    #     loaded_data = pickle.load(f)
################################################################
os.chdir('./main_shared_noise')
################################################################
for k, delay in enumerate(['15', '20', '25']):
    #####  create folder and copy executable compiled neural network
    d = int(delay) * np.ones(N*n)/1000.
    np.savetxt('./delays.dat', d, fmt = '%.5f')
    #### Run simulation
    os.system('./main %.1f'%(ttot))
    save_spikes(delay)
    os.system('mv noise.dat ../data_shared/noise_delay_%s.dat'%(delay))
    os.system('rm *dat')
