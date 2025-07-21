import numpy as np
import time as clocktime
import os
import pandas as pd
import struct
import pickle
import time
################################################################
################################################################
N = 16
##########################################################

def save_spikes(sim):
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
    with open('./data_sim/spikes_%s.pkl'%(sim), 'wb') as f:  # Save
        pickle.dump(spikes, f)
    os.system('rm *dat')
################################################################
###simulate noise
os.system('./main 0')
os.system('mv noise.dat data_sim/noise.dat')
save_spikes('noise')

# for freq in range(1, 101):
#     os.system('./main %d'%freq)
#     save_spikes('freq_%d'%freq)
