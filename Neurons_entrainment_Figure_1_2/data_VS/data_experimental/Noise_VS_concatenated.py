################################################################
import numpy as np
from pyhdf.SD import SD, SDC ## to load experimental hd4 files
from scipy import signal
from scipy import interpolate
pi = np.pi
################################################################
from mpi4py import MPI
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.size

######################################
dt = 0.000025   ## timestep for 40 kHz
tref = 0.01     ## 10 ms refractory period
vthr = -0.02    ## v threshold -20 mv
###################################################
ttot_sweep = 4.0
tsweep = np.arange(0, ttot_sweep, dt)
ltrace = len(tsweep)

epsilon = 0.0000001
######################################
samplf = np.arange(2, 203, 1)
window_width = 1
# window = signal.windows.tukey(int(ttot*(2*window_width)), alpha = 0.25)
################################################################
#### Detect spikes
def get_spikes(vtrace):
    spikes = []
    tdel = tref
    for ktime, vd in enumerate(vtrace):
        tdel += dt
        if (tdel > tref)*(vd > vthr):
            spikes.append(dt*ktime)
            tdel = 0
    return np.array(spikes)
################################################################
def interp_functions(noise, nsweeps):
    ttot = ttot_sweep * nsweeps
    time = np.arange(0, ttot, dt)
    freq = np.fft.fftfreq(len(time), dt)

    a = np.fft.fft(noise)
    int_func = []
    for f in samplf:
        ### calculate noise component
        w = np.zeros(len(a))
        w[np.where((freq>=f-window_width)*(freq<=f+window_width))] += 1
        w[np.where((freq>=-f-window_width)*(freq<=-f+window_width))] += 1
        af = a*w
        ncomp = np.real(np.fft.ifft(af)) ### noise component on this window
        ### calculate zero crossings
        zcindx = np.where(np.diff(np.sign(ncomp))==2) ## zeros crossing indexes
        lcomp, rcomp = ncomp[zcindx], ncomp[1:][zcindx] ## component values around zcross
        interp_time = dt*(lcomp/(lcomp-rcomp))   ## time interpolation
        tz = time[zcindx]+interp_time              ## time on the left plus interpolation
        ### Add interpolation function
        int_vector = [] ## vector to interpolate spike phase
        int_vector.append([0, 0])
        for t in tz:
            int_vector.append([t-epsilon, 1])
            int_vector.append([t+epsilon, 0])
        int_vector.append([ttot,1])
        int_vector = np.array(int_vector).T
        int_func.append(interpolate.interp1d(int_vector[0],int_vector[1]))
    return int_func
################################################################
################################################################
def get_vs_f(Sph):
    nspk = len(Sph)
    xv, yv = np.sum(np.cos(2*pi*Sph)), np.sum(np.sin(2*pi*Sph))
    return np.sqrt(xv**2 + yv**2)/nspk, (np.arctan2(yv, xv)/(2*pi))%1
################################################################
def get_VS(int_func, spikes):
    vs, va = np.zeros(len(samplf)), np.zeros(len(samplf))
    for k, func in enumerate(int_func):
        Sph = func(spikes)
        vs[k], va[k] = get_vs_f(Sph)
    return vs, va
############################
def process_data(cell, Volt_files, Noise_files):
    ############################
    ### Load noise stim trace
    data = SD(Noise_files, SDC.READ)
    noiseSet = 10**12*data.select('Amplitude').get()
    ############################
    #### Load voltage trace
    data = SD(Volt_files, SDC.READ)
    VoltsSet = data.select('Amplitude').get()
    ### remove bad sweeps in some cells
    if cell == 3:
        VoltsSet = VoltsSet[3:30]
        noiseSet = noiseSet[3:30]
    if cell == 10:
        VoltsSet = list(VoltsSet)
        del VoltsSet[11]
        noiseSet = list(noiseSet)
        del noiseSet[11]
    nsweeps = len(VoltsSet)
    ############################
    C_noise = []
    ### Get only noised part of the recording
    for ns in range(nsweeps):
        C_noise.append(noiseSet[ns][:ltrace])
    C_noise = np.concatenate(C_noise)
    ############################
    spikes = []
    for i, vtrace in enumerate(VoltsSet):
        spikes.append(i*ttot_sweep + get_spikes(vtrace[:ltrace]))
    spikes = np.concatenate(spikes)
    ############################
    Sphases = [[] for _ in range(len(samplf))]

    int_func_noise = interp_functions(C_noise, nsweeps)

    np.savetxt('data/VS_all_%d.txt'%(cell), get_VS(int_func_noise, spikes))

k = 0
#################    Cell 1             #########################
k+=1
if (k%size == rank):
    path = '../raw_data_sinewaves/13Jan2021/Cell4/ccElectricNoisePRC'
    rep = 6
    vtrace_file = '%s/13Jan2021.4.%d.hdf'%(path, rep)
    noise_file = '%s/13Jan2021.4.%d_stim.hdf'%(path, rep)
    process_data(1, vtrace_file, noise_file)

#################    Cell 2             #########################
k+=1
if (k%size == rank):
    path = '../raw_data_sinewaves/13Jan2021/Cell7/ccElectricNoisePRC'
    rep = 4
    vtrace_file = '%s/13Jan2021.7.%d.hdf'%(path, rep)
    noise_file = '%s/13Jan2021.7.%d_stim.hdf'%(path, rep)
    process_data(2, vtrace_file, noise_file)

#################    Cell 3             #########################
k+=1
if (k%size == rank):
    path = '../raw_data_sinewaves/26Jan2021/Cell2/ccElectricNoisePRC'
    rep = 18
    vtrace_file = '%s/26Jan2021.2.%d.hdf'%(path, rep)
    noise_file = '%s/26Jan2021.2.%d_stim.hdf'%(path, rep)
    remove = [0, 1, 2, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39]
    process_data(3, vtrace_file, noise_file)

################    Cell 4         ##############################
k+=1
if (k%size == rank):
    path = '../raw_data_sinewaves/26Jan2021/Cell3/ccElectricNoisePRC'
    rep = 18
    vtrace_file = '%s/26Jan2021.3.%d.hdf'%(path, rep)
    noise_file = '%s/26Jan2021.3.%d_stim.hdf'%(path, rep)
    process_data(4, vtrace_file, noise_file)

################    Cell 5         ##############################
k+=1
if (k%size == rank):
    path = '../raw_data_sinewaves/02Feb2021/Cell1/ccElectricNoisePRC'
    rep = 4
    vtrace_file = '%s/02Feb2021.1.%d.hdf'%(path, rep)
    noise_file = '%s/02Feb2021.1.%d_stim.hdf'%(path, rep)
    process_data(5, vtrace_file, noise_file)

################    Cell 6         ##############################
k+=1
if (k%size == rank):
    path = '../raw_data_sinewaves/09Feb2021/Cell2/ccElectricNoisePRC'
    rep = 11
    vtrace_file = '%s/09Feb2021.2.%d.hdf'%(path, rep)
    noise_file = '%s/09Feb2021.2.%d_stim.hdf'%(path, rep)
    process_data(6, vtrace_file, noise_file)

################    Cell 7         ##############################
k+=1
if (k%size == rank):
    path = '../raw_data_sinewaves/10Mar2021/Cell2/ccElectricNoisePRC'
    rep = 4
    vtrace_file = '%s/10Mar2021.2.%d.hdf'%(path, rep)
    noise_file = '%s/10Mar2021.2.%d_stim.hdf'%(path, rep)
    process_data(7, vtrace_file, noise_file)

################    Cell 8            #########################
k+=1
if (k%size == rank):
    path = '../raw_data_sinewaves/06Apr2021/Cell4/ccElectricNoisePRC'
    rep = 6
    vtrace_file = '%s/06Apr2021.4.%d.hdf'%(path, rep)
    noise_file = '%s/06Apr2021.4.%d_stim.hdf'%(path, rep)
    process_data(8, vtrace_file, noise_file)

#################    Cell 9            #########################
k+=1
if (k%size == rank):
    path = '../raw_data_sinewaves/13Apr2021/Cell2/ccElectricNoisePRC'
    rep = 6
    vtrace_file = '%s/13Apr2021.2.%d.hdf'%(path, rep)
    noise_file = '%s/13Apr2021.2.%d_stim.hdf'%(path, rep)
    process_data(9, vtrace_file, noise_file)

#################    Cell 10            #########################
k+=1
if (k%size == rank):
    path = '../raw_data_sinewaves/11May2021/Cell1/ccElectricNoisePRC'
    ####  remove trace 12 ####
    rep = 2
    vtrace_file = '%s/11May2021.1.%d.hdf'%(path, rep)
    noise_file = '%s/11May2021.1.%d_stim.hdf'%(path, rep)
    process_data(10, vtrace_file, noise_file)

#################    Cell 11            #########################
k+=1
if (k%size == rank):
    path = '../raw_data_sinewaves/01Jun2021/Cell4/ccElectricNoisePRC'
    rep = 11
    vtrace_file = '%s/01Jun2021.4.%d.hdf'%(path, rep)
    noise_file = '%s/01Jun2021.4.%d_stim.hdf'%(path, rep)
    process_data(11, vtrace_file, noise_file)

#################    Cell 12            #########################
k+=1
if (k%size == rank):
    path = '../raw_data_sinewaves/15Jun2021/Cell3/ccElectricNoisePRC'
    rep = 6
    vtrace_file = '%s/15Jun2021.3.%d.hdf'%(path, rep)
    noise_file = '%s/15Jun2021.3.%d_stim.hdf'%(path, rep)
    #### Sinewave amplitude = 10 pA ####
    process_data(12, vtrace_file, noise_file)

#################    Cell 13            #########################
k+=1
if (k%size == rank):
    path = '../raw_data_sinewaves/17Jun2021/Cell2/ccElectricNoisePRC'
    rep = 4
    vtrace_file = '%s/17Jun2021.2.%d.hdf'%(path, rep)
    noise_file = '%s/17Jun2021.2.%d_stim.hdf'%(path, rep)
    process_data(13, vtrace_file, noise_file)

#################    Cell 14            #########################
k+=1
if (k%size == rank):
    path = '../raw_data_sinewaves/01July2021/Cell1/ccElectricNoisePRC'
    rep = 6
    vtrace_file = '%s/01July2021.1.%d.hdf'%(path, rep)
    noise_file = '%s/01July2021.1.%d_stim.hdf'%(path, rep)
    process_data(14, vtrace_file, noise_file)

#################    Cell 15            #########################
k+=1
if (k%size == rank):
    path = '../raw_data_sinewaves/15July2021/Cell1/ccElectricNoisePRC'
    rep = 3
    vtrace_file = '%s/15July2021.1.%d.hdf'%(path, rep)
    noise_file = '%s/15July2021.1.%d_stim.hdf'%(path, rep)
    process_data(15, vtrace_file, noise_file)

################    Cell 16            #########################
k+=1
if (k%size == rank):
    path = '../raw_data_sinewaves/29July2021/Cell1/ccElectricNoisePRC'
    rep = 5
    vtrace_file = '%s/29July2021.1.%d.hdf'%(path, rep)
    noise_file = '%s/29July2021.1.%d_stim.hdf'%(path, rep)
    process_data(16, vtrace_file, noise_file)
# '''
