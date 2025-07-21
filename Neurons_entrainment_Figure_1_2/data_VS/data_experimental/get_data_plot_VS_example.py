import numpy as np
from pyhdf.SD import SD, SDC ## to load experimental hd4 files
from scipy import signal
from scipy import interpolate
pi = np.pi
import pickle
################################################################
dt = 0.000025   ## timestep for 40 kHz
tref = 0.01     ## 10 ms refractory period
vthr = -0.02    ## v threshold -20 mv

len_save_trace = int(2./dt)  #save 2 seconds of traces
###################################################
ttot_sweep = 4.0
tsweep = np.arange(0, ttot_sweep, dt)
ltrace = len(tsweep)
epsilon = 0.0000001
######################################
samplf = [17, 35, 52] ### cell rate = 34.67
window_width = 1
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
    filt_noise = []
    for f in samplf:
        ### calculate noise component
        w = np.zeros(len(a))
        w[np.where((freq>=f-window_width)*(freq<=f+window_width))] += 1
        w[np.where((freq>=-f-window_width)*(freq<=-f+window_width))] += 1
        af = a*w
        ncomp = np.real(np.fft.ifft(af)) ### noise component on this window
        filt_noise.append(ncomp[:len_save_trace])
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
    return int_func, filt_noise
################################################################
################################################################
def get_vs_f(Sph):
    nspk = len(Sph)
    xv, yv = np.sum(np.cos(2*pi*Sph)), np.sum(np.sin(2*pi*Sph))
    return np.sqrt(xv**2 + yv**2)/nspk, (np.arctan2(yv, xv)/(2*pi))%1
################################################################
def get_VS(int_func, spikes):
    vs, va, phases = np.zeros(len(samplf)), np.zeros(len(samplf)), []
    for k, func in enumerate(int_func):
        Sph = func(spikes)
        phases.append(Sph)
        vs[k], va[k] = get_vs_f(Sph)
    return vs, va, phases

############################
#################    Cell 13            #########################
path = './data_experiments'
rep = 4
vtrace_file = '%s/17Jun2021.2.%d.hdf'%(path, rep)
noise_file = '%s/17Jun2021.2.%d_stim.hdf'%(path, rep)
sine_file = '%s/17Jun2021.2.5.hdf'%(path)

# ############################
# ### Load noise stim trace
# data = SD(noise_file, SDC.READ)
# noiseSet = 10**12*data.select('Amplitude').get()
# ############################
# #### Load voltage trace
# data = SD(vtrace_file, SDC.READ)
# VoltsSet = data.select('Amplitude').get()
# nsweeps = len(VoltsSet)
# ############################
# C_noise = []
# ### Get only noised part of the recording
# for ns in range(nsweeps):
#     C_noise.append(noiseSet[ns][:ltrace])
# C_noise = np.concatenate(C_noise)
# ############################
# spikes = []
# for i, vtrace in enumerate(VoltsSet):
#     spikes.append(i*ttot_sweep + get_spikes(vtrace[:ltrace]))
# spikes = np.concatenate(spikes)
# ############################
# data = {}
#
# int_func_noise, filt_noise = interp_functions(C_noise, nsweeps)
# vs, va, phases = get_VS(int_func_noise, spikes)
#
# data['spikes'] = spikes
# data['filt_noise'] = filt_noise
# data['vtrace'] = VoltsSet[0][:len_save_trace]
# data['noise'] = noiseSet[0][:len_save_trace]
# data['vs'] = vs
# data['va'] = va
# data['phases'] = phases
#
# with open('data_experiments/data_figure_vs.pkl', 'wb') as f:  # Use 'wb' for binary mode
#     pickle.dump(data, f)


############################
#### Load voltage trace
data = SD(sine_file, SDC.READ)
Sine_traces = data.select('Amplitude').get()
t = data.select('time (sec)').get()

dt = 0.00005   ## timestep for 20 kHz
tref = 0.01     ## 10 ms refractory period
vthr = -0.02    ## v threshold -20 mv
###################################################
phases = []
vs, va, rates = np.zeros(100), np.zeros(100), np.zeros(100)
traces = []
for k, freq in enumerate(np.arange(1, 101)):
    vtrace = Sine_traces[k][10000:210000] ## remove first and last 0.5 seconds (no stimuli)
    traces.append(vtrace[20000:30000]) # save 0.5 s of recording starting on the t = 1.5 (already removed first 0.5 s)
    spikes = []
    tdel = tref
    for ktime, vd in enumerate(vtrace):
        tdel += dt
        if (tdel > tref)*(vd > vthr):
            spikes.append(dt*ktime)
            tdel = 0
    spikes = np.array(spikes)
    Sph = (spikes*freq)%1
    phases.append(Sph)
    vs[k], va[k] = get_vs_f(Sph)
    rates[k] = 1./np.mean(np.diff(spikes))

data = {}
data['rates'] = rates
data['vs'] = vs
data['va'] = va
data['phases'] = phases
data['traces'] = traces

with open('data_experiments/data_Sine_figure_vs.pkl', 'wb') as f:  # Use 'wb' for binary mode
    pickle.dump(data, f)
