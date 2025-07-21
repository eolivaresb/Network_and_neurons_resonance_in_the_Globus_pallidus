import numpy as np
################################################################
# N = 1000
# ##########################################################
# for delay in np.arange(0, 15, 2):
#     #### delays uniformaly distributed range = 2ms celtered in "delay"
#     d = np.random.uniform((delay-1)/1000., (delay+1)/1000., N)
#     if delay == 0: d = np.zeros(N) ### zero delay has no stdv
#     np.savetxt('./delays_%d.dat'%delay, d, fmt = '%.5f')

################################################################
N = 1000
n = 10
##########################################################
delay = 10
d = np.random.uniform((delay-1)/1000., (delay+1)/1000., N*n)
# d = np.zeros(N*n)
np.savetxt('./delays.dat', d, fmt = '%.5f')
