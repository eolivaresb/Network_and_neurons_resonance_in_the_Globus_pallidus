import numpy as np
import pickle
import matplotlib.pylab as plt
import matplotlib.gridspec as gridspec
################################################################
import matplotlib
matplotlib.rcParams.update({'text.usetex': False, 'font.family': 'stixgeneral', 'mathtext.fontset': 'stix',})
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42
################################################################
################################################################
delays = np.concatenate([np.arange(16), [20, 25]])
################################################################
connected = np.loadtxt('connected_pairs.txt').astype(int)
non_connected = np.loadtxt('non_connected_pairs.txt').astype(int)
################################################################
# with open('./rates.pkl', 'rb') as f: rates = pickle.load(f)
# rates = rates['noiseless']['0']
rates = np.loadtxt('rates.txt')

npairs = 2000
g_rate_connected, g_rate_non_connected = np.zeros(npairs), np.zeros(npairs)
for k, pair in enumerate(connected):
    g_rate_connected[k] = np.sqrt((rates[pair[0]]*rates[pair[1]]))
    g_rate_non_connected[k] = np.sqrt((rates[pair[0]]**2 + rates[pair[1]]**2)/2.)

indx_connected = g_rate_connected.argsort()
indx_non_connected = g_rate_non_connected.argsort()
################################################################
with open('./Coherence_hub.pkl', 'rb') as f: C = pickle.load(f)
cond = 'noiseless'

fz = 13
################################################################
def plot_imshow(ax, data):
    im = ax.imshow(data, cmap='jet', vmin=-0.01, vmax=0.07, aspect='auto')
    ax.set_xlim(0.1, 200)
    # ax.set_xticks([25, 50, 75, 100])
    # ax.set_xticklabels(['25', '50', '75', '100'])
    ax.set_yticks([])
    # ax.set_yticklabels(['1', '2000'])
    return im
################################################################

plt .close('all')
fig = plt.figure(figsize = [14, 8])
gm = gridspec.GridSpec(280, 573)
axes = [plt.subplot(gm[(i//6)*100:(i//6)*100+73, (i%6)*100:(i%6)*100+73]) for i in range(18)]

# axl1 = fig.add_axes([0.93, 0.64, 0.01, 0.23])
# axl2 = fig.add_axes([0.93, 0.31, 0.01, 0.23])
################################################################
# ##### Titles
# y1 = 1.1
# y2 = 0.0
# ax1[0].text(1.2, y1-0.03, 'Transmission delay =  4 ms', fontsize = 14, ha = 'center', transform = ax1[0].transAxes)
# ax2[0].text(1.2, y1+y2-0.03, 'Transmission delay = 12 ms', fontsize = 14, ha = 'center', transform = ax2[0].transAxes)
#
# arrowprops=dict(arrowstyle='|-|, widthA=0.2,widthB=0.2', linewidth = 1.25, edgecolor = 'k')
# ax1[0].annotate("",  xy= [0.0, y1], xycoords='axes fraction', xytext = [0.72,y1], textcoords = 'axes fraction', arrowprops=arrowprops)
# ax2[0].annotate("",  xy= [0.0, y2+y1], xycoords='axes fraction', xytext = [0.72,y2+ y1], textcoords = 'axes fraction', arrowprops=arrowprops)
#
# ax1[0].annotate("",  xy= [1.68, y1], xycoords='axes fraction', xytext = [2.42,y1], textcoords = 'axes fraction', arrowprops=arrowprops)
# ax2[0].annotate("",  xy= [1.68,y2+ y1], xycoords='axes fraction', xytext = [2.42,y2+ y1], textcoords = 'axes fraction', arrowprops=arrowprops)

################################################################
for k, d in enumerate(delays):
    ################################################################
    plot_imshow(axes[k], (C[cond]['connected'][d][indx_connected]))
    axes[k].set_title('delay = %d ms'%d)
    # plot_imshow(axes[k*2+1], (C[cond]['non_connected'][d][indx_non_connected])**(0.5))
    # axes[k*2+1].set_title('non_connected')
    print('max_con = %.3f   max_non_con = %.3f'%(np.min(C[cond]['connected'][d]), np.max(C[cond]['connected'][d])))
    # print('min_con = %.3f   min_non_con = %.3f'%(np.min(C[cond]['connected'][d]), np.min(C[cond]['non_connected'][d])))

    axes[k].plot(g_rate_connected[indx_connected], np.arange(npairs), '.w', ms = 0.06)
    # axes[2*k+1].plot(g_rate_non_connected[indx_non_connected], np.arange(npairs), '.w', ms = 0.06)

    ticks = [10, 20, 30, 40, 100]
    p = 0
    ticks_con = []
    for cindx in range(npairs):
        if g_rate_connected[indx_connected][cindx] > ticks[p]:
            ticks_con.append(cindx)
            p+=1

# for ax in ax1 + ax2:
#     ax.set_yticks(ticks_con)
#     ax.set_yticklabels([str(c) for c in ticks[:-1]])
#     ax.set_ylabel('Neurons pair\nmean rate (spk/s)', fontsize = fz, labelpad = 0)
#
# for ax in ax2:  ax.set_xlabel('Frequency (Hz)', fontsize = fz)
#
# plt.figtext(0.25, 0.94, 'Monosynaptically\nConnected pairs', fontsize = 14, ha = 'center')
# plt.figtext(0.72, 0.94, 'Non directly\nConnected pairs', fontsize = 14, ha = 'center')
#
#
# ################################################################
# f = np.arange(0, 500.1, 0.1)
# for k, d in enumerate(delays):
#     ax3[0].plot(f, np.mean(C['connected'][d], axis = 0), color = c[k], label = 'delay = %s ms'%d)
#     ax3[1].plot(f, np.mean(C['non_connected'][d], axis = 0), color = c[k], label = 'delay = %s ms'%d)
# ax3[0].legend(frameon = False)
#
# for a in ax3:
#     a.set_xlim(0.2, 105)
#     a.set_ylim(0, 0.1)
#     a.set_xlabel('Frequency (Hz)', fontsize = fz)
#     a.set_ylabel('Coherence', fontsize = fz)
#     a.set_yticks([0, 0.1])
#     a.spines['right'].set_visible(False)
#     a.spines['top'].set_visible(False)
#
# ###############################################################
# x1, x2 = 0.01, 0.48
# y1, y2, y3, fz = 0.9, 0.57, 0.25, 23
# plt.figtext(x1, y1, 'A', ha = 'left', va = 'center', fontsize = fz)
# plt.figtext(x1, y2, 'B', ha = 'left', va = 'center', fontsize = fz)
# plt.figtext(x1, y3, 'C', ha = 'left', va = 'center', fontsize = fz)
# plt.figtext(x2, y3, 'D', ha = 'left', va = 'center', fontsize = fz)
################################################################
fig.subplots_adjust(left = 0.04, bottom = 0.06, right = 0.96, top = 0.88)
#plt.savefig('figures/Coherence_intrinsic.png', dpi = 400)
plt.savefig('./Coherence_%s.png'%(cond), dpi = 400)
