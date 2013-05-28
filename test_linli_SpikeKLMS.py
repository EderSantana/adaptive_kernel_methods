import slash2; reload(slash2)
import spike_klms2; reload(spike_klms2)
import numpy as np
import tseries as ts; reload(ts)
from spike_klms2 import SpikeKLMS
import pylab as pl
from matplotlib import gridspec
from copy import copy
from sklearn.preprocessing import scale

pl.close('all')

time_step = 2
time_window = [-20, 80]
T = 2000
simulation = np.load('linli2_output.npy')
spike_events = simulation[:-1]
spike_events[0] = np.floor(spike_events[0])
target = simulation[-1]
target = target[range(0,target.shape[0],time_step)]

input = ts.nest_2_input(spike_events, T, time_window=time_window, \
        time_step=time_step)

sklms = SpikeKLMS(kernel="pop_mci", growing_criterion='dense', growing_param=[2., 5.], \
        ksize=5,learning_rate=.0005, n_jobs=4)
print sklms

sklms.fit_transform(copy(input), target[:len(input)])
G = gridspec.GridSpec(3,1)
pl.figure()
pl.subplot(G[:-1])
pl.plot(spike_events[0], spike_events[1], '.')
pl.subplot(G[-1])
pl.plot(target)
pl.plot(sklms.X_transformed_)
pl.show()

# Scaled figure
st = scale(target)
sout = scale(sklms.X_transformed_)

pl.figure()
pl.subplot(G[:-1])
pl.plot(spike_events[0], spike_events[1], '.')
pl.subplot(G[-1])
pl.plot(st)
pl.plot(sout)
pl.plot(sklms.centerIndex_, st[sklms.centerIndex_], '*r')
pl.show()

