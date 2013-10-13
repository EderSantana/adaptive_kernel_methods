import slash3; reload(slash3)
import spike_klms3; reload(spike_klms3)
import numpy as np
import tseries as ts; reload(ts)
from spike_klms3 import SpikeKLMS
import pylab as pl
from matplotlib import gridspec
from copy import copy
from sklearn.preprocessing import scale

pl.close('all')

time_step = 2
time_window = [-50, 20]
T = 2000
simulation = np.load('linli_generalization.npy')
spike_events = simulation[:-1]
spike_events[0] = np.floor(spike_events[0])
target = simulation[-1]
target = target[range(0,target.shape[0],time_step)]

input = ts.nest_2_input(spike_events, 3*T, time_window=time_window, \
        time_step=time_step)

#sklms = SpikeKLMS(kernel="pop_mci", growing_criterion='novelty', growing_param=[2., 10.], ksize=5,learning_rate=.0005, n_jobs=4)
sklms = SpikeKLMS(kernel="pop_mci", growing_criterion='dense', \
        growing_param=[1], ksize=5, learning_rate=.0005, \
        loss_function='least_squares', correntropy_sigma=1., n_jobs=1, \
        gamma=.01, dropout=1)
print sklms

# SKLMS with linear outter layer
sklms.fit_transform(input[:1], target[:1])
w = np.random.randn(2)
w[0] = 1
w[1] = 0
for i in range(1,T/2):
    print i
    dropin, dropin_coeff = sklms._dropout()
    gram                         = sklms._get_kernel([input[k] for k in dropin],\
                                                     input[i])
    sklms.X_online_              = np.hstack([sklms.X_online_, \
                                   #np.dot(dropin_coeff, gram)])
                                   w[1]+w[0]*np.dot(dropin_coeff, gram)])

    xout = sklms.X_online_[i]
    yout = w[0]*xout + w[1]
    #yout = xout
    e = target[i]-yout
    #w[0] = w[0] + 1*e*xout
    #w[1] = w[1] + .01*e
    err = e*w[0]
    sklms._trainNet(input, input[i], e,i)


#for i in range(3):
#sklms.fit(input[:T/2], target[:T/2])

#sklms.coeff_ = sklms.coeff_ / np.array(range(1000,0,-1))

#p_ = np.zeros(1000)
#for kay in range(1000):
#    for jay in range(kay+1,1001):
#        p_[kay] = p_[kay] + 1/jay  

#print p_
#p_ = np.cumsum(range(1000,0,-1))
sklms.coeff_ = sklms.coeff_ / p_
sklms.X_transformed_ = w[1]+w[0]*sklms.transform(input, input)
#sklms.X_transformed_ = sklms.transform(input[:T/2], input)
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
#pl.plot(sklms.centerIndex_, st[sklms.centerIndex_], '*r')
pl.show()

# Write results to a file
trainMSE = np.mean( (target[:2000] - sklms.X_transformed_[:2000])**2 )
testMSE = np.mean( (target[2000:] - sklms.X_transformed_[2000:])**2 )
trainCORR = np.corrcoef(target[:2000], sklms.X_transformed_[:2000])
testCORR = np.corrcoef(target[2000:], sklms.X_transformed_[2000:])
with open('results_test_linli.txt', 'a') as results:
    results.write('*Train MSE: %f | Test MSE: %f | train CORR: %f '
            '| test CORR: %f | kernel: %s | ksize: %f | dropout: %.1f | '
            'w_size %.1f, %.1f | l_rate: %f\n | growing: %s'
            % (trainMSE, testMSE, trainCORR[0,1], testCORR[0,1], \
            sklms.kernel, sklms.ksize, sklms.dropout, \
            time_window[0], time_window[1], sklms.learning_rate, sklms.growing_criterion))
