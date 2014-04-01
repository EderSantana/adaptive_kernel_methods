# -*- coding: utf-8 -*-
"""
Created on Thu Sep 26 08:24:38 2013

@author: eder
"""
import numpy as np
import pylab as pl
from scipy.linalg import toeplitz
import tseries as t
import kernel_lms; reload(kernel_lms)
from kernel_lms import KernelLMS
from sklearn.preprocessing import scale
from sklearn.linear_model import LinearRegression


data = np.loadtxt('MackeyGlass_t17.txt')
X = toeplitz(data , np.zeros(10))
#X = t.time_delay_input(data,10)
klms = KernelLMS(learning_rate=.001, gamma=1, growing_criterion="dense", \
                 growing_param=[1,.4], loss_function="least_squares", \
                 correntropy_sigma=.4, dropout=1)

d = np.squeeze(np.hstack([np.zeros(10)[np.newaxis], data[np.newaxis]]))
d = d[0:5e3]
klms.fit_transform(X[0:5e3],d[0:5e3])

_r = LinearRegression()
_R = np.hstack([klms.X_transformed_[np.newaxis].T, np.ones([5000,1])])
_r.fit(_R, d[np.newaxis].T)
_R2 = np.hstack([klms.X_transformed_[np.newaxis].T, np.ones([5000,1])])
reg = np.squeeze(_r.predict(_R2))


#plot(klms.X_transformed_[4e3:5e3])
#plot(d[4e3:5e3])

#figure
pl.close('all')
pl.plot((klms.X_transformed_[4e3:5e3]),label='klms')
pl.plot((d[4e3:5e3]))
pl.plot(reg[4e3:5e3],label='regression')
pl.legend( loc='upper left', numpoints = 1 )
pl.show()

# regularize with correlations

