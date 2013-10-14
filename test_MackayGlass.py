# -*- coding: utf-8 -*-
"""
Created on Thu Sep 26 08:24:38 2013

@author: eder
"""
import numpy as np
from scipy.linalg import toeplitz
import tseries as t
from kernel_lms import KernelLMS
from sklearn.preprocessing import scale

data = np.loadtxt('MackeyGlass_t17.txt')
X = toeplitz(data , np.zeros(10))
#X = t.time_delay_input(data,10)
klms = KernelLMS(learning_rate=.01, gamma=1, growing_criterion="dense", \
                 growing_param=[1,.4], loss_function="least_squares", \
                 correntropy_sigma=.4, dropout=.5)

d = np.squeeze(np.hstack([np.zeros(10)[np.newaxis], data[np.newaxis]]))
d = d[0:5e3]
klms.fit_transform(X[0:5e3],d[0:5e3])

#plot(klms.X_transformed_[4e3:5e3])
#plot(d[4e3:5e3])

#figure
plot(scale(klms.X_transformed_[4e3:5e3]))
plot(scale(d[4e3:5e3]))
