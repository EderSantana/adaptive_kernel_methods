# -*- coding: utf-8 -*-
"""
Created on Mon May 13 12:14:44 2013

@author: eder
"""
import numpy as np
from numpy import squeeze, newaxis, linspace
import pylab as pl
from sklearn.preprocessing import scale

import spike_klms2

pl.close('all')
reload(spike_klms2)
from spike_klms2 import SpikeKLMS
from scipy.io import loadmat
sts = loadmat("sts.mat").get("sts")
X = range(len(sts))
for i in xrange(len(X)):
    X[i] = squeeze(sts[i,0])
    if not(X[i].shape):
        X[i] = X[i][newaxis]

d   = squeeze(loadmat("targets.mat").get("targets"))
t   = linspace(1,500,500)
sklms = SpikeKLMS(kernel="mci", gamma=.05, learning_rate=.0003, ksize=.005, growing_criterion="dense",\
growing_param = [15, .01], dropout=5)
sklms.fit_transform(X, d)
pl.plot(sklms.X_transformed_)
pl.plot(d)

pl.figure()
pl.plot(sklms.coeff_)
pl.show()

fe = np.mean((d - sklms.X_transformed_)**2)/np.var(d)
print "Final error: %f" % fe
