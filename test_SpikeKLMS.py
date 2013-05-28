# -*- coding: utf-8 -*-
"""
Created on Mon May 13 12:14:44 2013

@author: eder
"""
from numpy import squeeze, newaxis, linspace

import spike_klms2

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
sklms = SpikeKLMS(kernel="mci", gamma=.05, learning_rate=.01, ksize=.005, growing_criterion="novelty",\
growing_param = [.1, .01])
sklms.fit_transform(X, d)
