# -*- coding: utf-8 -*-
"""
Preperes a time series to be input of time delay network.
Input time is considered to be among rows.

Created on Wed May 15 10:35:37 2013

@author: eder
"""
import numpy as np
import sys
def time_delay_input(X, N):
    n_rows = X.shape[0]
    zpad = np.zeros(n_rows%N)
    X = np.hstack([X, zpad])    
    Y = np.zeros([np.ceil(n_rows/N), N])
    for i in range(int(np.ceil(n_rows/N))):
        Y[i,:] = X[i*N:((i+1)*N)]
    return Y  

def spike_delay_input(S, t_block, T):
    i1 = (i for i in range(0, T, t_block))
    i2 = (i for i in range(t_block, T+t_block, t_block))
    Y = S[i1:i2]
    return Y

def nest_2_input(spike_events, T, time_window=[-20., 80.], time_step=2):
    """
    Prepares data from Nest simulation to be input of SpikeKLMS.

    Parameters
    ----------
    spike_events: 2 x total_n_spikes numpy array, where first rows contains spike times and the seconde rows the equivalent spike senders.

    T: int simulation time in miliseconds

    time_window: int list where time_window[1]-time_window[0] is the size of the time window analyzed. time_window[0] is a negative to consider lasting influeces of the previous stimulus.
    default: [-20, 80]

    time_step: time step in milisenconds to sliding the time window
    default: 2

    Output
    -----------
    population: list of list of spike times for each individual sender neuron.

    """

    spike_times   = spike_events[0]
    spike_senders = spike_events[1] - np.min(spike_events[1])
    n_neurons     = np.unique(spike_senders)
    
    population = [list() for i in range(0,T,time_step)] # empty list in the size of the simulation
    k=-1
    for tau in range(0,T,time_step):
       k += 1 
       population[k] = [np.array([]) for i in range(n_neurons.shape[0])] # empty list with as many neurons
       idx1 = spike_times>=(tau+time_window[0])
       idx2 = spike_times<=(tau+time_window[1])
       idx  = np.logical_and(idx1, idx2)
       win_times   = spike_times[idx] - max(tau+time_window[0], 0.) # scaled spike times
       win_senders = spike_senders[idx]
       
       for i in xrange(n_neurons.shape[0]): # for each neuron i, take the (scaled)times it sent a spike
           #population[k][i] = list()
           sender_idx       = (win_senders == n_neurons[i])
           population[k][i] = np.sort(win_times[sender_idx])

    return population
