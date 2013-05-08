"""
SLASH - Spike Linear Algebra Subroutines in Hilbert spaces
is a library implementing spike trains inner products in three levels:
    Level I   - single neuron vs single neuron inner products
    Level II  - population vs single neuron inner products
    Level III - population vs population inner products

At Level I function names start with lower case 's', 'p' is used to start Level II functions and finally, we used 'pp' to indicate Level III inner products.
    
    
SLASH was orignally conceived by Eder Santana as a BLAS equivalent for spike train linear algebra using Reproducing Kernel Hilbert Spaces.
    
Disclaimer: Part of this code was adapted from "spiketrainlib" by Il Memming Park. spiketrainlib follows BSD license.

References: Paiva et. al. Reproducing Kernel Hilbert Space Framework for spike train Signal Processing.
    
Author: Eder Santana <twitter/@edersantana>
License: BSD style
"""
import numpy as np

def check_n_spikes(x,y=None, sort_flag=False):
    """
    For the remaining of the code, it is more efficient to use y as the longer spike train.
    If only one spike train is passed as argument, this function defines y as x.
    Also, if the spike times weren't sorted yet, insure that sort is flagged to True
    """
    if y is None:
        y = x
    if sort_flag:
        x = np.sort(x)
        y = np.sort(y)
    
    if x.shape[0]>y.shape[0]:
        return y, x
    else:
        return x, y

def pairwise_l1(x,y):
    d = np.tile(y, [x.shape[0], 1]).T
    d -= x
    d = abs(d)
    return d.flatten()

def sMCI(x, y, ksize, y_neg_cum_sum_exp=None, y_pos_cum_sum_exp=None):
    """
    Memoryless Cross Intensity kernel (mCI)
        Input:
          x: (N1x1) numpy array of sorted spike times
          y: (N2x1) numpy array of sorted spike times
          ksize: kernel size
          y_neg_cum_sum_exp = sum_j exp(-y[-j]/ksize) 
          y_pos_cum_sum_exp = sum_j exp(y[j]/ksize)
        Output:
         v: sum_i sum_j exp(-|x[i] - y[j]|/ksize)
    """
    v = 0
    if x.shape[0]==0 and y.shape[0]==0:
        return v

    assert (ksize>0. and np.isreal(ksize)), "Kernel size must be non-negative real"        

    x, y = check_n_spikes(x,y) # we are assuming sorted spike trins

    if x.shape[0] <= 4 or y.shape[0] <= 4:
        # For small spike trains this can be afforded
        v = pairwise_l1(x, y)
        v = np.exp(-v/ksize)
        v = v.sum()
    else:
        # O( N1 log(N2)) implementation
        x /= ksize
        y /= ksize
        x_pos_exp = np.exp(x)
        x_neg_exp = np.exp(-x)
        
        if y_pos_cum_sum_exp is None:
            y_pos_cum_sum_exp = np.exp(y)
            y_pos_cum_sum_exp = np.cumsum(y_pos_cum_sum_exp)
        
        if y_neg_cum_sum_exp is None:
            y_neg_cum_sum_exp = np.exp(-y)
            y_neg_cum_sum_exp = np.flipud(y_neg_cum_sum_exp)
            y_neg_cum_sum_exp = np.cumsum(y_neg_cum_sum_exp)
            y_neg_cum_sum_exp = np.flipud(y_neg_cum_sum_exp)

        yidx = -1
        for xidx in xrange(x.shape[0]-1):
            while y[min(yidx+1,y.shape[0]-1)] <= x[xidx] and yidx<y.shape[0]-1:
                yidx += 1

            if yidx == -1:
                v = v + y_neg_cum_sum_exp[0] * x_pos_exp[xidx]
            elif yidx == (y.shape[0]-1):
                v = v + y_pos_cum_sum_exp[-1] * x_neg_exp[xidx]
            else:
                v = v + y_pos_cum_sum_exp[yidx] * x_neg_exp[xidx] + y_neg_cum_sum_exp[yidx+1] * x_pos_exp[xidx]

    return v

x = np.array([.001, .02, .3])
y = np.array([.02, .4])
v = sMCI(x, y, .01)