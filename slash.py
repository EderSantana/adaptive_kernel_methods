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

"""
SLASH - Level I
"""

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
    v = 0.
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
        x_pos_exp = np.exp(x / ksize)
        x_neg_exp = np.exp(-x / ksize)
        
        if y_pos_cum_sum_exp is None:
            y_pos_cum_sum_exp = np.exp(y / ksize)
            y_pos_cum_sum_exp = np.cumsum(y_pos_cum_sum_exp)
        
        if y_neg_cum_sum_exp is None:
            y_neg_cum_sum_exp = np.exp(-y / ksize)
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
                v = v + y_pos_cum_sum_exp[yidx] * x_neg_exp[xidx] + \
                    y_neg_cum_sum_exp[yidx+1] * x_pos_exp[xidx]

    return v

def sMCIdistance(x, y, ksize, x_neg_exp=None, x_pos_exp=None, \
                 x_neg_cum_sum_exp=None, x_pos_cum_sum_exp=None, \
                 y_neg_exp=None, y_pos_exp=None, \
                 y_neg_cum_sum_exp=None, y_pos_cum_sum_exp=None):
    """
    Squared Memoryless cross intensity (mCI) based distance
    Input:
      st1: (N1x1) sorted spike times
      st2: (N2x1) sorted spike times
      ksize: kernel size
    Output:
      d: mci11 + mci22 - 2*mci12
    """    
    
    d = 0.    
    if x.shape[0]==0 and y.shape[0]==0:
        return

    assert (ksize>0. and np.isreal(ksize)), "Kernel size must be non-negative real"        

    x, y = check_n_spikes(x,y) # we are assuming sorted spike trins
    
    # first, verify if inputs are equal    
    if x.shape[0] is y.shape[0]:
        return d
     
    # If at least one of the inpusts is small, those options are affodable and faster
    if x.shape[0] <= 4 and y.shape[0] <= 4:
        mci12 = pairwise_l1(x, y) / ksize
        mci11 = pairwise_l1(x, x) / ksize
        mci22 = pairwise_l1(y, y) / ksize
        
        mci12 = np.sum( np.exp(-mci12) )
        mci11 = np.sum( np.exp(-mci11) )
        mci22 = np.sum( np.exp(-mci22) )
        
    elif x.shape[0] <= 4:
        mci12 = pairwise_l1(x, y) / ksize
        mci11 = pairwise_l1(x, x) / ksize
        mci12 = np.sum( np.exp(-mci12) )
        mci11 = np.sum( np.exp(-mci11) )
        
        mci22 = sMCI(y, y, ksize)
        
    elif y.shape[0] <= 4:
        mci12 = pairwise_l1(x, y) / ksize
        mci22 = pairwise_l1(y, y) / ksize
        mci12 = np.sum( np.exp(-mci12) )
        mci11 = np.sum( np.exp(-mci22) )
        
        mci11 = sMCI(x, x, ksize)
        
    else: # if spike trains are larger enough, we better use the O(N1 log(N2) ) approx.
        
        # I reccomend analyzing mCI formula when exp(-abs(t1-t2)) 
        #    for t1>=t2 and t1<=t2 to understand this
        
        if  y_pos_cum_sum_exp is None:            
            y_pos_exp = np.exp(y / ksize)                    
            y_pos_cum_sum_exp = np.cumsum(y_pos_exp)
        
        if y_neg_cum_sum_exp is None:
            y_neg_exp = np.exp(-y / ksize)
            y_neg_cum_sum_exp = np.flipud(y_neg_exp)
            y_neg_cum_sum_exp = np.cumsum(y_neg_cum_sum_exp)
            y_neg_cum_sum_exp = np.flipud(y_neg_cum_sum_exp)        
            
        if  x_pos_cum_sum_exp is None:
            x_pos_exp = np.exp(x / ksize)
            x_pos_cum_sum_exp = np.cumsum(x_pos_exp)
        
        if x_neg_cum_sum_exp is None:
            x_neg_exp = np.exp(-x / ksize)
            x_neg_cum_sum_exp = np.flipud(x_neg_exp)
            x_neg_cum_sum_exp = np.cumsum(x_neg_cum_sum_exp)
            x_neg_cum_sum_exp = np.flipud(x_neg_cum_sum_exp)            
            
        # Calculate mci's
            
        # mci(x, y) and mci(x, x)
        yidx = -1
        mci12 = 0.
        mci11 = 0.
        for xidx in xrange(x.shape[0]-1):
            while y[min(yidx+1,y.shape[0]-1)] <= x[xidx] and yidx<y.shape[0]-1:
                yidx += 1

            if yidx == -1:
                mci12 = mci12 + y_neg_cum_sum_exp[0] * x_pos_exp[xidx]
            elif yidx == (y.shape[0]-1):
                mci12 = mci12 + y_pos_cum_sum_exp[-1] * x_neg_exp[xidx]
            else:
                mci12 = mci12 + y_pos_cum_sum_exp[yidx] * x_neg_exp[xidx] + \
                    y_neg_cum_sum_exp[yidx+1] * x_pos_exp[xidx]            
            # mci(x , x)
            if xidx == 1:
                mci11 = mci11 + x_pos_exp[0] * x_neg_cum_sum_exp[0]
            elif xidx == x.shape[0]-1:
                mci11 = mci11 + x_neg_exp[-1] * x_pos_cum_sum_exp[-1]
            else:
                mci11 = mci11 + x_pos_cum_sum_exp[xidx] * x_neg_exp[xidx] + \
                        x_pos_exp[xidx] * x_neg_cum_sum_exp[xidx+1]
        
        # mci(st2, st2)
        mci22 = 0.;
        for yidx in xrange(y.shape[0]-1):
            if yidx == 1:
                mci22 = mci22 + y_pos_exp[0] * y_neg_cum_sum_exp[0]
            elif yidx == y.shape[0]-1:
                mci22 = mci22 + y_neg_exp[-1] * y_pos_cum_sum_exp[-1]
            else:
                mci22 = mci22 + y_pos_cum_sum_exp[yidx] * y_neg_exp[yidx] + \
                        y_pos_exp[yidx] * y_neg_cum_sum_exp[yidx+1]
        
        print "mci11 = %f, mci22 = %f, mci12 = %f" % (mci11, mci22, mci12)
    d = mci11 + mci22 - 2*mci12
    return d

def sNCI(x, y, ksize, gamma, x_neg_exp=None, x_pos_exp=None, \
                 x_neg_cum_sum_exp=None, x_pos_cum_sum_exp=None, \
                 y_neg_exp=None, y_pos_exp=None, \
                 y_neg_cum_sum_exp=None, y_pos_cum_sum_exp=None):
    """
    nonlinear cross intensity kernel (nCI) - (aka. radial basis function for)
                                             (spike trains)
    Input:
      st1: (N1x1) sorted spike times
      st2: (N2x1) sorted spike times
      ksize: kernel size 
    Output: 
      v: exp(-gamma*squared_MCI_distance)  
    """
    v = sMCIdistance(x, y, ksize, x_neg_exp, x_pos_exp, \
                     x_neg_cum_sum_exp, x_pos_cum_sum_exp, \
                     y_neg_exp, y_pos_exp, \
                     y_neg_cum_sum_exp, y_pos_cum_sum_exp)
    v *= -gamma
    v = np.exp(v)
    return v

def test_sMCI():
    x = np.array([0.0010, 0.0020, 0.0200,  0.0300, 0.3000,])
    y = np.array([0.0040, 0.0050, 0.0200, 0.4000, 0.5000, 0.6000])
    v = sMCI(x, y, .01)
    print "Test_sMCI v = %10.5e" % v
    return

def test_sMCIdistace():
    x = np.array([0.0010, 0.0020, 0.0200,  0.0300, 0.3000,])
    y = np.array([0.0040, 0.0050, 0.0200, 0.4000, 0.5000, 0.6000])
    v = sMCIdistance(x, y, .01)
    print "Test_sMCIdistance v = %10.5e" % v
    return
    
def test_sNCI():
    x = np.array([0.0010, 0.0020, 0.0200,  0.0300, 0.3000,])
    y = np.array([0.0040, 0.0050, 0.0200, 0.4000, 0.5000, 0.6000])
    v = sNCI(x, y, .01, .01)
    print "Test_sNCI v = %10.5e" % v
    return