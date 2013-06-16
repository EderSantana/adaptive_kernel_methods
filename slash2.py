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
import sklearn as sk
from sklearn.externals.joblib import Parallel
from sklearn.externals.joblib import delayed
from sklearn.externals.joblib.parallel import cpu_count
from sklearn.utils import gen_even_slices
import pyximport
pyximport.install(setup_args={'include_dirs':[np.get_include()]})
import fast_slash as fs

"""
====================
SLASH - Level I
====================
"""

# TODO: define a function to calculate x_stuffs and mci to reduce code repetition.

def check_list_population(Xx):
    if isinstance(Xx, np.ndarray): # Check if single spike train
        Xx = list([Xx])
    if isinstance(Xx[0], np.ndarray): #Basic exception treatment, I didn't though if it will always work
        Xx = list([Xx])
    return Xx

def check_population(X):
    if not(isinstance(X, list)):
        X = list([X])
    return X

def check_spike_train(x):
    if x.shape:
        return x #takes only the first value
    else:
        return (x[np.newaxis])

def check_n_spikes(x, y=None, sort_flag=False):
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

def sMCI(x, y, ksize):
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
    x = check_spike_train(x)
    y = check_spike_train(y)
    if x.shape[0]==0 or y.shape[0]==0:
        return v

    assert (ksize>0. and np.isreal(ksize)), "Kernel size must be non-negative real"

    #v = pairwise_l1(x, y)
    #v = np.exp(-v/ksize)
    #v = v.sum()
    v = fs.mci(x, y, ksize)
    
    return v

def sMCIdistance(x, y, ksize, mci11=None, mci22=None):
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
        return d

    assert (ksize>0. and np.isreal(ksize)), "Kernel size must be non-negative real"        

    # first, verify if inputs are equal
    _chk = (x==y)
    if isinstance(_chk, np.ndarray):
        if all(_chk):
            return d
    elif _chk:
        return d

    # The second special case is a empty vector (silent neuron)
    if x.shape[0] == 0:
        d = sMCI(y, y, ksize)
    elif y.shape[0] == 0:
        d = sMCI(x, x, ksize)
    else:
        mci12 = sMCI(x, y, ksize)
        if mci11 is None or mci11==[]:
            mci11 = sMCI(x, x, ksize)
        if mci22 is None or mci22 == []:
            mci22 = sMCI(y, y, ksize)

        d = mci11 + mci22 - 2*mci12

    return d

def sNCI(x, y, ksize, gamma, mci11=None, mci22=None):
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
    v = sMCIdistance(x, y, ksize, mci11=mci11, mci22=mci22)
    v *= -gamma
    v = np.exp(v)
    return v

""" 
==========================
SLASH - Level II
==========================
"""


def pMCI(X, y, ksize, MCI11=None, mci22=None):
    """
    Memoryless Cross Intensity kernel (mCI)
        Input:
          X: is a length M of numpy arrays of sorted spike times
          y: (N2x1) numpy array of sorted spike times
          ksize: kernel size
          y_neg_cum_sum_exp = sum_j exp(-y[-j]/ksize) 
          y_pos_cum_sum_exp = sum_j exp(y[j]/ksize)
        Output:
         V_k: (numpy.array) sum_i sum_j exp(-|x_k[i] - y[j]|/ksize)
        
    """
    X = check_population(X)
    y = check_spike_train(y)
    
    V = np.zeros(len(X));
    for i in xrange(len(X)):
        V[i] = sMCI(X[i], y, ksize)

    return V

def pMCIdistance(X, y, ksize, MCI11=None, mci22=None):
    """
    Memoryless Cross Intensity kernel (mCI) based squared distance 
    Input:
        X: is a length M of numpy arrays of sorted spike times
        y: (N2x1) numpy array of sorted spike times
        ksize: kernel size
        y_neg_cum_sum_exp = sum_j exp(-y[-j]/ksize)
        y_pos_cum_sum_exp = sum_j exp(y[j]/ksize)
    Output:
        V_k: (numpy.array) mci(X_i, X_i) + mci(y, y) - 2*mci(X_i, y)
        
    """
    X = check_population(X)
    y = check_spike_train(y)    
    
    
    if MCI11 is None:
        MCI11 = [list() for i in range(len(X))]
    if mci22 is None:
        mci22 = sMCI(y, y, ksize)
    V = np.zeros(len(X))
    for idx in xrange(len(X)):
        V[idx] = sMCIdistance(X[idx], y, ksize, mci11=MCI11[i], mci22=mci22)
    
    return V
    
def pNCI(X, y, ksize, gamma, MCI11=None, mci22=None):
    """
    Nonlinear Cross Intensity kernel (mCI) based distance
    Input:
        X: is a length M of numpy arrays of sorted spike times
        y: (N2x1) numpy array of sorted spike times
        ksize: kernel size
        y_neg_cum_sum_exp = sum_j exp(-y[-j]/ksize)
        y_pos_cum_sum_exp = sum_j exp(y[j]/ksize)
    Output:
        V_k: exp(-gamma*squared_MCI_distance(X_k, y))  
        
    """
    # The following steps are processed inside pMCI called by pMCIdistance
    #X = check_population(X)
    #y = check_spike_train(y)
    
    V = np.zeros(len(X))
    V = pMCIdistance(X, y, ksize, MCI11=MCI11, mci22=mci22)
    V *= -gamma
    V = np.exp(V)
    return V

"""
====================
SLASH - Level III
====================
"""
# At this level populations of spike trains should be input in lists

def ppMCI(Xx,Y, ksize):
            
    Xx = check_list_population(Xx)
    Y  = check_population(Y)
    V = np.zeros(len(Xx))
    for i in xrange( len(Xx) ):
        for k in xrange( len(Xx[i]) ):
            Xx[i][k] = check_spike_train(Xx[i][k])
    for i in xrange( len(Y) ):
        Y[i] = check_spike_train(Y[i])
    for i in xrange(len(Xx)):
        for k in xrange(len(Y)):
            V[i] = V[i] + sMCI(Xx[i][k], Y[k], ksize)
    #V = fs.ppmci(np.ndarray(Xx), np.ndarray(Y), int(len(Xx)), int(len(Y)), ksize)
    
    return V

def ppMCIdistance(Xx,Y, ksize, pMCI11=None, MCI22=None):
    
    Xx = check_list_population(Xx)
    Y  = check_population(Y)

    if pMCI11 is None or pMCI11==[]:
        pMCI11 = [list() for i in range(len(Xx))]
        for i in xrange(len(Xx)):
            pMCI11[i] = [list() for k in range(len(Xx[i]))]
    if MCI22 is None or MCI22==[]:
        MCI22 = [list() for i in range(len(Y))]
        for k in xrange(len(Y)):
            MCI22[k] = sMCI(Y[k], Y[k], ksize)

    V = np.zeros(len(Xx))
    for i in xrange(len(Xx)):
        for k in xrange(len(Y)):
            V[i] = V[i] + sMCIdistance(Xx[i][k], Y[k], ksize, mci11=pMCI11[i][k], \
                                       mci22=MCI22[k])
    
    return V

def ppNCI(Xx, Y, ksize, gamma, pMCI11=None, MCI22=None):
    V = ppMCIdistance(Xx, Y, ksize, pMCI11=pMCI11, MCI22=MCI22)
    V = np.exp(-gamma*V)
    return V

"""
====================
SLASH - EIG I
====================
"""

def eigMCI(X, Y, ksize=.001, eig_idx=0, n_jobs=1, center_option="individual"):
    """
    First eigMCI calculates the kernel matrices Kx and Ky of populations of spiking neurons X and Y, induced the 'mci' spike kernel.
    Then, eigMCI calculates the inner product between Xx and Yy as the inner product between the leading eigenvectors of Kx and Ky.
        
    Parameters:
    -----------
    
    center_option: "individual" | "common" | "none"
        default: "individual"
        The centering applied to each kernel matrix Kx and Ky. "individual" consider the bias at each space. "common" consider the bias induced by Xx and subtracts it from Kx and Ky. "none" no centering is applied over Kx and Ky.
    
    eig_idx: int
        default: 1
        Index of the eigeventor used as representer of the whole population. The leading eigenvector is used as default. If the more than one eigenvetor is listed, the result will be trace of the inner product matrix between eigenvectors.

    """
    params = {"ksize": ksize}
    Kx = compute_spike_kernel_matriz(X, X, spike_kernel="mci", n_jobs=n_jobs, **params)
    Ky = compute_spike_kernel_matriz(Y, Y, spike_kernel="mci", n_jobs=n_jobs, **params)
    if center_option == "individual":
        kcenterer_x = sk.preprocessing.KernelCenterer()
        kcenterer_y = sk.preprocessing.KernelCenterer()
        kcenterer_x.fit(Kx)
        kcenterer_y.fit(Ky)
        Kxc = kcenterer_x.transform(Kx)
        Kyc = kcenterer_y.transform(Ky)
    
    elif center_option == "common":
        kcenterer_x = sk.preprocessing.KernelCenterer()
        kcenterer_x.fit(Kx)
        Kxc = kcenterer_x.transform(Kx)
        Kyc = kcenterer_x.transform(Ky)

    elif center_option == "none":
        Kxc = Kx
        Kyc = Ky

    Dx, Ex = np.linalg.eig(Kxc)
    Dy, Ey = np.linalg.eig(Kyc)

    v = np.dot( Ex[:,eig_idx].T, Ey[:,eig_idx] )
    if v.shape != ():
        v = np.trace(v)

    return v

"""
====================
SLASH - EIG II
====================
"""

def _eigdecompose_population(Xx, eig_idx=[0], spike_kernel='mci', n_jobs=1, **params):
    Xx = check_list_population(Xx)
    KXxc = [list() for i in xrange( len(Xx) )]
    dim        = len(Xx[0])
    n_eigs     = len(eig_idx)
    n_vecotors = len(Xx) * n_eigs
    EXx = np.zeros((dim, n_vecotors))
    for i in xrange( len(Xx) ):
        KXx = compute_spike_kernel_matriz(Xx[i], Xx[i], spike_kernel=spike_kernel, \
                                          n_jobs=n_jobs, **params)
        kcenterer_x = sk.preprocessing.KernelCenterer()
        kcenterer_x.fit(KXx)
        KXxc[i] = kcenterer_x.transform(KXx)
        D, E = np.linalg.eig(KXxc[i])
        EXx[:,(i*n_eigs):((i+1)*n_eigs)] = E[:, eig_idx]

    return EXx

def peigMCI(Xx, Yy=None, ksize=.001, eig_idx=[1], n_jobs=1, EXx=None):
    """
    This is a more efficient method to calculate the spike kernel matriz induced by eigMCI. In the present method, the matrices and eigenvectors are calculated before hand. This avoids the repeated eigendecompositions.
        
    """
    if Yy is None:
        Yy = Xx
    Xx = check_list_population(Xx)
    Yy = check_list_population(Xx)

    # TODO: accept eigendecomposition of Xx as input
    params = {"ksize": ksize}
    if isinstance(eig_idx, int):
        eig_idx = [eig_idx]

    # Calculate X matrix
    if EXx is None or EXx==[]:
        EXx = _eigdecompose_population(Xx, eig_idx=eig_idx, spike_kernel="mci", \
                                          n_jobs=n_jobs, **params)
    else:
        EXx = np.squeeze(EXx)
    # Calculate Y matrix
    if Yy is Xx:
        EYy = EXx
        #KYyc = KXxc
    else:
        EYy = _eigdecompose_population(Yy, eig_idx, spike_kernel="mci", \
                                              n_jobs=n_jobs, **params)
    # Inner product
    #import pdb; pdb.set_trace()
    V = np.dot(EXx.T, EYy)
    if V.shape:
        V = np.diag(V)
    return V

def peigMCIdistance(Xx, Yy=None, ksize=.001, eig_idx=[1], n_jobs=1, EXx=None):
    """
        This is a more efficient method to calculate the spike kernel matriz induced by eigNCI. In the present method, the matrices and eigenvectors are calculated before hand. This avoids the repeated eigendecompositions.
        
        """
    params = {"ksize": ksize, "gamma": gamma}
    if isinstance(eig_idx, int):
        eig_idx = [eig_idx]
    if Yy is None:
        Yy = Xx
    Xx = check_list_population(Xx)
    Yy = check_list_population(Yy)
    # Calculate X matrix
    if EXx is None or EXx==[]:
        EXx = _eigdecompose_population(Xx, eig_idx, spike_kernel="mcidistance", \
                                          n_jobs=n_jobs,**params)
    # Calculate Y matrix
    if Yy is Xx:
        EYy = EXx
        #KYyc = KXxc
    else:
        EYy = _eigdecompose_population(Yy,eig_idx,spike_kernel="mcidistance",\
                                              n_jobs=n_jobs, **params)
    # Inner product
    V = np.dot(EXx.T, EYy)
    if V.shape:
        V = np.diag(V)
    return V

def peigNCI(Xx, Yy=None, ksize=.001, gamma=1, eig_idx=[0], n_jobs=1, EXx=None):
    """
        This is a more efficient method to calculate the spike kernel matriz induced by eigNCI. In the present method, the matrices and eigenvectors are calculated before hand. This avoids the repeated eigendecompositions.
        
        """
    params = {"ksize": ksize, "gamma": gamma}
    if isinstance(eig_idx, int):
        eig_idx = [eig_idx]
    if Yy is None:
        Yy = Xx
    if isinstance( Xx[0], np.ndarray):
        Xx = [Xx]
    if isinstance( Yy[0], np.ndarray):
        Yy = [Yy]
    
    # Calculate X matrix
    if EXx is None or EXx==[]:
        EXx = _eigdecompose_population(Xx, eig_idx, spike_kernel="nci",\
                                          n_jobs=n_jobs, **params)
    # Calculate Y matrix
    if Yy is Xx:
        EYy = EXx
        #KYyc = KXxc
    else:
        EYy = _eigdecompose_population(Yy, eig_idx, spike_kernel="nci",\
                                              n_jobs=n_jobs, **params)
    # Inner product
    V = np.dot(EXx.T, EYy)
    if V.shape:
        V = np.diag(V)
    return V

"""
==========================
SLASH - Inner product call
==========================
"""

# Helper functions - distance
SPIKE_KERNEL_FUNCTIONS = {
    # If updating this dictionary, update the doc in both distance_metrics()
    # and also in pairwise_distances()!
    'mci': pMCI,
    'nci': pNCI,
    'mcidistance': pMCIdistance,
    'pop_mci': ppMCI,
    'pop_nci': ppNCI,
    'pop_mcidistance': ppMCIdistance,
    'eig_mci': peigMCI,
    'eig_nci': peigNCI,
    'eig_mcidistance': peigMCIdistance,
    }

def spike_kernels():
    """ Valid kernels for inner_prod

    This function simply returns the valid spike kernsl.
    It exists, however, to allow for a verbose description of the mapping for
    each of the valid strings.

    The valid distance metrics, and the function they map to, are:
      ===============   ========================================
      metric            Function
      ===============   ========================================
      'mci'             slash.pMCI
      'nci'             slash.pNCI
      'mcidistance'     slash.pMCIdistance
      'pop_mci'         slash.ppMCI
      'pop_nci'         slash.ppNCI
      'pop_mcidistance' slash.ppMCIdistance
      'eig_mci'         slash.peigMCI
      'eig_nci'         slash.peigNCI
      'eig_mcidistance' slash.peigMCIdistance
      ===============   ========================================
        
    """
    return SPIKE_KERNEL_FUNCTIONS
    
KERNEL_PARAMS = {
    "mci": frozenset(["ksize"]),
    "nci": frozenset(["ksize", "gamma"]),
    "mcidistance": frozenset(["ksize"]),
    "pop_mci": frozenset(["ksize"]),
    "pop_nci": frozenset(["ksize", "gamma"]),
    "pop_mcidistance": frozenset(["ksize"]),
    "eig_mci": frozenset(["ksize", "EXx"]),
    "eig_mcidistance": frozenset(["ksize", "EXx"]),
    "eig_nci": frozenset(["ksize", "gamma", "EXx"]),
}

def inner_prod(X, Y=None, spike_kernel="mci", filter_params=False, n_jobs=1, \
               **kwds):
    """Compute the spike kernel between spike trains X and optional array Y.

    This method takes either a spike trains or a kernel matrix, and returns a 
    kernel matrix. If the input is a vector array, the kernels are computed. 
    If the input is a kernel matrix, it is returned instead.

    This method provides a safe way to take a kernel matrix as input, while
    preserving compatibility with many other algorithms that take a vector
    array.

    If Y is given (default is None), then the returned matrix is the pairwise
    kernel between the arrays from both X and Y.

    Valid values for spike_kernel are::
        ['mci', 'nci']

    Parameters
    ----------
    X : array [n_samples_a, n_samples_a] if spike_kernel == "precomputed", or,
              [n_samples_a, n_features] otherwise
        Array of pairwise kernels between samples, or a feature array.

    Y : array [n_samples_b, n_features]
        A second feature array only if X has shape [n_samples_a, n_features].

    spike_kernel : string, or callable
        The spike kernel to use when calculating inner product between instances 
        in a spike train. If spike_kernel is a string, it must be one of the ones
        in slash.SPIKE_KERNEL_FUNCTIONS.
        If spike_kernel is "precomputed", X is assumed to be a kernel matrix.
        Alternatively, if metric is a callable function, it is called on each
        pair of instances (rows) and the resulting value recorded. The callable
        should take two arrays from X as input and return a value indicating
        the distance between them.

    n_jobs : int
        The number of jobs to use for the computation. This works by breaking
        down the pairwise matrix into n_jobs even slices and computing them in
        parallel.

        If -1 all CPUs are used. If 1 is given, no parallel computing code is
        used at all, which is useful for debuging. For n_jobs below -1,
        (n_cpus + 1 + n_jobs) are used. Thus for n_jobs = -2, all CPUs but one
        are used.

    filter_params: boolean
        Whether to filter invalid parameters or not.

    `**kwds` : optional keyword parameters
        Any further parameters are passed directly to the kernel function. For
        example, the kernel size ksize or gamma value for nCI

    Returns
    -------
    K : array [n_samples_a, n_samples_a] or [n_samples_a, n_samples_b]
        A kernel matrix K such that K_{i, j} is the kernel between the
        ith and jth spike trains of the given list X, if Y is None.
        If Y is not None, then K_{i, j} is the kernel between the ith spike 
        train X and the jth spike train from Y.

    Notes
    -----
    If metric is 'precomputed', Y is ignored and X is returned.

    """
    if spike_kernel == "precomputed":
        return X
    elif spike_kernel in SPIKE_KERNEL_FUNCTIONS:
        if filter_params:
            kwds = dict((k, kwds[k]) for k in kwds
                        if k in KERNEL_PARAMS[spike_kernel])
        func = SPIKE_KERNEL_FUNCTIONS[spike_kernel]
        if n_jobs == 1:
            return func(X, Y, **kwds)
        else:
            return _parallel_inner_prod(X, Y, func, n_jobs, **kwds)
    elif callable(spike_kernel):
        # Check matrices first (this is usually done by the metric).
        #X, Y = check_pairwise_arrays(X, Y)
        n_x, n_y = len(X), len(Y)
        # Calculate kernel for each element in X and Y.
        K = np.zeros((n_x, n_y), dtype='float')
        for i in range(n_x):
            start = 0
            if X is Y:
                start = i
            for j in range(start, n_y):
                # Kernel assumed to be symmetric.
                K[i][j] = spike_kernel(X[i], Y[j], **kwds)
                if X is Y:
                    K[j][i] = K[i][j]
        return K
    else:
        raise ValueError("Unknown kernel %r" % spike_kernel)
    return

    
def _parallel_inner_prod(X, Y, func, n_jobs, **kwds):
    """Break the pairwise matrix in n_jobs even slices
    and compute them in parallel"""
    if n_jobs < 0:
        n_jobs = max(cpu_count() + 1 + n_jobs, 1)

    if Y is None:
        Y = X

    ret = Parallel(n_jobs=n_jobs, verbose=0)(
        delayed(func)(X[s], Y, **kwds)
        for s in gen_even_slices(len(X), n_jobs))

    return np.hstack(ret)

def compute_spike_kernel_matriz(X, Y, spike_kernel="mci", filter_params=True, \
                                n_jobs=1, **kwds):
    K = np.zeros([len(X), len(Y)])
    for i in xrange(len(Y)):
        V = inner_prod(X, Y[i], spike_kernel, filter_params, n_jobs, **kwds)
        K[:,i] = V
    return K
    
def _mci(x, x_neg_exp, x_pos_exp, x_neg_cum_sum_exp, x_pos_cum_sum_exp):
        mci = 0.
        for k in xrange(x.shape[0]):
            if k == 1:
                mci = mci + x_pos_exp[0] * \
                x_neg_cum_sum_exp[0]
            elif k == (x).shape[0]-1:
                mci = mci + x_neg_exp[-1] * x_pos_cum_sum_exp[-1]
            else:
                mci = mci + x_pos_cum_sum_exp[k] * \
                x_neg_exp[k] + x_pos_exp[k] * x_neg_cum_sum_exp[k+1]
    
        return mci

"""
====================
TESTS  
====================
"""
def test_sMCI():
    #x = np.array([0.0010, 0.0020, 0.0200,  0.0300, 0.3000,])
    #y = np.array([0.0040, 0.0050, 0.0200, 0.4000, 0.5000, 0.6000])
    x = np.array([ 0.05684891,  0.12183341,  0.21504637,  0.03783751,  0.09107526, 0.07008217,  0.05355698,  0.20495672])
    x.sort()
    y = np.array([0.0502541 ,  0.1180693 ,  0.10801783,  0.21467571,  0.04431462, 0.08787606,  0.06541311,  0.17787088])
    y.sort()
    v = sMCI(x, y, .001)
    print "Test_sMCI    v = %10.5e" % v
    print "Right answer v = %e" % 0.803689147953212
    return

def test_sMCIdistance():
    x = np.array([ 0.05684891,  0.12183341,  0.21504637,  0.03783751,  0.09107526, 0.07008217,  0.05355698,  0.20495672])
    x.sort()
    y = np.array([0.0502541 ,  0.1180693 ,  0.10801783,  0.21467571,  0.04431462, 0.08787606,  0.06541311,  0.17787088])
    y.sort()
    
    x = np.array([ 0.01505,  0.02274,  0.03102,  0.03963,  0.04812])
    y = np.array([ 0.00756,  0.01715,  0.02756,  0.03913])
    v = sMCIdistance(x, y, .01)
    print "Test_sMCIdistance v = %10.5e" % v
    return

def test_sNCI():
    x = np.array([0.0010, 0.0020, 0.0200,  0.0300, 0.3000,])
    y = np.array([0.0040, 0.0050, 0.0200, 0.4000, 0.5000, 0.6000])
    v = sNCI(x, y, .01, .01)
    print "Test_sNCI v = %10.5e" % v
    return

def test_pMCI():
    x = np.array([0.0010, 0.0020, 0.0200,  0.0300, 0.3000,])
    X = [np.array([]), x]
    y = np.array([0.0040, 0.0050, 0.0200, 0.4000, 0.5000, 0.6000])
    V = pMCI(X, y, .01)
    print "Test_pMCI V = [%10.5e, %10.5e]" % (V[0], V[1])
    return

def test_pMCIdistance():
    x = np.array([0.0010, 0.0020, 0.0200,  0.0300, 0.3000,])
    X = [np.array([]), x]
    y = np.array([0.0040, 0.0050, 0.0200, 0.4000, 0.5000, 0.6000])
    V = pMCIdistance(X, y, .01)
    print "Test_pMCIdistance V = [%10.5e, %10.5e]" % (V[0], V[1])
    return
    
def test_pNCI():
    x = np.array([0.0010, 0.0020, 0.0200,  0.0300, 0.3000,])
    X = [np.array([]), x]
    y = np.array([0.0040, 0.0050, 0.0200, 0.4000, 0.5000, 0.6000])
    V = pNCI(X, y, .01, .01)
    print "Test_pNCI V = [%10.5e, %10.5e]" % (V[0], V[1])
    return

def test_InnerProd():
    x = np.array([0.0010, 0.0020, 0.0200,  0.0300, 0.3000,])
    X = [np.array([]), x]
    y = np.array([0.0040, 0.0050, 0.0200, 0.4000, 0.5000, 0.6000])

    params = {"gamma": .01,
              "ksize": .01}
    
    V = inner_prod(X, y, spike_kernel="nci", n_jobs=1, filter_params=True, \
                  **params)
    print "Test_InnerProd V = [%10.5e, %10.5e]" % (V[0], V[1])
    return
