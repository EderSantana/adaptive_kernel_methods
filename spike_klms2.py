"""Spike Kernel Least Mean Square Algorithm"""

# Author: Eder Santana <edersantanajunior@hotmail.com>
# License: BSD Style.

import numpy as np
import slash2 as slash

import numpy.random
from sklearn.base import BaseEstimator, TransformerMixin
#from sklearn.metrics.pairwise import pairwise_kernels, euclidean_distances


class SpikeKLMS(BaseEstimator, TransformerMixin):
    """Spike Kernel Least Mean Square Algorithm (KLMS)

    Non-linear filtering in spike times feature space by linear filtering in 
    Hilbert spaces

    Parameters
    ----------

    learning_rate: float
        Step size for gradient descent adaptation. This parameter is very 
        important since regularizes the kernel method and, for a given data set, 
        define convergence time and misadjustment

    growing_criterion: "dense" | "novelty" | "quantized" |"surprise"
        Default: "dense:"

    growing_param: float, float, optional

    kernel: "mci" | "nci" 
        Kernel.
        Default: "mci"

    ksize : int, default=.01
        Kernel size for mCI

    gamma : float, optional
        Kernel coefficient for nCI (ak. rbf of spike trains)

    TODO: add suuport to users custom spike kernels    
    kernel_params : mapping of string to any, optional
        Parameters (keyword arguments) and values for kernel passed as
        callable object. Ignored by other kernels.

    Attributes
    ----------
    
    `X_online_`:
        Projection of spike train data while filter is trained

    `X_transformed_`:
        Projection of the spike train data on the trained filter
   
    `coeff_`:
        Filter coefficients

     `centers_`:
        Centers of growing network

     `centerIndex_`:
         Indexes of the input data kept as centers kept by the network
         
     `XX_`:
         Transformations on centers_, this is stored to avoid extra calculations

    References
    ----------
    Kernel LMS was intoduced in:
        The Kernel LMS algorithm by Weifeng Liu et. al.
    """

    def __init__(self, kernel="mci", learning_rate=0.01, growing_criterion="dense", \
                 growing_param=None, loss_function="least_squares", \
                 loss_param=None, gamma=None, ksize=0.01, kernel_params=None, \
                 correntropy_sigma=None, n_jobs=1, dropout=0):
        
        self.n_jobs = n_jobs
        self.kernel = kernel
        self.kernel_params = kernel_params
        self.learning_rate = learning_rate
        self.loss_function = loss_function
        if self.loss_function != "least_squares":
            self.loss_param = loss_param
            if self.loss_function == "minimum_correntropy":
                self.correntropy_sigma = correntropy_sigma
        self.growing_criterion = growing_criterion
        if self.growing_criterion != "dense":
            self.mci11 = list()        
            self.growing_param = growing_param
        self.gamma = gamma
        self.ksize = ksize
        self.eig11 = list()
        self.centers_ = np.array([])
        self.coeff_ = np.array([])
        self.centerIndex_ = []
        self.X_online_ = np.array([])
        self.X_transformed_ = np.array([])
        self.dropout = dropout
        
    """
    TODO: add support for precomputed gram matrix to make fit_transform faster  
    @property
    def _pairwise(self):
        return self.kernel == "precomputed"
    """
    def _get_kernel(self, X, Y=None):
        if callable(self.kernel):
            params = self.kernel_params or {}
        else:
            params = {"gamma": self.gamma,
                      "ksize": self.ksize,
                      "EXx"  : self.eig11}
        return slash.inner_prod(X, Y, spike_kernel=self.kernel,
                                filter_params=True, n_jobs=self.n_jobs, **params)

    def fit(self, X, d, err=None):
        """Fit the model from data in X.
            
            Parameters
            ----------
            X: array-like, shape (n_samples, n_features)
            Training vector, where n_samples in the number of samples and n_features is the number of features.
            d: array-like, shape (n_samples)
            Desired or teaching vector
            
            Returns
            -------
            self : object
            Returns the instance itself.
   
            """
 
        Nend = len(X)
        N1 = 0
        # If initializing network
        if self.coeff_.shape[0] == 0:
            self.centers_     = list([X[0]])
            if self.growing_criterion != "dense":
                self._appendMCI(X[0])
            if self.kernel=='eig_mci' or self.kernel=='eig_nci':
                self._appendEIG(X[0])
                #self.eig11.append(slash._eigdecompose_population(X[0],\
                #        spike_kernel=self.kernel[4:]))
            self.centerIndex_ = list()
            self.coeff_       = np.array([])
            new_coeff         = self.learning_rate * self._loss_derivative(d[0])
            self.coeff_       = np.append( self.coeff_, new_coeff )
            self.X_online_    = np.zeros(Nend)
            N1                = 1
        
        # For initialized networks
        for k in xrange(N1,Nend):
            print k
            dropin_centers, dropin_coeff = self._dropout()
            gram              = self._get_kernel(dropin_centers,X[k])
            self.X_online_[k] = np.dot(dropin_coeff, gram)
            self._trainNet(X[k], d[k]-self.X_online_[k],k)
        
        return self

    def transform(self, Z):
        """Project data Z into the fitted filter

        Parameters
        ----------
        Z: array-like, shape (n_samples, n_features)

        Returns
        -------
        Z_out: array-like, shape (n_samples)
        """
        Z_out = [{} for i in range(len(Z))]
        
        for i in xrange(len(Z)):
            print i
            Z_out[i] = np.dot(self.coeff_, self._get_kernel(self.centers_,Z[i]))
    
        return Z_out

    def fit_transform(self, X, d, err=None):
        """Fit the model from data in X and transform X.

        Parameters
        ----------
        X: array-like, shape (n_samples, n_features)
            Training vector, where n_samples in the number of samples
            and n_features is the number of features.

        Returns
        -------
        X_transformed_: array-like, shape (n_samples)
       
        """
        self.fit(X, d, err)

        self.X_transformed_ = self.transform(X)

        return self.X_transformed_
 
    def _trainNet(self, newX, err, k):
        """ Append centers to the network following growing_criterion
            
        Returns
        -------
            `self` with possibly larger centers_, coeff_ and centerIndex_
        """
        if self.coeff_.shape[0] == 0:
            self.centers_ = list([newX])
            self.coeff_ = np.append(self.coeff_, self.learning_rate *
                                    self._loss_derivative(err))
            self.centerIndex_ = k
        else:
            #===========================
            #     DENSE GROW
            #===========================
            if self.growing_criterion == "dense":
                self.centers_.append(newX)                
                self.coeff_ = np.append(self.coeff_, self.learning_rate *
                                        self._loss_derivative(err))
                self.centerIndex_ = [self.centerIndex_, k]
                if self.kernel=='eig_mci' or self.kernel=='eig_nci':
                    self._appendEIG(newX)
            #===========================
            #     NOVELTY GROW
            #===========================
            elif self.growing_criterion == "novelty":
                distanc = slash.ppMCIdistance(self.centers_, newX, \
                                             ksize=self.ksize, pMCI11=self.mci11)
 
                if np.max(distanc)>self.growing_param[0] and \
                np.abs(err)>self.growing_param[1]:
                    self.centers_.append(newX)
                    self.coeff_ = np.append(self.coeff_, self.learning_rate *
                                           self._loss_derivative(err))
                    self.centerIndex_.append(k)
                    self._appendMCI(newX)
                    if self.kernel=='eig_mci' or self.kernel=='eig_nci':
                        self._appendEIG(newX)
            #===========================
            #     QUANTIZED GROW
            #===========================
            elif self.growing_criterion == "quantized":
                distanc = slash.ppMCIdistance(self.centers_, newX, \
                                             ksize=self.ksize, pMCI11=self.mci11)
                if np.min(distanc)<= self.growing_param[0]:
                    _min_idx = np.where(distanc == np.min(distanc))
                    self.coeff_[_min_idx] += self.learning_rate * \
                            self._loss_derivative(err)
                else:
                    self.centers_.append(newX)
                    self.coeff_ = np.append(self.coeff_, self.learning_rate *
                                           self._loss_derivative(err))
                    self.centerIndex_.append(k)
                    self._appendMCI(newX)
                    if self.kernel=='eig_mci' or self.kernel=='eig_nci':
                        self._appendEIG(newX)
        
        return self

    def _loss_derivative(self,err):
        """ 
        Evaluate the derivative of loss_function on d, y 
        
        """
        if self.loss_function == "least_squares":
            return err
        elif self.loss_function == "minimum_correntropy":
            return (err)*np.exp(-(err)**2/(2*self.correntropy_sigma**2))
        else:
            raise Exception("Invalid loss function: %s" % self.loss_function)
            
    def _appendMCI(self, newX):
        """Save precalculated data about centers to avoid repetitive
           calculations at SLASH Level II
    
        """
        newXx = slash.check_population(newX)
        MCI11 = [list() for i in range(len(newXx))]
        for i in xrange(len(newXx)):
            MCI11[i] = slash.sMCI(newXx[i], newXx[i], self.ksize)
        self.mci11.append(MCI11)
        return self
    
    def _appendEIG(self, newX):
        eig_params = {'ksize': self.ksize, 'gamma': self.gamma, \
                'spike_kernel':'pop_'+self.kernel[4:]}
        EXx = slash._eigdecompose_population(newX, **eig_params)
        if self.eig11 == []:
            self.eig11 = EXx
        else:
            self.eig11 = np.hstack([self.eig11, EXx])
        return self

    def _dropout(self):
        """
        Drops out some filters to enhance generalization

        """
        net_size = self.coeff_.shape[0]
        shuf_idx = np.random.permutation(net_size)
        if self.dropout==0:
            dropin = range(net_size)
        elif self.dropout>=0 and self.dropout<1: # if dropout is probability
            _bigger = max(1 , (net_size*self.dropout) ) 
            dropin  = shuf_idx[:_bigger]
        
        elif isinstance(self.dropout, int): # if dropout is number of units to keep
            _smaller = min(net_size, self.dropout)
            dropin = shuf_idx[:_smaller]
        else:
            raise Exception('dropout should be int or prabability')
        
        dropin_centers = range(len(dropin))
        #dropin_coeff   = np.zeros(len(dropin))
        dropin_coeff = self.coeff_[dropin]
        for i in xrange(len(dropin)):
            dropin_centers[i] = self.centers_[dropin[i]]
            #dropin_coeff[i]      = self.coeff_[dropin[i]]
        return dropin_centers, dropin_coeff
