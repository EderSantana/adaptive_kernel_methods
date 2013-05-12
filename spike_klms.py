"""Spike Kernel Least Mean Square Algorithm"""

# Author: Eder Santana <edersantanajunior@hotmail.com>
# License: BSD Style.

import numpy as np
import slash

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.metrics.pairwise import pairwise_kernels, euclidean_distances


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

    growing_criterion: "dense" | "novelty" | "surprise"
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

    alpha: int
        Hyperparameter of the ridge regression that learns the
        inverse transform (when fit_inverse_transform=True).
        Default: 1.0

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
                 correntropy_sigma=None):
        self.kernel = kernel
        self.kernel_params = kernel_params
        self.learning_rate = learning_rate
        self.loss_function = loss_function
        self.loss_param = loss_param
        self.growing_criterion = growing_criterion
        self.gamma = gamma
        self.ksize = ksize
        self.centers_ = np.array([])
        self.coeff_ = np.array([])
        self.centerIndex_ = []
        self.X_online_ = np.array([])
        self.X_transformed_ = np.array([])
        self.growing_param = growing_param
        self.correntropy_sigma = correntropy_sigma
        self.XX = 0

 
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
                      "ksize": self.ksize}
        return slash.inner_prod(X, Y, spike_kernel=self.kernel,
                                filter_params=True, **params)

    def fit(self, X, d, **params):
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
        
        N = len(X)
        self.centers_ = X[0]
        if self.growing_param != "dense":
            # TODO: add this treatment [X] at SLASH
            self.XX = self._get_kernel([X], X)
        self.centerIndex_ = 0
        new_coeff = self.learning_rate * self._loss_derivative(d[0],0)
        self.coeff_ = np.append( self.coeff_, new_coeff );
        self.X_online_ = np.zeros(N)
        
        for k in range(1,N):
            gram = self._get_kernel(self.centers_,X[k])
            self.X_online_[k] = np.dot(self.coeff_, gram)
            self._appendCenter(X[k], d[k], self.X_online_[k],k,self.XX)
        
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
       
        Z_out = np.dot(self.coeff_, self._get_kernel(self.centers_,Z))
        return Z_out

    def fit_transform(self, X, d, **params):
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
        self.fit(X, d, **params)

        self.X_transformed_ = self.transform(X)

        return self.X_transformed_
 
    def _appendCenter(self, newX, d, y, k, XX):
        """ Append centers to the network following growing_criterion
            
        Returns
        -------
            `self` with possibly larger centers_, coeff_ and centerIndex_
        """
        if self.coeff_.shape == 0:
            self.centers_ = list([newX])
            self.coeff_ = np.append(self.coeff_, self.learning_rate *
                                   self._loss_derivative(d,y))
            self.centerIndex_ = k
        else:
            if self.growing_criterion == "dense":
                self.centers_.append(newX)                
                self.coeff_ = np.append(self.coeff_, self.learning_rate *
                                        self._loss_derivative(d, y))
                self.centerIndex_ = [self.centerIndex_, k]

            elif self.growing_criterion == "novelty":
                """ The calculation of the euclidean distances were taking to much time. Using the expanded formula and storing the XX=X**2 terms will speeds things up.
                """
                _precalc = self.XX
                distanc = slash.pMCIdistance(self.centers_, newX, \
                                             ksize=self.ksize, **_precalc)
 
                if np.max(distanc)>self.growing_param[0] and np.abs(d-y)>self.growing_param[1]:
                    self.centers_.append(newX)
                    self.coeff_ = np.append(self.coeff_, self.learning_rate *
                                           self._loss_derivative(d, y))
                    self.centerIndex_ = [self.centerIndex_, k]
                    self.XX = self._appendXX(newX)
        return self

    def _loss_derivative(self,d,y):
        """ Evaluate the derivative of loss_function on d, y """
        if self.loss_function == "least_squares":
            return d-y
        elif self.loss_function == "minimum_correntropy":
            return (d-y)*np.exp(-(d-y)**2/(2*self.correntropy_sigma**2))
        else:
            raise Exception("Invalid loss function: %s" % self.loss_function)
            
    def _appendXX(self, newX):
        """Save precalculated data about centers to avoid repetitive 
            calculations at SLASH Level II
            """
        x_pos_exp = np.exp(newX / self.ksize)
        x_pos_cum_sum_exp = np.cumsum(x_pos_exp)
        
        x_neg_exp = np.exp(-newX / self.ksize)
        x_neg_cum_sum_exp = np.flipud(x_neg_exp)
        x_neg_cum_sum_exp = np.cumsum(x_neg_cum_sum_exp)
        x_neg_cum_sum_exp = np.flipud(x_neg_cum_sum_exp)
        
        X_pos_exp = (self.XX).get("X_pos_exp")
        X_pos_exp = np.append(X_pos_exp, x_pos_exp)
        X_pos_cum_sum_exp = (self.XX).get("X_pos_cum_sum_exp")
        X_pos_exp = np.append(X_pos_cum_sum_exp, x_pos_cum_sum_exp)
        X_neg_exp = (self.XX).get("X_neg_exp")
        X_neg_exp = np.append(X_neg_exp, x_neg_exp)
        X_neg_cum_sum_exp = (self.XX).get("X_neg_cum_sum_exp")
        X_neg_exp = np.append(X_neg_cum_sum_exp, x_neg_cum_sum_exp)
        
        (self.XX).updata({"X_neg_exp": X_neg_exp, "X_pos_exp": X_pos_exp,\
        "X_neg_cum_sum_exp": X_neg_cum_sum_exp, \
        "X_pos_cum_sum_exp": X_neg_cum_sum_exp
        })
            
                
