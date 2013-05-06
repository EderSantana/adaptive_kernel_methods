"""Kernel Least Mean Square Algorithm"""

# Author: Eder Santana <edersantana@live.com>
# License: BSD Style.

import math
import numpy as np
from scipy import linalg

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.metrics.pairwise import pairwise_kernels, euclidean_distances


class KernelLMS(BaseEstimator, TransformerMixin):
    """Kernel Least Mean Square Algorithm (KLMS)

    Non-linear filtering in feature space by linear filtering in Hilbert spaces

    Parameters
    ----------

    learning_rate: float
        Step size for gradient descent adaptation. This parameter is very important since regularizes the kernel method and, for a given data set, define convergence time and misadjustment

    growing_criterion: "dense" | "novelty" | "surprise"
        Default: "dense:"

    growing_param: float, float, optional

    kernel: "linear" | "poly" | "rbf" | "sigmoid" | "cosine" | "precomputed"
        Kernel.
        Default: "linear"

    degree : int, default=3
        Degree for poly, rbf and sigmoid kernels. Ignored by other kernels.

    gamma : float, optional
        Kernel coefficient for rbf and poly kernels. Default: 1/n_features.
        Ignored by other kernels.

    coef0 : float, optional
        Independent term in poly and sigmoid kernels.
        Ignored by other kernels.

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
        Projection of the fitted data while filter is trained

    `X_transformed_`:
        Projection of the fitted data on the trained filter
   
    `coeff_`:
        Filter coefficients

     `centers_`:
        Centers of growing network

     `centerIndex_`:
         Indexes of the input data kept as centers kept by the network

    References
    ----------
    Kernel LMS was intoduced in:
        The Kernel LMS algorithm by Weifeng Liu et. al.
    """

    def __init__(self, kernel="rbf", learning_rate=0.01, growing_criterion="dense", growing_param=None, loss_function="least_squares", loss_param=None,
                 gamma=None, degree=3, coef0=1, kernel_params=None, correntropy_sigma=None):
        self.kernel = kernel
        self.kernel_params = kernel_params
        self.learning_rate = learning_rate
        self.loss_function = loss_function
        self.loss_param = loss_param
        self.growing_criterion = growing_criterion
        self.gamma = gamma
        self.degree = degree
        self.coef0 = coef0
        self.centers_ = np.array([])
        self.coeff_ = np.array([])
        self.centerIndex_ = []
        self.X_online_ = np.array([])
        self.X_transformed_ = np.array([])
        self.growing_param = growing_param
        self.correntropy_sigma = correntropy_sigma

 
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
                      "degree": self.degree,
                      "coef0": self.coef0}
        return pairwise_kernels(X, Y, metric=self.kernel,
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
        
        N = X.shape[0]
        self.centers_ = X[0]
        self.centerIndex_ = 0
        new_coeff = self.learning_rate * self._loss_derivative(d[0],0)
        self.coeff_ = np.append( self.coeff_, new_coeff );
        self.X_online_ = np.zeros(N)
        
        for k in range(1,N):
            gram = self._get_kernel(self.centers_,X[k])
            self.X_online_[k] = np.dot(self.coeff_, gram)
            self._appendCenter(X[k], d[k], self.X_online_[k],k)
        
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
 
    def _appendCenter(self, newX, d, y, k):
        """ Append centers to the networking following growing_criterion
            
        Returns
        -------
            `self` with possibly larger centers_, coeff_ and centerIndex_
        """
        if self.coeff_.shape == 0:
            self.centers_ = newX
            self.coeff_ = np.append(self.coeff_, self.learning_rate *
                                   self._loss_derivative(d,y))
            self.centerIndex_ = k
        else:
            if self.growing_criterion == "dense":
                self.centers_ = np.vstack([self.centers_, newX])
                self.coeff_ = np.append(self.coeff_, self.learning_rate *
                                        self._loss_derivative(d, y))
                self.centerIndex_ = [self.centerIndex_, k]

            elif self.growing_criterion == "novelty":
                """ TODO: This calculation of euclidean distances are taking to much time. Using the expanded formula and storing the X**2 terms will speeds things up. In this mode, keeping all the centers provides faster results (I tried with 3e3 centers).
                """
                distanc = euclidean_distances(self.centers_, newX, squared=True)
                if max(distanc)>self.growing_param[0] and np.abs(d-y)>self.growing_param[1]:
                    self.centers_ = np.vstack([self.centers_, newX])
                    self.coeff_ = np.append(self.coeff_, self.learning_rate *
                                           self._loss_derivative(d, y))
                    self.centerIndex_ = [self.centerIndex_, k]
        return self

    def _loss_derivative(self,d,y):
        """ Evaluate the derivative of loss_function on d, y """
        if self.loss_function == "least_squares":
            return d-y
        elif self.loss_function == "minimum_correntropy":
            return (d-y)*np.exp(-(d-y)**2/(2*self.correntropy_sigma**2))
