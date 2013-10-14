import numpy as np
import scipy as sp
import pylab as pl
import time
from kernel_lms import KernelLMS


t = np.arange(0,1500,.5)
f = .09
N = t.shape[0]-1
x1 = np.sin( 2*np.pi*f*t[0:-1] )
x2 = np.sin( 2*np.pi*f*t[1:] )
X = np.vstack( [x1, x2] ).T
d = np.sin( 4*np.pi*f*t[1:] )
n1 = .01*np.random.randn(np.floor(.9*N))
n2 = .01*np.random.randn(np.ceil(.1*N))+2
n = np.random.permutation(np.hstack([n1,n2]))
dhat = d + n;

klms  = []
klms = KernelLMS(learning_rate=.9, gamma=1, growing_criterion="novelty", 
                 growing_param=[1,.4], loss_function="minimum_correntropy",
                 correntropy_sigma=.4, dropout=.5)

t1 = time.time()
klms.fit_transform(X,dhat)
t2 = time.time()
print "Elapsed time = %f" % (t2-t1)

plot(klms.X_transformed_)
plot(d)