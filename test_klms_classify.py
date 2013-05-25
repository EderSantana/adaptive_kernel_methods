#import pdb; pdb.set_trace()
import numpy as np
from kernel_lms import KernelLMS

# =========================
#       Learn XOR
# =========================
klms = KernelLMS(kernel="rbf", learning_mode = "regression", learning_rate=.001,\
        gamma=.5, growing_criterion="dense",growing_param=[.1,.1],\
        loss_function="least_squares", correntropy_sigma=.4)

X = np.vstack([[0, 0], [0, 1], [1, 0], [1, 1]])
d = np.array([0, 1, 1, 0])

w = np.random.rand(2)
#w = np.ones(2)
#w = np.array([1, 1])

klms.fit_transform(X, d)                           # Initialize net
for i in xrange(1,2000):
    xout = klms.transform(X)[-4:]                  # Forwad
    yout = np.dot(w,[xout, np.ones_like(xout)])
    yout = 1/(1+np.exp(-yout))                     # Calculate last layer output
    
    e = (d-yout)*yout*(1-yout)
    w = w + .01 * (e*[xout, np.ones_like(xout)]).sum(axis=1)
    err = e*w[0]
    klms.fit_transform(X, d, err)                  # Backpropagate error
    
print "Learning XOR:"
print klms.X_transformed_[-4:]
print (yout>=.5).astype(int)
print w


# ========================
#       Learn AND
# ========================
"""
klms = KernelLMS(kernel="poly",learning_mode = "regression", learning_rate=.001,\
    gamma=.5, growing_criterion="dense",growing_param=[.1,.1], \
    loss_function="least_squares", correntropy_sigma=.4)

X = np.vstack([[0, 0], [0, 1], [1, 0], [1, 1]])
d = np.array([0, 0, 0, 1])

for i in xrange(3000):
    klms.fit_transform(X, d)

print "Learning AND"
print (klms.X_transformed_[-4:]>=.5).astype(int)
"""