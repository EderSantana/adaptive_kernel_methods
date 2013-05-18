import numpy as np
from kernel_lms import KernelLMS

klms = KernelLMS(kernel="linear", learning_mode = "regression", learning_rate=.001, gamma=.5,\
growing_criterion="dense",growing_param=[1,.4], loss_function="least_squares",\
correntropy_sigma=.4)

# Learn XOR
X = np.vstack([[0, 0], [0, 1], [1, 0], [1, 1]])
d = np.array([0, 1, 1, 0])

for i in xrange(1000):
    d = np.array([0, 1, 1, 0])
    klms.fit_transform(X, d)
    
print klms.X_transformed_[-4:]

klms = KernelLMS(kernel="linear",learning_mode = "regression", learning_rate=.001, gamma=.5,\
growing_criterion="dense",growing_param=[1,.4], loss_function="least_squares",\
correntropy_sigma=.4)

# Learn AND
X = np.vstack([[0, 0], [0, 1], [1, 0], [1, 1]])
d = np.array([0, 0, 0, 1])

for i in xrange(1000):
    d = np.array([0, 1, 1, 0])
    klms.fit_transform(X, d)
    
print klms.X_transformed_[-4:]
