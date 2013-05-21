import pylab as pl
import numpy as np
from numpy.linalg import eig
from numpy.random import rand, randn
from slash import sMCI
from sklearn.decomposition import KernelPCA
from sklearn.preprocessing import KernelCenterer

np.random.seed(22)

def generate_spike_classes(n_classes, n_templates):
    classes = [{} for i in range(n_classes)]
    for k in xrange(n_classes):
        class_temp = []
        for i in xrange(n_templates):
            template = .25 * rand(10)
            class_temp.append(np.outer(np.ones(10), template) * \
                     (rand(10,10)<=.8))
        classes[k] = np.vstack(class_temp)
    classes = np.vstack(np.array(classes))
    return classes

def generate_spike_times(classes):
    population = [{} for i in range(40)]

    for k in xrange(40):
        temp = classes[k] + .003*randn(10)
        temp = temp[classes[k]>0]
        temp.sort()
        population[k] = temp
            
    return population

def compute_K_matrix(X, Y=None):
    if Y is None:
        K = np.zeros([len(X), len(X)])
    else:
        K = np.zeros([len(X), len(Y)])

    if Y is None or Y is X:
        Y = X
        for i in xrange(len(X)):
            for j in range(i):
                K[i,j] = sMCI(X[i], Y[j], .002)
                K[j,i] = K[i,j]
    else:
        for i in xrange(len(X)):
            for j in range(len(Y)):
                K[i,j] = sMCI(X[i], Y[j], ksize = .002)

    return K
#if __init__ == __main__:

classes = generate_spike_classes(2 ,2)
train = generate_spike_times(classes)
test  = generate_spike_times(classes)
K = compute_K_matrix(train)
###############################
kcenterer = KernelCenterer()  #
kcenterer.fit(K)              # Center Kernel Matrix
Kc = kcenterer.transform(K)   #
###############################
D, E = eig(Kc)
proj = np.dot(Kc, E[:,0:2])

Kt = compute_K_matrix(train, test)
Ktc = kcenterer.transform(Kt)
proj2 = np.dot(Kt, E[:,0:2])

kpca = KernelPCA(kernel="precomputed", n_components=2)
kpca.fit(Kc)
X = kpca.transform(Kc)

###############################
# Plot results
###############################
red = range(20)
blu = range(21,40)
pl.plot(proj2[red,0], proj2[red,1], '.r', proj2[blu,0], proj2[blu,1], '.b')
pl.show()
