import slash
import pylab as pl
import numpy as np
from numpy.linalg import eig
from numpy.random import rand, randn
from slash import sMCI, sNCI, ppMCI, ppNCI
from sklearn.decomposition import KernelPCA
from sklearn.preprocessing import KernelCenterer

def rasterPlot(population):
    pl.figure()
    for i in range(len(population)):
        yax = i + np.ones_like(population[i])
        pl.plot(population[i], yax, '.k')
    pl.show()

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

    population = [{} for i in range(len(classes))]

    for k in xrange(len(classes)):
        temp = classes[k] + .003*randn(10)
        temp = temp[temp>0]
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
                K[i,j] = ppMCI([X[i]], Y[j], .002, .01)
                K[j,i] = np.conj(K[i,j])
    else:
        for i in xrange(len(X)):
            for j in range(len(Y)):
                K[i,j] = ppMCI([X[i]], Y[j], .002, .01)

    return K

if __name__ == '__main__':

    classes1 = generate_spike_classes(1 ,2)
    classes2 = generate_spike_classes(1 ,2)
    train = [{} for i in range(40)]
    test = [{} for i in range(40)]
    for i in xrange(20):
        train[i] = generate_spike_times(classes1)
        test[i]  = generate_spike_times(classes1)
    for i in xrange(20):
        train[i+20] = generate_spike_times(classes2)
        test[i+20]  = generate_spike_times(classes2)

    rasterPlot(train[0])
    rasterPlot(train[30])

    K = compute_K_matrix(train)
    #K = slash.peigNCI(train)
    ###############################
    #N = K.shape[0]
    #H = np.eye(N) - np.tile(1./N, [N, N]);
    #Kc = np.dot(np.dot(H, K), H)
    kcenterer = KernelCenterer()  #
    kcenterer.fit(K)              # Center Kernel Matrix
    Kc = kcenterer.transform(K)   #
    ###############################
    D, E = eig(Kc)
    proj = np.dot(Kc, E[:,0:2])
    
    ################################ Center test
    Kt = compute_K_matrix(train, test)
    #Kt = slash.peigNCI(train, test)
    #M = Kt.shape[0]
    #A = np.tile(K.sum(axis=0), [M, 1]) / N
    #B = np.tile(Kt.sum(axis=1),[N, 1]) /N
    #Kc2 = Kt - A - B + K.sum()/ N**2;
    Kc2 = kcenterer.transform(Kt)
    proj2 = np.dot(Kc2, E[:,0:2])

    #kpca = KernelPCA(kernel="precomputed", n_components=2)
    #kpca.fit(Kc)
    #X = kpca.transform(Kc)

    ###############################
    # Plot results
    ###############################
    red = range(len(test)/2)
    blu = range(len(test)/2 + 1, len(test))
    pl.figure()
    pl.plot(proj2[red,0], proj2[red,1], '.r', proj2[blu,0], proj2[blu,1], '.b')
    pl.show()
