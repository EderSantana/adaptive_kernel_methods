# Adaptive Kernel Methods following scikit-learn API
Here we mainly test the Kernel LMS algorithm for R^n vectors and spike trains. 
KernelLMS deals with Rˆn vectors and SpikeKLMS with spike times series
To test those methods either run 'test_KLMS.py' or 'test_SpikeKLMS.py'.
This will show how the methods work for regression. There also an example of KLMS for classification at 'test_klms_classify.py'

Spike_KLMS algorithm used SLASH library for spike train signal processing, which is included here.
SLASH provides spike-spike, population-spike and population-population inner products.
To test SLASH for unserpervised learning, we included Paiva's PCA at 'test_paiva.py' and a modification for populations at 'test_populationPCA.py' 

