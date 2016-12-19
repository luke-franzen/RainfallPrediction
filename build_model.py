import numpy as np
import h5py
from sklearn.metrics import mean_absolute_error
import pandas as pd
import matplotlib.pyplot as plt
from hpelm import HPELM

'''
used HP-ELM implementation here (https://pypi.python.org/pypi/hpelm/1.0.4)
trains ELM in batch, minimizing RAM requirements
was able to process 3 PB of information < 20 GB RAM 
'''

#location of pre-processed training/testing data/labels
X_train = np.load("/Users/lfranzen/bda/formatted_data/onehr/train_halfdata_101.npy", mmap_mode='r+')
y_train = np.load("/Users/lfranzen/bda/formatted_data/onehr/train_halflabels_101.npy", mmap_mode='r+')

X_test = np.load("/Users/lfranzen/bda/formatted_data/onehr/test_halfdata_101.npy", mmap_mode='r+')
y_test = np.load("/Users/lfranzen/bda/formatted_data/onehr/test_halflabels_101.npy", mmap_mode='r+')

def main():
    for neurons in range(100,2100,100):
        elm = HPELM(X_train.shape[1], 1, 'c', None, 100, None, 'single')
        elm.add_neurons(neurons, 'sigm')
        print("Starting incremental HH and HT files")
        for x in range(0, X_train.shape[0], 100):
            elm.add_data(X_train, y_train, istart=x, icount=100, fHH="HH_"+str(neurons)+".hdf5", fHT="HT_"+str(neurons)+".hdf5")

        print("Finished creating HH and HT files with "+str(neurons)+" neurons.")
        print("Starting to solve ELM with "+str(neurons)+" neurons.")
        elm.solve_corr("HH_"+str(neurons)+".hdf5", "HT_"+str(neurons)+".hdf5")
        elm.predict(X_train, "trainpreds_101_"+str(neurons)+"neurons.hdf5")
        elm.predict(X_test, "testpreds_101_"+str(neurons)+"neurons.hdf5")

if __name__ == "__main__":
    main()
