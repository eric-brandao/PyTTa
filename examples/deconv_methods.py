# -*- coding: utf-8 -*-
"""
Created on Wed Aug  7 08:44:21 2024

@author: Eric Brandao

Deconvolution - some methods
"""

import pytta
import numpy as np
import matplotlib.pyplot as plt

#%% Loading measured files
xt_dict = pytta.load('xt.hdf5')
xt = xt_dict[list(xt_dict.keys())[0]] # this is a signal object

yt_dict = pytta.load('yt.hdf5')
yt = yt_dict[list(yt_dict.keys())[0]] # this is a signal object
yt_split = yt.split()
#%% Deconv - regularized sweep as it is in class
ht = pytta.ImpulsiveResponse(excitation = xt, recording = yt_split[0], 
                             samplingRate = xt.samplingRate, regularization = True,
                             freq_limits = [100, 10000], method = 'linear')

#%% Testing methods
ht_naive = ht._naive_deconv(xt, yt_split[0])
ht_regu = ht._regularized_deconv(xt, yt_split[0], freq_limits = [100, 10000])
ht_regu_zp = ht._regularized_zp_deconv(xt, yt_split[0], freq_limits = [100, 10000], num_zeros = None)
ht_welch_h1 = ht._welch_h1_deconv(xt, yt_split[0],winType = 'hann', 
                     winSize = 2**16, overlap = 0.6)

#%% Plots
plt.figure(figsize = (8, 4))
plt.plot(ht.IR.timeVector, 20*np.log10(np.abs(ht.IR.timeSignal)/np.amax(np.abs(ht.IR.timeSignal))), 
         alpha = 1, label = 'original')
# plt.plot(ht_regu.timeVector, 20*np.log10(np.abs(ht_regu.timeSignal)/np.amax(np.abs(ht_regu.timeSignal))), 
#           alpha = 0.4, label = 'regularized')
# plt.plot(ht_regu_zp.timeVector, 20*np.log10(np.abs(ht_regu_zp.timeSignal)/np.amax(np.abs(ht_regu_zp.timeSignal))), 
#           alpha = 0.8, label = 'regularized')
plt.plot(ht_welch_h1.timeVector, 20*np.log10(np.abs(ht_welch_h1.timeSignal)/np.amax(np.abs(ht_welch_h1.timeSignal))), 
          alpha = 0.8, label = 'H1')

# plt.plot(ht_naive.timeVector, 20*np.log10(np.abs(ht_naive.timeSignal)/np.amax(np.abs(ht_naive.timeSignal))), 
#          alpha = 0.4, label = 'naive')
plt.legend()
plt.xlabel('Time (s)')
plt.ylabel('Magnitude (dB)')
#plt.xlim((-0.1, ht.IR.timeVector[-1]))
plt.ylim((-80, 10))
plt.grid()
plt.tight_layout()

#%%
plt.figure(figsize = (8, 4))
plt.semilogx(ht.IR.freqVector, 20*np.log10(np.abs(ht.IR.freqSignal)), 
         alpha = 1, label = 'original')
# plt.plot(ht_regu.timeVector, 20*np.log10(np.abs(ht_regu.timeSignal)/np.amax(np.abs(ht_regu.timeSignal))), 
#           alpha = 0.4, label = 'regularized')
plt.plot(ht_regu_zp.freqVector, 20*np.log10(np.abs(ht_regu_zp.freqSignal)), 
          alpha = 0.8, label = 'regularized w/ zp')
plt.plot(ht_welch_h1.freqVector, 20*np.log10(np.abs(ht_welch_h1.freqSignal)), 
          alpha = 0.8, label = 'H1')

# plt.plot(ht_naive.timeVector, 20*np.log10(np.abs(ht_naive.timeSignal)/np.amax(np.abs(ht_naive.timeSignal))), 
#          alpha = 0.4, label = 'naive')
plt.legend()
plt.xlabel('Time (s)')
plt.ylabel('Magnitude (dB)')
plt.xlim((100, 10000))
plt.ylim((-80, 10))
plt.grid()
plt.tight_layout()


