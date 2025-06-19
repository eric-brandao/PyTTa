# -*- coding: utf-8 -*-
"""
Created on Wed Aug  7 08:44:21 2024

@author: Eric Brandao

Deconvolution

We explore deconvolution with the exponential sweep signal (ESS) and the formulation of the inverse filter

H(f) = Y(f) * (1/X(f)), where 1/X(f) - represents the inverse filter in freq dommain. 
Due to the noise and lack of amplitude in the ESS out of its frequency range, we need to be smart
about the deconvolution. 

Furthermore, ideally X(f) * 1/X(f) = 1, which means the inverse filter convolved with the ESS should
result in an approximate dirac.
"""

#%% Imports
import pytta
import numpy as np
import matplotlib.pyplot as plt
import scipy

#%% Generate a sweep signal
fs = 44100
fft_degree = 19
start_margin = 0.1
stop_margin = 1
freq_min = 250
freq_max = 10000
method = 'logarithmic'

xt = pytta.generate.sweep(freqMin = freq_min, freqMax = freq_max, samplingRate = fs, fftDegree = fft_degree, 
                          startMargin = start_margin, stopMargin = stop_margin, method = method, 
                          windowing='hann')
xt.timeSignal = 2.35*xt.timeSignal
#%% Generate an inverse filter in frequency dommain
Cfreq = pytta.classes.signal._calculate_regu_spk(inputSignal = xt, freq_limits = [freq_min, freq_max])
Cfreq.plot_time();
#Cfreq.plot_freq(xLim = (20, 20000), yLim = (-100,80));

#%% Generate an inverse filter in time dommain
inv_sweep = pytta.generate.inverse_sweep(xt)
inv_sweep.plot_time();
#inv_sweep.plot_freq(xLim = (20, 20000), yLim = (-100,80))

#%% merge inverse filters
inv_merged = pytta.merge(Cfreq, inv_sweep)
inv_merged.plot_time();
inv_merged.plot_freq();

#%% Check if the sweep convolved with the inverse filter is a dirac
dirac_freq = xt * Cfreq
dirac_time = xt * inv_sweep
#%% plots of dirac
time = dirac_freq.timeVector
delta_freq = np.roll(dirac_freq.timeSignal/np.amax(dirac_freq.timeSignal), 
                     shift = int(len(time)/2))
delta_time = np.roll(dirac_time.timeSignal/np.amax(dirac_time.timeSignal), 
                     shift = int(len(time)/2))

punit = 0.0001
plt.figure(figsize=(15,3))
plt.subplot(1,3,1)
plt.plot(time, delta_freq, linewidth = 1)
plt.plot(time, delta_freq, 'ok')
plt.axvline(time[-1]/2, linestyle = '--', color = 'grey')
plt.grid(linestyle = '--', which='both')
plt.xlim(((1-punit)*time[-1]/2, (1+punit)*time[-1]/2))
plt.xlabel('Time (s)')
plt.ylabel(r'$\delta(t)$')

plt.subplot(1,3,2)
plt.plot(time, delta_time, linewidth = 1)
plt.plot(time, delta_time, 'ok')
plt.axvline(time[-1]/2, linestyle = '--', color = 'grey')
plt.axvline(time[-1]/2 + 0.5/xt.samplingRate, linestyle = '--', color = 'brown', alpha = 0.7)
plt.grid(linestyle = '--', which='both')
plt.xlim(((1-punit)*time[-1]/2, (1+punit)*time[-1]/2))
plt.xlabel('Time (s)')
plt.ylabel(r'$\delta(t)$')

plt.subplot(1,3,3)
plt.plot(time, delta_freq, label = 'freq')
plt.plot(time, delta_time, label = 'time')
plt.legend()
plt.grid(linestyle = '--', which='both')
plt.xlim(((1-0.0005)*time[-1]/2, (1+0.0005)*time[-1]/2))
plt.xlabel('Time (s)')
plt.ylabel(r'$\delta(t)$')
plt.tight_layout()





#%% Loading measured files
xt_dict = pytta.load('xt_17102024.hdf5')
xt_meas = xt_dict[list(xt_dict.keys())[0]] # this is a signal object
xt_meas.startMargin = start_margin
xt_meas.stopMargin = stop_margin

yt_dict = pytta.load('yt_17102024.hdf5')
yt = yt_dict[list(yt_dict.keys())[0]] # this is a signal object
yt.plot_freq()
yt_meas = yt.split()
yt_meas[1].startMargin = start_margin
yt_meas[1].stopMargin = stop_margin

#%% Deconv - regularized sweep as it is in class
ht = pytta.ImpulsiveResponse(excitation = yt_meas[1], recording = yt_meas[0], 
                             samplingRate = xt.samplingRate, regularization = True,
                             freq_limits = [250, 10000], method = 'linear')
ht.plot_time_dB()

#%% Testing methods
ht_naive = ht._naive_deconv(yt_meas[1], yt_meas[0])
ht_regu = ht._regularized_deconv(yt_meas[1], yt_meas[0], freq_limits = [250, 10000])
ht_invfilter = ht._deconv_invfilter(yt_meas[1], yt_meas[0])

#ht_regu_zp = ht._regularized_zp_deconv(yt_split[1], yt_split[0], freq_limits = [100, 10000], num_zeros = None)
# ht_invfilter_fd = ht._deconv_invfilter_fd(yt_split[1], yt_split[0], freq_limits = [100, 10000])
#%% Plots
hts = pytta.merge(ht_regu, ht_invfilter)
hts.plot_time_dB();
hts.plot_freq();

#%%
mean_diff = np.abs(ht_regu.freqSignal)/np.abs(ht_invfilter.freqSignal)
plt.figure()
plt.semilogx(ht_regu.freqVector, 20*np.log10(mean_diff))
plt.xlim((100, 10000))
#%%
plt.figure(figsize = (8, 4))
plt.plot(ht_regu.timeVector, 20*np.log10(np.abs(ht_regu.timeSignal)/np.amax(np.abs(ht_regu.timeSignal))), 
         alpha = 1, label = 'original')
plt.plot(ht_invfilter.timeVector, 
         20*np.log10(np.abs(ht_invfilter.timeSignal)/np.amax(np.abs(ht_invfilter.timeSignal))), '-', 
          linewidth = 0.5, alpha = 0.8, label = 'inv filter')
# plt.plot(ht_regu.timeVector, 20*np.log10(np.abs(ht_regu.timeSignal)/np.amax(np.abs(ht_regu.timeSignal))), 
#           alpha = 0.4, label = 'regularized')
# plt.plot(ht_regu_zp.timeVector, 20*np.log10(np.abs(ht_regu_zp.timeSignal)/np.amax(np.abs(ht_regu_zp.timeSignal))), 
#           alpha = 0.8, label = 'regularized')
# plt.plot(ht_welch_h1.timeVector, 20*np.log10(np.abs(ht_welch_h1.timeSignal)/np.amax(np.abs(ht_welch_h1.timeSignal))), 
#           alpha = 0.8, label = 'H1')

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

plt.plot(ht_invfilter.freqVector, 20*np.log10(np.abs(ht_invfilter.freqSignal)), 
          alpha = 0.4, label = 'inv filter')
# plt.plot(ht_regu.timeVector, 20*np.log10(np.abs(ht_regu.timeSignal)/np.amax(np.abs(ht_regu.timeSignal))), 
#           alpha = 0.4, label = 'regularized')
# plt.plot(ht_regu_zp.freqVector, 20*np.log10(np.abs(ht_regu_zp.freqSignal)), 
#           alpha = 0.8, label = 'regularized w/ zp')
# plt.plot(ht_welch_h1.freqVector, 20*np.log10(np.abs(ht_welch_h1.freqSignal)), 
#           alpha = 0.8, label = 'H1')

# plt.plot(ht_naive.timeVector, 20*np.log10(np.abs(ht_naive.timeSignal)/np.amax(np.abs(ht_naive.timeSignal))), 
#          alpha = 0.4, label = 'naive')
plt.legend()
plt.xlabel('Time (s)')
plt.ylabel('Magnitude (dB)')
plt.xlim((100, 10000))
#plt.ylim((-80, 10))
plt.grid()
plt.tight_layout()



# #%% ######## Import C computed from ITA toolbox
# folder = "D:/Work/UFSM/Disciplinas/PDS1/Aulas/"
# #C_ita = pytta.read_wav(folder + "ft_ita_from_pyttasweep.wav")
# C_ita = scipy.io.loadmat(folder + "ft_ita_from_pyttasweep.mat")
# C_ita_pytta = pytta.SignalObj(C_ita['Csig'], 'time', C.samplingRate, signalType='energy') # transform to energy - it is in ITA-toolbox

# #%%
# scipy.io.savemat(folder + "ft_pyttaSO_to_ita.mat", {'C_ita_pytta':C_ita_pytta.timeSignal})

# #%%
# errornorm_in_time = np.linalg.norm(C.timeSignal- C_ita_pytta.timeSignal)/np.linalg.norm(C.timeSignal)

# plt.figure()
# plt.plot(C.timeVector, C.timeSignal)
# plt.plot(C.timeVector, C_ita_pytta.timeSignal, alpha = 0.5)
# plt.plot(C.timeVector, C.timeSignal- C_ita_pytta.timeSignal, 'k')

# # plt.figure()
# # plt.plot( C.timeSignal- C_ita_pytta.timeSignal, 'k')

# plt.figure()
# plt.semilogx(C.freqVector, 20*np.log10(np.abs(C.freqSignal)))
# plt.plot(C.freqVector, 20*np.log10(np.abs(C_ita_pytta.freqSignal)))
# plt.grid()
# plt.xlim((20, 20000))
