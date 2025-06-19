# -*- coding: utf-8 -*-
"""
Created on Fri Sep 20 15:07:12 2024

@author: Eric Brandao
"""

import pytta
import numpy as np

#%% Generate a sin signal and plot
sin1 = pytta.generate.sin(Amp = 1000, freq=1000,
        timeLength = 6, phase = 0, samplingRate = 44100, fftDegree=None)
sin1.plot_time(xLim = (0, 0.003))
sin1.plot_freq(xLim = (500, 2000), yLim = (-10, 3))
#%% Use the time signal to pass to a SignalObj (time dommain)
sin2 = pytta.SignalObj(sin1.timeSignal, domain = 'time', samplingRate=sin1.samplingRate,
                      signalType = 'power')
sin2.plot_time(xLim = (0, 0.003))
sin2.plot_freq(xLim = (500, 2000), yLim = (-10, 3))
residual = np.abs(sin2.freqSignal - sin1.freqSignal)
print("Avg. residual norm: {}".format(np.linalg.norm(residual)/len(residual)))
#%% Use the freq signal to pass to a SignalObj (freq dommain)
sin3 = pytta.SignalObj(sin1.freqSignal, domain = 'freq', samplingRate=sin1.samplingRate,
                      signalType = 'power')
sin3.plot_time(xLim = (0, 0.003))
sin3.plot_freq(xLim = (500, 2000), yLim = (-10, 3))
residual = sin3.timeSignal - sin1.timeSignal
print("Avg. residual norm: {}".format(np.linalg.norm(residual)/len(residual)))

#%% Create a log sweep
sweep = pytta.generate.sweep(freqMin = 100, freqMax = 10000, samplingRate = 44100, 
                             fftDegree = 19, startMargin = 0.1, stopMargin = 0.5, 
                             method = 'logarithmic', windowing='hann')
sweep.plot_time()
sweep.plot_freq(xLim = (20, 20000))

#%% Use the time signal to pass to a SignalObj (freq dommain)
sweep_f = pytta.SignalObj(sweep.freqSignal, domain = 'freq', samplingRate=sweep.samplingRate,
                      signalType = 'power')
# sweep_f.plot_time()
# sweep_f.plot_freq(xLim = (20, 20000))
sweep_f.plot_spectrogram(yLim = (100, 20000), normalize = False, winSize = 2048,
                         dinamic_range = 70, log_in_freq_scale = True)
residual = sweep_f.timeSignal - sweep.timeSignal
print("Avg. residual norm: {}".format(np.linalg.norm(residual)/len(residual)))