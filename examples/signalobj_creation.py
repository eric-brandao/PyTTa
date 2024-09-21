# -*- coding: utf-8 -*-
"""
Created on Fri Sep 20 15:07:12 2024

@author: Eric Brandao
"""

import pytta
import numpy as np

#%% Generate a sin signal and plot
sin1 = pytta.generate.sin(Amp = 1.0, freq=1000,
        timeLength = 1, phase = 0, samplingRate = 44100, fftDegree=None)

sin1.plot_time(xLim = (0, 0.003))
sin1.plot_freq(xLim = (500, 2000), yLim = (-10, 3))

#%% Use the time signal to pass to a SignalObj (time dommain)

sin2 = pytta.SignalObj(sin1.timeSignal, domain = 'time', samplingRate=sin1.samplingRate,
                      signalType = 'power')

sin2.plot_time(xLim = (0, 0.003))
sin2.plot_freq(xLim = (500, 2000), yLim = (-10, 3))

#%% Use the time signal to pass to a SignalObj (freq dommain)

sin3 = pytta.SignalObj(sin1.freqSignal, domain = 'freq', samplingRate=sin1.samplingRate,
                      signalType = 'energy')

sin3.plot_time(xLim = (0, 0.003))
sin3.plot_freq(xLim = (500, 2000), yLim = (-10, 3))
