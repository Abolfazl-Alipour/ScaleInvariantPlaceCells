#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun  5 12:11:39 2023

@author: panlab
"""
import numpy as np
import scipy.io as sio
from numpy import genfromtxt
import seaborn as sns
import matplotlib.pyplot as plt
from nice_figures import *

load_style('APS', '1-column')


inputSig = sio.loadmat('figureForanalyticalResults.mat')['f_tau']
inputSig = np.array(inputSig,dtype='float64')
decays = sio.loadmat('DecayUnitsForFigure.mat')['t']
decays = np.array(decays)
placeCells = sio.loadmat('PlaceCellsForFigure.mat')['T']
placeCells = np.array(placeCells)
fig=plt.figure(figsize=(7.5,15))
ax=fig.add_subplot(111)
ax=plt.Figure(),plt.plot( inputSig.squeeze()),plt.ylim([-.05,1.05]),plt.xlim([-100,10100]),plt.show()
ax.spines[['right', 'top']].set_visible(False)
fig=plt.figure(figsize=(7.5,15))
ax=fig.add_subplot(111)
plt.Figure(),plt.plot(decays[10,:]),plt.plot(decays[20,:]),plt.plot(decays[30,:]),plt.plot(decays[40,:]),plt.plot(decays[50,:]),plt.plot(decays[60,:]),plt.plot(decays[70,:]),plt.xlim([-100,10100]),plt.show()
fig=plt.figure(figsize=(7.5,15))
ax=fig.add_subplot(111)
plt.Figure(),plt.plot(placeCells[10,:]),plt.plot(placeCells[20,:]),plt.plot(placeCells[30,:]),plt.plot(placeCells[40,:]),plt.plot(placeCells[50,:]),plt.plot(placeCells[60,:]),plt.plot(placeCells[70,:]),plt.xlim([-100,10100]),plt.show()
# plt.Figure(),sns.heatmap(placeCells)
