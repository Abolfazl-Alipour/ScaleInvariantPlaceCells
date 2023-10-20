#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr  3 18:21:30 2023

@author: panlab
"""
import torch
import matplotlib.pyplot as plt
import numpy as np
from nice_figures import *
load_style('APS', '1-column')
CnctvtyMat=torch.load('USEMEweightMatrixactivityPunishment0CA1Size100Dropout0weight_decayFactor0rho0.7.pt')
# UseMEweightMatrixactivityPunishment Val0CA1Size100Dropout0weight_decayFactor0rho0.1
CnctvtyMat=np.asarray(CnctvtyMat.detach().cpu())
fig, ax = plt.subplots()
ax.matshow(CnctvtyMat.T)
fig, ax = plt.subplots()
# vector4Plot=CnctvtyMat[40:59,49]
vector4Plot=[-1.1952325105667114,
        -1.5893654823303223,
        0.6046406626701355,
        0.4105531573295593,
        0.18251751363277435,
        0.8744351267814636,
        -1.2761683464050293,
        -0.18953658640384674,
        -1.712805986404419,
        0.2640540301799774,
        0.996334433555603,
        -0.27204352617263794,
        0.555494487285614,
        -0.19586658477783203,
        -0.8023511171340942,
        0.032471563667058945,
        0.5348628759384155,
        0.31476739048957825,
        0.28037047386169434]
ax.plot(np.linspace(0, 18,19), vector4Plot)

ax.set_xticks([0, 9, 18])
ax.set_xticklabels(['-9', '0', '9'])
ax.tick_params(axis="x", labelsize=14)
ax.tick_params(axis="y", labelsize=14)
# ax.set_aspect(22/24)
ax.set_aspect(22/4)
# plt.grid(which='both', linewidth=.1)

# plt.ylim([-12,12])
plt.xlim([-2,20])
plt.ylim([-2,2])

# ax.legend(fontsize=26,loc='upper left')
plt.xlabel('NeuronID',fontsize=14)
plt.ylabel('Weight Value',fontsize=14)
# fig.tight_layout()

fig.show()
#%%
fig, ax = plt.subplots()
ax.plot(CnctvtyMat[:,49]),plt.plot(np.array([50,50]),np.array([-2,2]),'k')
plt.xlabel('Source NeuronID',fontsize=24)
plt.ylabel('Weight Value',fontsize=24)
plt.show()
# ax.legend(fontsize=26,loc='upper left')
fig.tight_layout()
plt.savefig('WeightMatrixCrossSection'+'.png')

# rects1 = ax.bar(x - width/2, ObjAccMean*100, width,label='Object',color='blue',capsize=5)
# rects2 = ax.bar(x + width/2, widthAccMean*100, width, label='Width',color='lightcoral',capsize=5)

# # Add some text for labels, title and custom x-axis tick labels, etc.
# ax.set_ylabel('Training Accuracy (%)',fontsize=24)
# #ax.set_xlabel('Network Size',fontsize=24)
# ax.set_title('Training accuracy in feedforward classifiers',fontsize=28)
# ax.set_xticks(x)
# ax.set_xticklabels(xlabels,fontsize=24)
# #ax.set_yticks(yticks)
# #ax.set_yticklabels(ylabels,fontsize=24)
# ax.tick_params(axis="y", labelsize=24)
# ax.legend(fontsize=26,loc='upper left')
# ax.set_ylim(40, 95)
# fig.tight_layout()



plt.show()