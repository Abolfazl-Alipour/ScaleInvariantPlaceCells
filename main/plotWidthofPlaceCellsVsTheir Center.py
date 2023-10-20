#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May 10 15:44:18 2023

@author: panlab
"""

import json
import numpy as np
import matplotlib.pyplot as plt
from nice_figures import *
load_style('APS', '1-column')
with open('AllCells Val_9946_f807685b08245d79a254.plotly.json') as f:
    d = json.load(f)
    print(d)


dataDict=d['data'][0]
allCells=np.asarray(dataDict['z'])
plt.imshow(allCells)

    
indxHolder = np.full([allCells.shape[0]], np.nan)
# indxHolder[:, 0] = range(ActivityCollector4Time.shape[0])
# indxHolder[:, 1] = (ActivityCollector4Time != 0).argmax(axis=1)
for NeuronID in range(np.size(allCells, axis=0)):
    indxHolder[NeuronID] = [NeuronID, np.argmax(allCells[NeuronID])][1]


widthHolder = np.full([allCells.shape[0]], np.nan)

for NeuronID in range(np.size(allCells, axis=0)):
    widthHolder[NeuronID] = np.sum(allCells[NeuronID]>.5)

fig=plt.figure(figsize=(15,15))
ax=fig.add_subplot(111)
ax.set_title("width vs center",fontsize=12)
plt.xlabel('width',fontsize=12)
plt.ylabel('center',fontsize=12)
ax.tick_params(axis="x", labelsize=8)
ax.tick_params(axis="y", labelsize=8)
ax.plot(indxHolder,widthHolder ,'.b')
plt.xscale('log')
plt.yscale('log')
plt.show()


fig=plt.figure(figsize=(15,15))
ax=fig.add_subplot(111)
histData=plt.hist(indxHolder,100)
plt.xscale('log')
plt.yscale('log')
plt.show()

fig=plt.figure(figsize=(15,15))
ax=fig.add_subplot(111)
ax.plot(histData[1][1:],histData[0],'.k')
plt.xlabel('Center',fontsize=12)
plt.ylabel('Count',fontsize=12)
plt.xscale('log')
plt.yscale('log')
plt.show()
