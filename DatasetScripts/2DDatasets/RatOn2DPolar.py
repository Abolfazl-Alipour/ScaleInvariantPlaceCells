#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 14 09:06:38 2023

@author: panlab
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb  9 17:25:24 2023

@author: panlab
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Dec 17 05:40:02 2022
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""


# Time signals
import os
os.chdir('/media/panlab/Thesis/DeepCodes')
from RatSimulatorForBackup import RatSimulator
os.chdir('/media/panlab/Thesis/DeepCodes')
import numpy as np
import matplotlib.pyplot as plt
import torch
from tqdm import tqdm 
import seaborn as sns
from torch.distributions import VonMises
from decimal import *
import math
from nice_figures import *
from matplotlib.patches import Wedge
load_style('APS', '1-column')
np.random.seed(1234)
def cart2pol(x, y):
    rho = np.sqrt(x**2 + y**2)
    phi = np.arctan2(y, x)
    return(rho, phi)

def pol2cart(rho, phi):
    x = rho * np.cos(phi)
    y = rho * np.sin(phi)
    return(x, y)

class RatOnPlanePolar():
    def __init__(self, n_steps,GoalLocation=False):
        self.n_steps=n_steps
        self.dt=0.02
        self.end=np.array([1,1])
        self.start=0
        self.rat=RatSimulator(self.n_steps)
        self.theta_bin_centers=1
        
    def generateTrajectory(self,figureOn=False,CA1Size=100,upper=0,lower=-1,linearlySpaced=False):

        hiddenSizeMEC=100 #self.theta_bin_centers*
        [Rvelocities, angVel, loc, angle]=self.rat.generateTrajectory()

        Rs=np.sqrt(loc[:,0]**2+loc[:,1]**2)
        # Rs=Rs[1:]
        absAngles=np.arctan2(loc[:,1],loc[:,0])

        
        
        Rvelocities=np.diff(Rs,axis=0)
        if linearlySpaced:
            s=torch.linspace(upper,lower,int(hiddenSizeMEC/self.theta_bin_centers))#
        else:
            s=torch.logspace(np.log10(lower),np.log10(upper),int(hiddenSizeMEC/self.theta_bin_centers))#
        
        
        x_out1X=torch.zeros([Rvelocities.shape[0]+1,int(hiddenSizeMEC/self.theta_bin_centers)],dtype=torch.float32)
        x_out1X[0,:]=1
        

        for time_index in range(1,Rvelocities.shape[0]):
                x_out1X[time_index,:] = np.exp(-s*Rvelocities[time_index-1])*x_out1X[time_index-1,:]# + f*self.dt


        decays=x_out1X
        inputData=torch.tensor(np.stack([loc[:,0],loc[:,1],np.insert(Rvelocities,[self.n_steps-1],0),absAngles]),dtype=torch.float32).T
        # breakpoint()
        if figureOn:
            

            '''
            plt.ioff()
            plt.figure()
            ax = plt.axes()
            placeFieldsX=(decays[:,1:100]-decays[:,0:99]).T
            sns.heatmap(placeFieldsX)
            ax.set_title(['upper= ' +str(upper) +' lower= ' +str(lower) +'PositvXdirection'])
            plt.savefig('upper= ' +str(upper) +' lower= ' +str(lower)+'.png')
            # '''
            
            # '''
            plt.ion()
            fig=plt.figure(figsize=(15,15))
            ax=fig.add_subplot(111)
            ax.set_title("Agent's Trajectory",fontsize=2)
            plt.xlabel('X',fontsize=2)
            plt.ylabel('Y',fontsize=2)
            # ax.set_xticks([0,500,1000,1500])
            # ax.set_xticklabels(['0', '50', '100','150'])
            ax.tick_params(axis="x", labelsize=14)
            ax.tick_params(axis="y", labelsize=14)
            ax.plot(loc[:,0], loc[:,1],'b')
            ax.set_aspect('equal')
            wedge = Wedge((0.5, 0.5), 1.5, 0, 360, width=1.2, transform=ax.transAxes, facecolor='white', edgecolor='none')
            ax.add_patch(wedge)
            ax.set_aspect('equal')
            ax.set_position([0, 0, 1, 1])
            plt.axis('off')
            # ax.plot(loc2[:,0], loc2[:,1],'.y')
            ax.set_xlim(-1000,1000)
            ax.set_ylim(-1000,1000)
            # ax.plot(np.asarray(np.cos(absAngles[1:])*Rs[:-1]), np.asarray(np.sin(absAngles[1:])*Rs[:-1]),'b')
            # ax.plot(np.asarray(np.cos(polarCoordinateAngle)*polarCoordinateR), np.asarray(np.sin(polarCoordinateAngle)*polarCoordinateR),'g')
           
            # '''
            
            # plt.figure()
            # ax = plt.axes()
            # placeFieldsX=(x_out1XNegative[:,1:99]-x_out1XNegative[:,0:98]).T
            # sns.heatmap(placeFieldsX)
            # ax.set_title(['upper= ' +str(upper) +' lower= ' +str(lower) +'NegtvXdirection'])
            # plt.savefig('upper= ' +str(upper) +' lower= ' +str(lower)+'.png')
            # # placeFieldsY=(x_out1XNegative[:,1:99]-x_out1XNegative[:,0:98]).T
            # placeFieldsMaxesX=torch.max(placeFieldsX,dim=1)[0]
            # for ff in range(placeFieldsX.shape[0]):
            #     placeFieldsX[ff,:]=placeFieldsX[ff,:]/placeFieldsMaxesX[ff]
            # placeFieldsMaxesY=torch.max(placeFieldsY,dim=1)[0]
            # for ff in range(placeFieldsY.shape[0]):
            #         placeFieldsY[ff,:]=placeFieldsY[ff,:]/placeFieldsMaxesY[ff]
            # plt.figure()
            # # placeFields2D=placeFieldsY,placeFieldsX
            # plt.plot(linspace(0,1,100),placeFieldsX[0,:],'. plt.ion()
                # fig=plt.figure(figsize=(15,15))
                # ax=fig.add_subplot(111)
                # ax.set_title("Agent's Trajectory",fontsize=2)
                # plt.xlabel('X',fontsize=2)
                # plt.ylabel('Y',fontsize=2)
                # # ax.set_xticks([0,500,1000,1500])
                # # ax.set_xticklabels(['0', '50', '100','150'])
                # ax.tick_params(axis="x", labelsize=14)
                # ax.tick_params(axis="y", labelsize=14)
                # ax.plot(loc[:,0], loc[:,1],'.b')
                # # ax.plot(loc2[:,0], loc2[:,1],'.y')
                # ax.set_xlim(-820,820)
                # ax.set_ylim(-820,820)
            # plt.plot(placeFieldsY[0,:],linspace(0,1,100),'.r')
            # sns.heatmap(placeFields2D, ax = ax)


            # plt.figure()
            # ax = plt.axes()
            # # plt.plot(x_out1XPositive)
            # plt.plot(x_out1X)
            # ax.set_title(['upper= ' +str(upper) +' lower= ' +str(lower) +'PositvXdirection'])
            # plt.savefig('upper= ' +str(upper) +' lower= ' +str(lower)+'Decay.png')
            
            
            
            '''
            plt.figure()
            plt.plot(decays)
            plt.figure()
            sns.heatmap(decays)
            '''
            # fig=plt.figure(figsize=(15,15))
            # ax=fig.add_subplot(111)
            # ax.set_title("Trajectory of The Agent")
            # col=np.arange(len(loc))
            # # xAxis=np.ones(len(loc))
            # ax.scatter(loc[:,0],loc[:,1],c=col)
            # ax.plot(loc[:,0],loc[:,1])
            # ax.set_xlim(-0.1,10)
            # ax.set_ylim(0.95,1.05)
            # plt.show()

        
        
        #direction selective  
        # decays=torch.tensor(np.concatenate([x_out1XPositive[:self.n_steps-1-self.firstTimeSteps2Remove,:], x_out1YPositive[:self.n_steps-1-self.firstTimeSteps2Remove,:],x_out1XNegative[:self.n_steps-1-self.firstTimeSteps2Remove,:],x_out1YNegative[:self.n_steps-1-self.firstTimeSteps2Remove,:]],axis=1),dtype=torch.float32)
        #direction agnostic   
        
        # breakpoint()
        # if sample=='uniformDistribution':
        #     Indcs=np.round(np.random.randint(0,len(locAll)-1,8000),0).astype('int')
        #     inputData=inputDataRaw[Indcs,:]
        #     allRangeXout1=allRangeXout1[Indcs,:]
        # elif sample == 'normalDistribution':
        #     Indcs=np.round(np.random.normal(8000,SD,1000),0).astype('int')
        #     while Indcs.max()>len(inputDataRaw)-1:
        #         Indcs=np.round(np.random.normal(8000,SD,1000),0).astype('int')
        #     inputData=inputDataRaw[Indcs,:]
        #     allRangeXout1=allRangeXout1[Indcs,:]
        # targetLocs=torch.tensor(targe'tFiring,dtype=torch.float32)       
        return [inputData,decays]
    
 #%% 
# angleVec=np.linspace(6, 360,2)
   
maxOfSteps=8000
testtrack=RatOnPlanePolar(maxOfSteps)
h=[]
numOfSamples=5
# for angle in tqdm(angleVec): 
for hh in tqdm(range(numOfSamples)):
    result=testtrack.generateTrajectory(figureOn=True,upper=1e-1,lower=1e-3)
    h.append( result)
    # CleanTrainSet=torch.stack(h[0][:60000])   
  
    # torch.save(CleanTrainSet.transpose(1,2), 'trainingSet.pt')
    # collector=[]
    # for ii in range(numOfSamples):
    #     tmp=h[ii][1]
    #     collector.append(tmp)
    
    # collector=np.vstack(collector)
    # collector=(collector-np.mean(collector))/np.std(collector)
    # # collector.mean()
    # # collector.std()
    # for ii in range(numOfSamples):
    #     h[ii][1]=collector[ii*5000:(ii*5000)+5000,:]
    
    
torch.save(h, 'TrainingSetPolar2DNoModRawLogSpacedCIRCULAR.pt') 
# dummyList=torch.tensor(np.linspace(6, 360,60),dtype=torch.float32)
# torch.save(dummyList, 'dummyList.pt')
####################3

