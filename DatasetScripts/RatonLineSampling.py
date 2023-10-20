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
os.chdir('DIRECTORY OF YOUR CODES')
from RatSimulatorForBackup import RatSimulator
os.chdir('DIRECTORY WHERE YOU WANT YOUR DATASETS TO BE SAVED')
import numpy as np
import matplotlib.pyplot as plt
import torch
from tqdm import tqdm 
import seaborn as sns
from nice_figures import *
load_style('APS', '2-column')
np.random.seed(1234)
class RatOnPlaneLaplace():
    def __init__(self, n_steps,datapointPerTrial=1000,GoalLocation=False):
        self.n_steps=n_steps
        # self.dt=0.01
        self.end=np.array([1,1])
        self.start=0
        self.rat=RatSimulator(self.n_steps)
        self.firstTimeSteps2Remove=10
        self.datapointPerTrial=datapointPerTrial
        
    def generateTrajectory(self,figureOn=False,CA1Size=100,hiddenSizeMEC=100,upper=0,lower=-1,sample='uniformDistribution',SD=2000,linearlySpaced=True):
        loc=np.full((self.n_steps),np.nan)
        loc[0]=np.array([0])
        # breakpoint()
        # scalingFactor=1#np.abs(np.random.normal(0,5))
        # loc[0,:]=np.random.uniform(0,10,size=[1,2])
        # velocityX=np.random.normal(0,5000,size=[self.n_steps,1])
        # velocityY=np.random.normal(0,5000,size=[self.n_steps,1])
        # velocityXNeg=-np.abs(np.random.normal(0,50,size=[self.n_steps,1]))
        # velocityYNeg=-np.abs(np.random.normal(0,50,size=[self.n_steps,1]))
        # velocity=np.abs(np.random.normal(0,100,size=[self.n_steps,1]))#*scalingFactor
        velocity=np.ones(self.n_steps)#*scalingFactor
        # if (np.random.randint(0, 2, size=1, dtype=int).any()==1): #initial movement directiom
            # print('ee')
            # velocity=-velocity
        # velocityY=np.abs(np.random.normal(size=[self.n_steps,1]))
        
        # breakpoint()
        # directionX=np.full((self.n_steps-1,1),np.nan)
        # directionY=np.full((self.n_steps-1,1),np.nan)
        # noOfSwitchPoints=np.random.randint(0, 1000, size=1, dtype=int)
        # switchPoints=np.random.randint(0, high=self.n_steps, size=noOfSwitchPoints, dtype=int)
        # switchPoints=np.sort(switchPoints)
        
        # for kk in range (0,len(switchPoints)):
 
            
            # if kk>0 and kk%2 : 
                # velocity[switchPoints[kk-1]:switchPoints[kk]]=-velocity[switchPoints[kk-1]:switchPoints[kk]]
                
       
                
                

                
        for i in range (0,self.n_steps-1):
            loc[i+1]=loc[i]+velocity[i]
        # locNeg=-loc
        # breakpoint()
        # directionX=directionX[~np.isnan(directionX)]
        # directionY=directionY[~np.isnan(directionY)]
        # velocityX=directionX[~np.isnan(directionX)]
        # velocityY=directionY[~np.isnan(directionY)]
        # loc=loc[~np.isnan(loc[:,1])]
        # loc=loc[:-1,:]
        # locX=loc[:,0]
        # locY=loc[:,1]
        # loc=loc[~np.isnan(loc)]

        # if loc[-1]>self.n_steps:
            # return [torch.Tensor([[1000,1000],[1000,1000]]),['tooLong']]
        # elif loc[-1]<=100:  
        #     timeSig=torch.zeros([100,100])
        #     for ss in range(10):
        #         timeSig[ss*10:(ss+1)*10,ss*10:(ss+1)*10]=1
        

        # [Rvelocities, angVel, loc, angle]=self.rat.generateTrajectory()
        # loc=loc[self.firstTimeSteps2Remove:,:]
        # cartesianVelocity=np.diff(loc,axis=0)*100
        # velocityX=cartesianVelocity[:,0]
        # velocityY=cartesianVelocity[:,1]
        if linearlySpaced:
            s=torch.linspace(upper,lower,hiddenSizeMEC)#
        else:
            s=torch.logspace(np.log10(lower),np.log10(upper),hiddenSizeMEC)#
        
        
        ############## direction sensative ##########################
        # x_out1XPositive=torch.zeros([self.n_steps,hiddenSizeMEC])
        # x_out1XPositive[0,:]=1
        # x_out1XNegative=torch.zeros([self.n_steps,hiddenSizeMEC])
        # x_out1XNegative[0,:]=1
        # for time_index in range(velocityX.shape[0]):
        #     if time_index>0 and velocityX[time_index-1]>0:  

        #         # Exponential integrator 
        #         x_out1XPositive[time_index,:] = np.exp(-s*velocityX[time_index-1])*x_out1XPositive[time_index-1,:]# + f*self.dt
        #         x_out1XNegative[time_index,:] = x_out1XNegative[time_index-1,:]# + f*self.dt
        #     if time_index>0 and velocityX[time_index-1]<0:  
        #             # Exponential integrator 
        #         x_out1XNegative[time_index,:] = np.exp(-s*-velocityX[time_index-1])*x_out1XNegative[time_index-1,:]# + f*self.dt
        #         x_out1XPositive[time_index,:] = x_out1XPositive[time_index-1,:]# + f*self.dt
              
                
        # x_out1YPositive=torch.zeros([self.n_steps,hiddenSizeMEC])
        # x_out1YPositive[0,:]=1
        # x_out1YNegative=torch.zeros([self.n_steps,hiddenSizeMEC])
        # x_out1YNegative[0,:]=1
        # for time_index in range(velocityY.shape[0]):
        #     if time_index>0 and velocityY[time_index-1]>0:      
        #         # Exponential integrator 
        #         x_out1YPositive[time_index,:] = np.exp(-s*velocityY[time_index-1])*x_out1YPositive[time_index-1,:]# + f*self.dt
        #         x_out1YNegative[time_index,:] =x_out1YNegative[time_index-1,:]# + f*self.dt

        #     if time_index>0 and velocityY[time_index-1]<0: 
        #         x_out1YNegative[time_index,:] = np.exp(-s*-velocityY[time_index-1])*x_out1YNegative[time_index-1,:]# + f*self.dt
        #         x_out1YPositive[time_index,:] = x_out1YPositive[time_index-1,:]# + f*self.dt

          ############## direction agnostic ##########################
        x_out1X=torch.zeros([velocity.shape[0],hiddenSizeMEC],dtype=torch.float32)
        x_out1X[0,:]=1
       
        for time_index in range(1,velocity.shape[0]):
                #exponential decay
                # x_out1X[time_index,:] = np.exp(-s*velocity[time_index-1])*x_out1X[time_index-1,:]# + f*self.dt
                
                # just addition
                x_out1X[time_index,:] =s*velocity[time_index-1]+x_out1X[time_index-1,:]# + f*self.dt

                
                # clamping the values at 1 and keeping abobe 1 from exploding:
        # x_out1XNegative=torch.zeros([velocity.shape[0],hiddenSizeMEC],dtype=torch.float32)
        # x_out1XNegative[0,:]=1

        # for time_index in range(1,velocity.shape[0]):
                # x_out1XNegative[time_index,:] = np.exp(-s*-velocity[time_index-1])*x_out1XNegative[time_index-1,:]# + f*self.dt
                # clamping the values at 1 and keeping abobe 1 from exploding:
        
        # allRangeXout1=np.vstack([np.flipud(x_out1XNegative),x_out1X])
        
                # if any(x_out1X[time_index,:]> 1.0):
                #     print (time_index)
                #     x_out1X[ time_index, x_out1X[time_index,:]>1]
                    
                #     tmp2=torch.sqrt(tmp1>1)
                #     x_out1X[time_index,:][x_out1X[time_index,:]> 1.0]=tmp2
                    
                #     x_out1X[time_index,:]> 1.0 = np.sqrt(x_out1X[time_index,:]> 1.0)
                # elif any(x_out1X[time_index,:]> 2.0):
                #     np.sqrt(x_out1X[time_index,:]> 1.0)
                #     x_out1X[time_index,:] = np.exp(-s*velocity[time_index-1])*x_out1X[time_index-1,:]# + f*self.dt

                #     print (time_index)
        
              
                
        # x_out1Y=torch.zeros([velocityY.shape[0],hiddenSizeMEC])
        # x_out1Y[0,:]=1
        # for time_index in range(velocityY.shape[0]):
            # if time_index>0:
                # Exponential integrator 
                # x_out1Y[time_index,:] = np.exp(-s*velocityY[time_index-1])*x_out1Y[time_index-1,:]# + f*self.dt
               
        # breakpoint()  
        if figureOn:
            # plt.figure()
            # plt.plot(x_out1)
            # plt.title(['upper= ' +str(upper) +' lower= ' +str(lower)])
            
            
            # plt.ioff()
            # plt.figure()
            # ax = plt.axes()
            # placeFieldsX=(x_out1XPositive[:,1:99]-x_out1XPositive[:,0:98]).T
            # sns.heatmap(placeFieldsX)
            # ax.set_title(['upper= ' +str(upper) +' lower= ' +str(lower) +'PositvXdirection'])
            # plt.savefig('upper= ' +str(upper) +' lower= ' +str(lower)+'.png')
            
            # # location vs decay
            # # plt.ion()
            # plt.ioff()
            # plt.figure()
            # ax = plt.axes()
            # plt.plot(loc,x_out1X)
            # ax.set_title(['upper= ' +str(upper) +' lower= ' +str(lower) +'Location&Decay'])
            # plt.savefig('upper= ' +str(upper) +' lower= ' +str(lower)+'LocAndDecay.png')
            
            plt.ion()
            # plt.ioff()
            plt.figure()
            ax = plt.axes()
            # plt.plot(torch.cat([torch.ones(1,10),x_out1X[:10,:]]))
            plt.plot(x_out1X)
            plt.xlabel('Location',fontsize=4)
            plt.ylabel('Leaky Integrator Value',fontsize=4)
            ax.set_title('Decay vs. Time'+' \nFastest  decay constant= ' +str(upper) +' Slowest decay constant= ' +str(lower) )
            ax.tick_params(axis="x", labelsize=14)
            ax.tick_params(axis="y", labelsize=14)
            ax.set_xticks([0,500,1000,1500])
            ax.set_xticklabels(['0', '50', '100','150'])
            plt.show()
            plt.savefig('upper= ' +str(upper) +' lower= ' +str(lower)+'Decay.png')
            
            
            ##########################
            #velocity
            velocitySampleVec=np.diff(torch.cat([torch.tensor([0,0]),inputData[:10,0]]))
            # velocitySampleVec=np.diff(inputData[:10,0])
            # a=torch.ones(10)*velocitySampleVec[0]
            b=torch.ones(10)*velocitySampleVec[1]
            c=torch.ones(10)*velocitySampleVec[2]
            d=torch.ones(10)*velocitySampleVec[3]
            e=torch.ones(10)*velocitySampleVec[4]
            f=torch.ones(10)*velocitySampleVec[5]
            g=torch.ones(10)*velocitySampleVec[6]
            h=torch.ones(10)*velocitySampleVec[7]
            i=torch.ones(10)*velocitySampleVec[8]
            j=torch.ones(10)*velocitySampleVec[9]
            k=torch.ones(10)*velocitySampleVec[10]

            vec4Plot=torch.cat([b,c,d,e,f,g,h,i,j,k])
            plt.ion()
            # plt.ioff()
            plt.figure()
            ax = plt.axes()
            plt.plot(vec4Plot,'k')
            plt.xlabel('Location',fontsize=4)
            plt.ylabel('Leaky Integrator Value',fontsize=4)
            ax.set_title('Decay vs. Time'+' \nFastest  decay constant= ' +str(upper) +' Slowest decay constant= ' +str(lower) )
            
            ax.set_xticks([0,20,40,60,80])
            ax.set_xticklabels(['0', '2', '4','6','8'])
            ax.tick_params(axis="x", labelsize=14)
            ax.tick_params(axis="y", labelsize=14)
            plt.show()
      ##################
            
            
            plt.ioff()
            plt.figure()
            ax = plt.axes()
            placeFieldsX=(x_out1X[:,1:99]-x_out1X[:,0:98]).T
            sns.heatmap(placeFieldsX)
            ax.set_title(['upper= ' +str(upper) +' lower= ' +str(lower) +'PositvXdirection'])
            plt.savefig('upper= ' +str(upper) +' lower= ' +str(lower)+'.png')
            
            
            
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
            # plt.plot(linspace(0,1,100),placeFieldsX[0,:],'.k')
            # plt.plot(placeFieldsY[0,:],linspace(0,1,100),'.r')
            # sns.heatmap(placeFields2D, ax = ax)


            # plt.figure()
            # ax = plt.axes()
            # # plt.plot(x_out1XPositive)
            # plt.plot(x_out1X)
            # ax.set_title(['upper= ' +str(upper) +' lower= ' +str(lower) +'PositvXdirection'])
            # plt.savefig('upper= ' +str(upper) +' lower= ' +str(lower)+'Decay.png')

            # plt.figure()
            # sns.heatmap(x_out1)
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
        # tmp=np.fliplr([locNeg,locNeg])
        # locAll=np.hstack([tmp[0],loc])
        # velocity=np.hstack([velocity,velocity])
        # breakpoint()
        inputDataRaw=torch.tensor(np.stack([loc,velocity]),dtype=torch.float32).T
        #direction selective  
        # decays=torch.tensor(np.concatenate([x_out1XPositive[:self.n_steps-1-self.firstTimeSteps2Remove,:], x_out1YPositive[:self.n_steps-1-self.firstTimeSteps2Remove,:],x_out1XNegative[:self.n_steps-1-self.firstTimeSteps2Remove,:],x_out1YNegative[:self.n_steps-1-self.firstTimeSteps2Remove,:]],axis=1),dtype=torch.float32)
        #direction agnostic   
        
        # breakpoint()
        if sample=='uniformDistribution':
            Indcs=np.round(np.random.randint(0,len(loc)-1,self.datapointPerTrial),0).astype('int')#should be changed to 1000
            inputData=inputDataRaw[Indcs,:]
            x_out1X=x_out1X[Indcs,:]
        elif sample == 'normalDistribution':
            Indcs=np.abs(np.round(np.random.normal(0,SD,self.datapointPerTrial),0).astype('int'))
            acceptableIndcs=Indcs[Indcs<len(inputDataRaw)-1]
            if Indcs.max()>=len(inputDataRaw)-1:
                # breakpoint()
                while len(acceptableIndcs)<len(inputDataRaw):
                    Indcs=np.abs(np.round(np.random.normal(0,SD,self.datapointPerTrial),0).astype('int'))
                    acceptableIndcs=np.hstack([acceptableIndcs,Indcs[Indcs<len(inputDataRaw)-1]])
            # breakpoint()
            acceptableIndcs=acceptableIndcs[0:self.datapointPerTrial]
            inputData=inputDataRaw[acceptableIndcs,:]
            x_out1X=x_out1X[acceptableIndcs,:]
        # targetLocs=torch.tensor(targe'tFiring,dtype=torch.float32)       
        return [inputData,x_out1X]
    
 #%% 
# FOR TRYING DIFFERENT STDs:
#sdVec=[100,250,500,750,1000,1500]
sdVec=[1500]#,1500]
upper=[1e-1]#,1e-2
lower=[1e-3]#,1e-4,1e-5]
for SD in sdVec:    
    for upperVal in upper:
        for lowerVal in lower:
            maxOfSteps=1800
            testtrack=RatOnPlaneLaplace(maxOfSteps)
            h=[]
            numOfSamples=12000
            for hh in tqdm(range(numOfSamples)):
                result=testtrack.generateTrajectory(figureOn=False,upper=upperVal,lower=lowerVal,sample='uniformDistribution',SD=SD,linearlySpaced=False)
                # if result[0][0,0]!= torch.Tensor([1000]):
                h.append( result)
            torch.save(h, 'TrainingSetuniformDistributionUpperVal'+str(upperVal)+'lowerVal'+str(lowerVal)+'SimpleAdditionLog.pt') 
        # torch.save(h, 'TrainingSetNormalSampleSD'+str(SD)+'.pt') 
    # locations=[]
    # for i in range(len(h)):
    #     locations.append(h[i][0][:,0].T)

    # ##################################
    # locations2=np.hstack(locations)

    # plt.figure()
    # ax = plt.axes(),plt.hist(locations2,50)
    
    # plt.title('Hist of Locations (avg. 100 trials), SD= ' +str(SD) )
    # plt.xlabel('Location')
    
    # plt.savefig('Standard Deviation= ' +str(SD) )
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
    
sdVec=[1500]#,1500]
upper=[1e-1]#,1e-2]
lower=[1e-3]#,1e-4,1e-5]
for SD in sdVec:    
    for upperVal in upper:
        for lowerVal in lower:
            maxOfSteps=3000
            testtrack=RatOnPlaneLaplace(maxOfSteps)
            h=[]
            numOfSamples=100
            for hh in tqdm(range(numOfSamples)):
                result=testtrack.generateTrajectory(figureOn=False,upper=upperVal,lower=lowerVal,sample='uniformDistribution',SD=SD,linearlySpaced=True)
                # if result[0][0,0]!= torch.Tensor([1000]):
                h.append( result)
            torch.save(h, 'TestSetuniformDistributionUpperVal'+str(upperVal)+'lowerVal'+str(lowerVal)+'SimpleAdditionLinSpaced3000Length.pt') 
     
####