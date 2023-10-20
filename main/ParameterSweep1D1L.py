
"""
Created on Wed Oct 19 11:24:01 2022

@author: reza alipour
"""


import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader,random_split
from torchvision import transforms, utils
import torchmetrics
import seaborn as sns
import pytorch_lightning as pl
from torch import nn
import torch
import torch.nn.functional as F
from argparse import ArgumentParser
import plotly.express as px
from nice_figures import *
load_style('APS', '1-column')
from pytorch_lightning.loggers import WandbLogger
pl.seed_everything(hash("setting random seeds") % 2**32 - 1)

import wandb
import plotly.graph_objects as go# ⚡ 🤝 🏋️‍♀️
import torch.nn.utils.parametrize as parametrize
import os
os.chdir('/media/panlab/Thesis/DeepCodes')
from resultAnalyzer import *
import gc
import random
from pytorch_lightning.callbacks import ModelCheckpoint
import plotly.io as pio
import plotly.express as px
# get the computation device
def get_device():
    if torch.cuda.is_available():
        device = 'cuda:0'
    else:
        device = 'cpu'
    return device
device = get_device()

def kl_divergence(rho, rho_hat):
   
    # if you average over dim=1, you constrain the sparsity of the layer, if you average over 0, you constrain each neuron.
    rho_hat =   torch.mean(torch.sigmoid(rho_hat), 1) # sigmoid because we need the probability distributions
    rho = torch.tensor([rho] * len(rho_hat)).to(device)
    return torch.sum(rho * torch.log(rho/rho_hat) + (1 - rho) * torch.log((1 - rho)/(1 - rho_hat)))



class TrajectoryDataset(Dataset):
    

    def __init__(self, torchFile,transform=None,num_workers=32):
        """
        Args:
            csv_file (string): Path to the npy file with Trajectories.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.TrajectorySamples = torch.load(torchFile)

        self.transform = transforms.Normalize(0, 1,inplace=True)

    def __len__(self):
        return len(self.TrajectorySamples)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()


        
        sample = self.TrajectorySamples[idx]

       
        return sample






class sharedWeightLayer(nn.Module):

    def __init__(self,noOfOffDiagValues=10,upDiag=0.001,Diag=0.0,lowDiag=-0.001,hiddenSizeMEC=100,CA1Size=100):
        super(sharedWeightLayer, self).__init__()
        self.noOfOffDiagValues=noOfOffDiagValues
        self.OffDiagonalValues=nn.Parameter(torch.tensor(np.random.normal(0,1,self.noOfOffDiagValues*2),dtype=torch.float64),requires_grad=True)
        
        self.Diag=nn.Parameter(torch.tensor(np.random.normal(0,1,1),dtype=torch.float64),requires_grad=True)
        
        buff_len = 25
        k = 8
        Taustar_min = 1
        Taustar_max = 10 
       

        self.tstr_min = Taustar_min
        self.tstr_max = Taustar_max
        self.buff_len = buff_len
        self.k = k
        
        # Create power-law growing Taustarlist and compute corresponding s
        self._a = (self.tstr_max/self.tstr_min)**(1./buff_len)-1
        pow_vec = np.arange(-self.k,buff_len + self.k) #-1
        self.Taustarlist = self.tstr_min * (1 + self._a)**pow_vec
        s = self.k/self.Taustarlist
       
        s=torch.logspace(100,20000,hiddenSizeMEC)#torch.logspace(upper,lower,hiddenSizeMEC)#
        

        self._DerivMatrix = np.zeros((hiddenSizeMEC,CA1Size))
        for i in range(1,hiddenSizeMEC-1):
            self._DerivMatrix[i, i-1] = -(s[i+1]-s[i])/(s[i]-s[i-1])/(s[i+1] - s[i-1])
            self._DerivMatrix[i, i] = ((s[i+1]-s[i])/(s[i]- s[i-1])-(s[i]-s[i-1])/(s[i+1]-s[i]))/(s[i+1] - s[i-1])
            self._DerivMatrix[i, i+1] = (s[i]-s[i-1])/(s[i+1]-s[i])/(s[i+1] - s[i-1])
        self._DerivMatrix =nn.Parameter(torch.tensor(self._DerivMatrix,dtype=torch.float32),requires_grad=False)
        
    def forward(self, x):
       
        x=x-torch.tril(x,-self.noOfOffDiagValues)
        x=x-torch.triu(x,+self.noOfOffDiagValues)
        # x[x==x.diagonal(0)]=self.Diag
        
        
        for i in range(x.shape[0]):
            for j in range(x.shape[1]):
                for DiagNo in range (1,self.noOfOffDiagValues):
                    if i==j-DiagNo:
                        x[i,j]=self.OffDiagonalValues[DiagNo]
                    elif i==j+DiagNo:
                        x[i,j]=self.OffDiagonalValues[DiagNo+self.noOfOffDiagValues]
                    elif i==j:
                        x[i,j]=self.Diag
                
          
            
        # x=self._DerivMatrix
        # a=np.asarray(x.detach().cpu())
        # plt.plot(),sns.heatmap(a)
        return x







learningRate=1e-4
CA1Size=50#,100,200]
activityPunishment=0#1e-8#,1e-9]#[1e-7,1e-8,1e-9]
Dropout=0.25#,.5]#[.1,.25,.5,.75]
timeStepsToPredict=1
weight_decayFactor=0#.3
rho=100
KLmultiplier=50
                        
os.chdir('/media/panlab/Thesis/DeepCodes')
pl.seed_everything(hash("setting random seeds") % 2**32 - 1)



class EC_HPC(pl.LightningModule):
    def __init__(self,input_size=1,
                  batch_size=128,rho=rho,lowerVal=100,upperVal=1,activityPunishment=activityPunishment,KLmultiplier=KLmultiplier,learningRate=learningRate,hiddenSizeMEC=100,CA1Size=CA1Size,Dropout=Dropout,weight_decayFactor=weight_decayFactor ,num_workers =32):
        super().__init__()
       
        self.KLmultiplier=KLmultiplier
        self.activityLogs=[]
        self.activityLogsVal=[]
        self.save_hyperparameters()
        self.CA1L2Size=CA1Size
        self.lowerVal=lowerVal
        self.upperVal=upperVal

        self.RHO=rho

        self.CA1L2=        nn.Sequential(
                     nn.Linear(hiddenSizeMEC, out_features=CA1Size,bias=False),
                       nn.ReLU()) 
              

        
        
       
        #implementing local weights:
        parametrize.register_parametrization(self.CA1L2[0], "weight", sharedWeightLayer())   
        
        
        
        
       
        self.BooleanDropOut=True
        self.droupoutValue=Dropout
        self.dropout = nn.Dropout(self.droupoutValue)
        
       
        self.CA1ReadoutNode=nn.Sequential(
            nn.Linear(CA1Size, out_features=hiddenSizeMEC,bias=False))
        
        self.weight_decayFactor=weight_decayFactor    
        self.activityPunishment=True
        self.activityPunishmentWeight=activityPunishment
        self.Timesteps2Presidc=timeStepsToPredict
        

        self.save_hyperparameters()
        self.batch_size = batch_size
    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=learningRate,weight_decay=self.weight_decayFactor)#

    def setup(self, stage=None,num_workers =32):
    
        # we set up only relevant datasets when stage is specified
        if stage == 'fit' or stage is None:
            # breakpoint()
            # trajDSet=TrajectoryDataset('TrainingSetuniformDistribution'+'.pt')#TrainingSetBoundedAndCut
            # if doing log spaced
            trajDSet=TrajectoryDataset('TrainingSetuniformDistributionUpperValLOG'+str(self.upperVal)+'lowerVal'+str(self.lowerVal)+'.pt')#TrainingSetBoundedAndCut
            
            
            #simple Addition (control experiment)
            #linspaced
            # trajDSet=TrajectoryDataset('TrainingSetuniformDistributionUpperVal0.1lowerVal0.001SimpleAdditionLinear.pt')#TrainingSetBoundedAndCut
             #log spaced
            # trajDSet=TrajectoryDataset('TrainingSetuniformDistributionUpperVal0.1lowerVal0.001SimpleAdditionLog.pt')#TrainingSetBoundedAndCut
            self.trajDSet_train, self.trajDSet_val = random_split(trajDSet, [11500, 500]) #4586, 400 for bounded cut 3151 2554, 500, for 1440:3677
        if stage == 'test' or stage is None:
            # self.trajDSet_test=TrajectoryDataset('TrainingSetuniformDistribution.pt')
           # if doing log spaced
            # self.trajDSet_test=TrajectoryDataset('TestSetuniformDistributionUpperValLOG'+str(self.upperVal)+'lowerVal'+str(self.lowerVal)+'.pt')
            # if doing log spaced and longer track:
            self.trajDSet_test=TrajectoryDataset( 'TestSetuniformDistributionUpperVal0.1lowerVal0.001LogSpacedAnd3000Length.pt')
            
            
            #simple Addition
            #linspaced
            # self.trajDSet_test=TrajectoryDataset( 'TestSetuniformDistributionUpperVal0.1lowerVal0.001SimpleAdditionLinSpaced3000Length.pt')
            #log spaced
            # self.trajDSet_test=TrajectoryDataset( 'TestSetuniformDistributionUpperVal0.1lowerVal0.001SimpleAdditionLogSpaced3000Length.pt')
            
    # we define a separate DataLoader for each of train/val/test
    def train_dataloader(self,num_workers =32):
        trajDSet_train = DataLoader(self.trajDSet_train, batch_size=self.batch_size)
        return trajDSet_train
    
    def val_dataloader(self,num_workers =32):
        trajDSet_val = DataLoader(self.trajDSet_val, batch_size=1 * self.batch_size)
        return trajDSet_val
    
    def test_dataloader(self,num_workers =32):
        trajDSet_test = DataLoader(self.trajDSet_test, batch_size=1 * self.batch_size)
        return trajDSet_test
        
    
    def training_step(self, batch, batch_idx):
        # breakpoint()
        if self.BooleanDropOut:
            x_out1=self.dropout(torch.flatten(batch[1] , end_dim=1) )
        else:
            x_out1=torch.flatten(batch[1] , end_dim=1)
        targets=torch.flatten(batch[1] , end_dim=1)       # x_out1=self.dropout(batch[0][:,:,1])

        PlaceCells1=self.CA1L2(x_out1)
        

   
        preds=self.CA1ReadoutNode(PlaceCells1)
        preds=preds.squeeze(-1)
        loss=nn.MSELoss()
               
        if self.activityPunishment:
            activityPunishmentLoss=(self.activityPunishmentWeight*(torch.sum(PlaceCells1**2)))
            MSELoss=loss(preds,targets)
            kl_loss=self.KLmultiplier*kl_divergence(self.RHO, PlaceCells1)
            distance_loss=MSELoss+activityPunishmentLoss+kl_loss
            
           
        self.log("training Total/loss_epoch",distance_loss, on_step=False, on_epoch=True)  # d
        self.log("training KL/loss_epoch",kl_loss, on_step=False, on_epoch=True)  # d
        self.log("training MSE/loss_epoch",MSELoss, on_step=False, on_epoch=True)  # d
        return distance_loss

    def validation_step(self, batch, batch_idx):
   
        if self.BooleanDropOut:
            x_out1=self.dropout(torch.flatten(batch[1] , end_dim=1))
        else:
            x_out1=torch.flatten(batch[1] , end_dim=1)
        targets=torch.flatten(batch[1] , end_dim=1)    
        actualLocation=batch[0][:,:]

        
        PlaceCells1=self.CA1L2(x_out1)
       
        preds=self.CA1ReadoutNode(PlaceCells1)
        preds1=self.CA1ReadoutNode(PlaceCells1)
        preds=preds.squeeze(-1)
        loss=nn.MSELoss()

        if self.activityPunishment:
            activityPunishmentLoss=(self.activityPunishmentWeight*(torch.sum(PlaceCells1**2)))
            MSELoss=loss(preds,targets)
            kl_loss=self.KLmultiplier*kl_divergence(self.RHO, PlaceCells1)
            distance_loss=MSELoss+activityPunishmentLoss+kl_loss
            
             
         
            # logging
        self.log("validation Total/loss_epoch",distance_loss, on_step=False, on_epoch=True)  # d
        self.log("validation KL/loss_epoch",kl_loss, on_step=False, on_epoch=True)  # d
        self.log("validation MSE/loss_epoch",MSELoss, on_step=False, on_epoch=True)  # d 
        
        OriginalDecays=((batch[1])*1).to(device='cpu').detach()
        predictedDecays=(preds1*1).to(device='cpu').detach()
        predictedDecays=torch.reshape(predictedDecays,(batch[1].shape))
        PlaceCellsL1=(PlaceCells1*1).to(device='cpu').detach()
        PlaceCellsL1=torch.reshape(PlaceCellsL1,(batch[1].shape[0],batch[1].shape[1],self.CA1L2Size))

        
        self.activityLogsVal.append([actualLocation,PlaceCellsL1,OriginalDecays[0,:,0],predictedDecays[0,:,0],distance_loss.detach()])

         
        return batch
    def on_validation_epoch_end(self):
        # breakpoint()
        [LocHistArraySortedMax,TimeHistArraySortedMax,LocHistArraySortedFirts2Fire,TimeHistArraySortedFirts2Fire,Time2LocMatchedHistArraySortedMax,LocHistArraySortedMaxlog10] =plotPlaceFields(self.activityLogsVal,False,2,maxLocationPossible=1800)
        self.activityLogsVal=[]

        fig = go.Figure(data=go.Heatmap(
                    z=np.flipud(LocHistArraySortedMax),
                    x=np.linspace(0,180,181),
                    type = 'heatmap',
                    colorscale = 'inferno'))
        fig.show()
        wandb.log({"AllCells Val": fig})
        
        fig = go.Figure(data=go.Heatmap(
                    z=np.flipud(LocHistArraySortedMax),
                    # x=np.logspace(np.log(.001), np.log(180), 181),
                    type = 'heatmap',
                    colorscale = 'inferno'))
        
        fig.update_xaxes(type="log")
        # fig.show()
        wandb.log({"AllCellsLog Val": fig})
        # plt.close('all')
        placeCells=0
        
        placeCellIndex=[]
        for ll in range(LocHistArraySortedMax.shape[0]): #72= 80/90, we only count neurons that fire in 20% of bins between 5-95 !
                if (np.sum(LocHistArraySortedMax[int(ll),:]>.5)<72) & (np.sum(LocHistArraySortedMax[int(ll),:5]>.9)<5)& (np.sum(LocHistArraySortedMax[int(ll),-5:]>.9)<5):
                    # if : 72
                        # if (np.sum(LocHistArraySortedMax[int(ll),-10:]>.5)<1):#np.mean(LocHistArraySortedMax[int(ll),:])
                            placeCells+=1
                            placeCellIndex.append(ll)
                            # print(placeCells,ll)
        if LocHistArraySortedMax[placeCellIndex,:].shape[0]>0:
            fig2 = go.Figure(data=go.Heatmap(
                        z=np.flipud(LocHistArraySortedMax[placeCellIndex,:]),
                        x=np.linspace(0,100,101),
                        type = 'heatmap',
                        colorscale = 'inferno'))
            fig.show()
            wandb.log({"PlaceCellsVal": fig2})

        wandb.log({"Percentage of Putative Place CellsVal": placeCells*100/self.CA1L2Size})
        wandb.log({"Number of Putative Place CellsVal": placeCells})
        wandb.log({"Percentage of Putative Place Cells relative to active cells Val": placeCells*100/LocHistArraySortedMax.shape[0]})
        wandb.log({"LocHistArraySortedMaxVal": LocHistArraySortedMax})
        noOfBins=200
        emptyBins=np.sum(np.sum(LocHistArraySortedMax[placeCellIndex,:],axis=0)==0)
        if emptyBins>0:
            coverageRangePercent=((noOfBins-emptyBins)/noOfBins)*100
        else:
            coverageRangePercent=0
        wandb.log({"PrcntcoverageRange Val": coverageRangePercent})   
        wandb.log({"PrcntcoverageRangeMultipliedbyPrcntPlaceCell Val": coverageRangePercent*(placeCells*100/LocHistArraySortedMax.shape[0])})
        torch.save(self.CA1L2[0].weight,'weightMatrix' +'activityPunishment Val'+str(self.activityPunishmentWeight)+'CA1Size'+str(self.CA1L2Size)+'Dropout'+str(self.droupoutValue)+'weight_decayFactor'+str(self.weight_decayFactor)+'rho'+str(self.RHO)+ '.pt')

        CnctvtyMat=np.asarray(self.CA1L2[0].weight.detach().cpu())
        fig5 = go.Figure(go.Scatter(x=np.linspace(1, 19,19), y=CnctvtyMat[40:59,49]))
        wandb.log({"CrossSection": fig5})

        fig3 = go.Figure(data=go.Heatmap(
                    z=np.flipud(self.CA1L2[0].weight.detach().cpu())))
        fig3.show()
        # px.imshow(self.CA1L2[0].weight.detach().cpu())
        
        wandb.log({"WeightMatrix Val": fig3})
    def test_step(self, batch,batch_idx):

        x_out1=torch.flatten(batch[1] , end_dim=1)
        targets=torch.flatten(batch[1] , end_dim=1)     

        actualLocation=batch[0][:,:]


        PlaceCells1=self.CA1L2(x_out1)

        preds1=self.CA1ReadoutNode(PlaceCells1)
        preds=preds1.squeeze(-1)
        loss=nn.MSELoss()
        # kl_loss = nn.KLDivLoss(log_target=True,reduction='sum')
        if self.activityPunishment:
            activityPunishmentLoss=(self.activityPunishmentWeight*(torch.sum(PlaceCells1**2)))
            MSELoss=loss(preds,targets)
            kl_loss=self.KLmultiplier*kl_divergence(self.RHO, PlaceCells1)
            distance_loss=MSELoss+activityPunishmentLoss+kl_loss
            

        self.log("test total/loss_epoch",distance_loss, on_step=False, on_epoch=True)  # d
        self.log("test KL/loss_epoch",kl_loss, on_step=False, on_epoch=True)  # d
        self.log("test MSE/loss_epoch",MSELoss, on_step=False, on_epoch=True)  # d
        
        
        
       
        OriginalDecays=((batch[1])*1).to(device='cpu').detach()
        predictedDecays=(preds1*1).to(device='cpu').detach()
        predictedDecays=torch.reshape(predictedDecays,(batch[1].shape))
        PlaceCellsL1=(PlaceCells1*1).to(device='cpu').detach()
        PlaceCellsL1=torch.reshape(PlaceCellsL1,(batch[1].shape[0],batch[1].shape[1],self.CA1L2Size))
        # PlaceCellsL2=(PlaceCells2*1).to(device='cpu').detach()
        # randmID=torch.randint(10,[1])
        self.activityLogs.append([actualLocation,PlaceCellsL1,OriginalDecays[0,:,0],predictedDecays[0,:,0],distance_loss.detach()])
        return distance_loss
    def test_epoch_end(self, batch):
        # breakpoint()
        [LocHistArraySortedMax,TimeHistArraySortedMax,LocHistArraySortedFirts2Fire,TimeHistArraySortedFirts2Fire,Time2LocMatchedHistArraySortedMax,LocHistArraySortedMaxlog10] =plotPlaceFields(self.activityLogs,False,2,maxLocationPossible=3000)
        #     # fig=px.imshow(LocHistArraySortedMax)
        # plt.off()
        # fig1,ax1 = plt.subplots()
        # px.imshow(LocHistArraySortedMax,binary_string=True)
        fig = go.Figure(data=go.Heatmap(
                    z=np.flipud(LocHistArraySortedMax),
                    x=np.linspace(0,180,181),
                    type = 'heatmap',
                    colorscale = 'inferno'))
        fig.show()
        wandb.log({"AllCells": fig})
        
        fig = go.Figure(data=go.Heatmap(
                    z=np.flipud(LocHistArraySortedMax),
                    # x=np.logspace(np.log(.001), np.log(180), 181),
                    type = 'heatmap',
                    colorscale = 'inferno'))
        
        
        fig.update_xaxes(type="log")
        # fig.show()
        wandb.log({"AllCellsLog": fig})
        # plt.close('all')
        placeCells=0
        # cherrypickedPlaceFields.append(LocHistArraySortedMax[int(ll),:])
        # cherrypickedPlaceFields=np.vstack(cherrypickedPlaceFields)
        placeCellIndex=[]
        for ll in range(LocHistArraySortedMax.shape[0]): 
                if (np.sum(LocHistArraySortedMax[int(ll),:]>.5)<72) & (np.sum(LocHistArraySortedMax[int(ll),:5]>.9)<5)& (np.sum(LocHistArraySortedMax[int(ll),-5:]>.9)<5):
                    # if :
                        # if (np.sum(LocHistArraySortedMax[int(ll),-10:]>.5)<1):#np.mean(LocHistArraySortedMax[int(ll),:])
                            placeCells+=1
                            placeCellIndex.append(ll)
                            # print(placeCells,ll)
        if LocHistArraySortedMax[placeCellIndex,:].shape[0]>0:
            fig2 = go.Figure(data=go.Heatmap(
                        z=np.flipud(LocHistArraySortedMax[placeCellIndex,:]),
                        x=np.linspace(0,100,101),
                        type = 'heatmap',
                        colorscale = 'inferno'))
            fig.show()
            wandb.log({"PlaceCells": fig2})
        # newPlaceCEllMatrix=
        
        # for placeCs in placeCellIndex:
        # breakpoint()
        wandb.log({"Percentage of Putative Place Cells": placeCells*100/self.CA1L2Size})
        wandb.log({"Number of Putative Place Cells": placeCells})
        wandb.log({"Percentage of Putative Place Cells relative to active cells": placeCells*100/LocHistArraySortedMax.shape[0]})
        wandb.log({"LocHistArraySortedMax": LocHistArraySortedMax})
        noOfBins=200
        emptyBins=np.sum(np.sum(LocHistArraySortedMax[placeCellIndex,:],axis=0)==0)
        if emptyBins>0:
            coverageRangePercent=((noOfBins-emptyBins)/noOfBins)*100
        else:
            coverageRangePercent=0
        wandb.log({"PrcntcoverageRange": coverageRangePercent})   
        wandb.log({"PrcntcoverageRangeMultipliedbyPrcntPlaceCell": coverageRangePercent*(placeCells*100/LocHistArraySortedMax.shape[0])})
        torch.save(self.CA1L2[0].weight,'weightMatrix' +'activityPunishment'+str(self.activityPunishmentWeight)+'CA1Size'+str(self.CA1L2Size)+'Dropout'+str(self.droupoutValue)+'weight_decayFactor'+str(self.weight_decayFactor)+'rho'+str(self.RHO)+ '.pt')
        CnctvtyMat=np.asarray(self.CA1L2[0].weight.detach().cpu())
        fig5 = go.Figure(go.Scatter(x=np.linspace(1, 19,19), y=CnctvtyMat[40:59,49]))
        wandb.log({"CrossSection": fig5})
        


        fig3 = go.Figure(data=go.Heatmap(
                    z=np.flipud(self.CA1L2[0].weight.detach().cpu())))
        fig3.show()
        # px.imshow(self.CA1L2[0].weight.detach().cpu())
        
        wandb.log({"WeightMatrix": fig3})
        
       
  #######################################



sweep_config = {
  "method": "grid",
  "metric": {
    "name": "PrcntcoverageRangeMultipliedbyPrcntPlaceCell",
    "goal": "maximize"
  },
  "parameters": {
      "CA1Size": {
    "values": [100]#50,100,200,300]#[10,25,50,75,100,150,200,250,300]
      },
    "Dropout": {
       "values": [0]#0.1,0.2,0.3]#[0,.1,.2,.3]#[0, .05,.1,.15,.2,.25,.3]
    },
# "epochs": {
#   "values": [100]
# },
    
    "weight_decayFactor": {
        "values":[0]#.0011e-2,0.001,0.0001]#.00001,0.0001,0.001,1e-2]# [0,0.001,.01,.1]#[0,0.001, .005,.01,0.025,.05,.075,.1]
    },
    # "learning_rate": {
    #   "values": [1e-4]
    # },
    "activityPunishment": {
"values": [0]#1e-9,1e-8,1e-7,1e-6]#,0,1e-9]#[0,1e-9,1e-8,1e-7]#[0,1e-9, 2.5e-9,5e-9,7.5e-9,1e-8, 2.5e-8,5e-8,7.5e-8,1e-7, 2.5e-7,5e-7,7.5e-7]
    },

    "KLmultiplier": {
"values": [1e-4]#,3,4,5]#,1,2,3,4,5,6,7,8,9]#,5,10]#,100,1000]#0,5,10,100]#[0,1e-9,1e-8,1e-7]#,0,1e-9]#[0,1e-9,1e-8,1e-7]#[0,1e-9, 2.5e-9,5e-9,7.5e-9,1e-8, 2.5e-8,5e-8,7.5e-8,1e-7, 2.5e-7,5e-7,7.5e-7]
    },
  
    "rho": {
"values": [.1]#0.01,0.1,.2,.3,.4,.5,.6,.7]#0.001,0.01,0.05,0.1,.2,.3,.4,.5,.6,.7,.8,.9]#[0.0001,0.001,0.01,0.1,.25,0.5,0.75]#.01,0.1,.25,0.5,0.75,0.0001,0.001,0.01,0.1,.25,0.35,0.45]#,0.55,.6,.65,.7,0.75]#,5,10]#,100,1000]#0,5,10,100]#[0,1e-9,1e-8,1e-7]#,0,1e-9]#[0,1e-9,1e-8,1e-7]#[0,1e-9, 2.5e-9,5e-9,7.5e-9,1e-8, 2.5e-8,5e-8,7.5e-8,1e-7, 2.5e-7,5e-7,7.5e-7]
    },
    "upperVal": {
"values": [1e-1]#,1e-2]#[0.0001,0.001,0.01,0.1,.25,0.5,0.75]#.01,0.1,.25,0.5,0.75,0.0001,0.001,0.01,0.1,.25,0.35,0.45]#,0.55,.6,.65,.7,0.75]#,5,10]#,100,1000]#0,5,10,100]#[0,1e-9,1e-8,1e-7]#,0,1e-9]#[0,1e-9,1e-8,1e-7]#[0,1e-9, 2.5e-9,5e-9,7.5e-9,1e-8, 2.5e-8,5e-8,7.5e-8,1e-7, 2.5e-7,5e-7,7.5e-7]
    },
    "lowerVal": {
"values": [1e-3]#,1e-4,1e-5]#[0.0001,0.001,0.01,0.1,.25,0.5,0.75]#.01,0.1,.25,0.5,0.75,0.0001,0.001,0.01,0.1,.25,0.35,0.45]#,0.55,.6,.65,.7,0.75]#,5,10]#,100,1000]#0,5,10,100]#[0,1e-9,1e-8,1e-7]#,0,1e-9]#[0,1e-9,1e-8,1e-7]#[0,1e-9, 2.5e-9,5e-9,7.5e-9,1e-8, 2.5e-8,5e-8,7.5e-8,1e-7, 2.5e-7,5e-7,7.5e-7]
    },
  }
}

def train(config=None):
    with wandb.init(config=config,entity="placecellthesis"):
        config = wandb.config

        gc.collect()                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                            
        torch.cuda.empty_cache()

        try:
            # seed_everything(seed=427)
            
            PCmodel = EC_HPC(                                                                                                                                                                                                   
                learningRate=1e-1,#config.learning_rate,
                CA1Size=config.CA1Size,
                activityPunishment=config.activityPunishment,
                Dropout=config.Dropout,
                KLmultiplier=config.KLmultiplier,
                weight_decayFactor=config.weight_decayFactor,
                rho=config.rho,
                batch_size=128,
                upperVal=config.upperVal,
                lowerVal=config.lowerVal
                )
            wandb_logger = WandbLogger(log_model=True)
            # checkpoint_callback = ModelCheckpoint(monitor="1_val/sharpe", mode="max")
            trainer = pl.Trainer(
                # gpus=1,
                max_epochs=100,
                logger=wandb_logger,
                # log_every_n_steps=5000
                # callbacks=[checkpoint_callback]
            )
            
            trainer.fit(PCmodel)
            trainer.test(PCmodel)
            
        except Exception as e:
            print(e)

        # del PCmodel
        # del wandb_logger
        # del checkpoint_callback
        # del trainer
            
        gc.collect()
        torch.cuda.empty_cache()



sweep_id = wandb.sweep(sweep_config, project='1DPlaceCells1LParameterSweepJan2022NotNormalizedUniq')
wandb.agent(sweep_id, function=train)
#####################################################
