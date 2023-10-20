#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 22 14:38:09 2023

@author: panlab
"""
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct 28 11:11:54 2022

@author: panlab
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 19 11:24:01 2022

@author: panlab
"""

from tqdm import tqdm 
# import pandas as pd
# import wandb
# wandb.login()
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader,random_split
from torchvision import transforms, utils
import torchmetrics
import seaborn as sns
import pytorch_lightning as pl
pl.seed_everything(1234)
from torch import nn
import torch
from torch.distributions import VonMises
# from pl_bolts.models.autoencoders.components import (
#     resnet18_decoder,
#     resnet18_encoder,
# )
# from brian2 import *
# from brian2tools import *
# from vaePytorchLightiningImagePlottingCallback import ImageSampler
from argparse import ArgumentParser

# from pytorch_lightning.loggers import WandbLogger


# 🏋️‍♀️ Weights & Biases
# import wandb

# ⚡ 🤝 🏋️‍♀️
# from pytorch_lightning.loggers import WandbLogger
import torch.nn.utils.parametrize as parametrize
import os
os.chdir('/media/panlab/Thesis/DeepCodes')
from resultAnalyzer import *
pl.seed_everything(hash("setting random seeds") % 2**32 - 1)



class TrajectoryDataset(Dataset):
    

    def __init__(self, torchFile,transform=None,num_workers=30):
        """
        Args:
            csv_file (string): Path to the npy file with Trajectories.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.TrajectorySamples = torch.load(torchFile)

        self.transform = transform

    def __len__(self):
        return len(self.TrajectorySamples)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()


      
        sample = self.TrajectorySamples[idx]

        if self.transform:
           sample[1] = (sample[1]-sample[1].mean())/sample[1].std()

        return sample


class TrajectoryDataModule(pl.LightningDataModule):

    def __init__(self, data_dir='./', batch_size=2):
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.transform = None
        

    def setup(self, stage=None):

        # we set up only relevant datasets when stage is specified
        if stage == 'fit' or stage is None:
            
            trajDSet=TrajectoryDataset('TrainingSetPolar2DNoModRaw.pt')
            self.trajDSet_train, self.trajDSet_val = random_split(trajDSet, [399, 1]) 
        if stage == 'test' or stage is None:
           self.trajDSet_test=TrajectoryDataset('TrainingSetPolar2DNoModRawLogSpacedCIRCULAR.pt')

    # we define a separate DataLoader for each of train/val/test
    def train_dataloader(self):
        trajDSet_train = DataLoader(self.trajDSet_train, batch_size=self.batch_size)
        return trajDSet_train

    def val_dataloader(self):
        trajDSet_val = DataLoader(self.trajDSet_val, batch_size=1 * self.batch_size)
        return trajDSet_val

    def test_dataloader(self):
        trajDSet_test = DataLoader(self.trajDSet_test, batch_size=1 * self.batch_size)
        return trajDSet_test












learningRateVec=[1e-2,1e-3,1e-4,1e-5]
recurrentSizeVec=[500]#,1000,2000]
CA1SizeVec=[50,250,500]#,100,200]
activityPunishmentVec=[1e-7,1e-8,1e-9]
dropOutVec=[.1,.25,.5,.75]
timeStepsToPredictVec=[1,5,10,20,40]

weightDecayVec=[0]#,1e-3]
learningRateVec=[1e-4]
L2SizeVec=[1000]
CA1SizeVec=[250]
activityPunishmentVec=[0]
dropOutVec=[.25]
timeStepsToPredictVec=[1]

os.chdir('DIRECTORY OF YOUR CODES')
pl.seed_everything(hash("setting random seeds") % 2**32 - 1)
torch.cuda.empty_cache()

weightDecay=0
learningRate=1e-4

activityPunishment=0
Dropout=.25
thetaBinCenters=8
placeCellLayerIn1DModel=100

class EC_HPC(pl.LightningModule):
    def __init__(self,input_size=1,
                  hiddenSizeMEC=100,thetaBinCenters=thetaBinCenters,hiddenSizeCA3=400,hiddenSizeCA1=thetaBinCenters*placeCellLayerIn1DModel,dropout=0.25,a=-1.,b=1.,c=-1.):
        super().__init__()
        
            
        self.activityLogs=[]
        self.save_hyperparameters()
        self.thetaBinCenters=thetaBinCenters
        self.CA1L2Size=hiddenSizeCA1
               
        self.CA1L2=        nn.Sequential(
                     nn.Linear(hiddenSizeMEC, out_features=self.CA1L2Size,bias=False),
                     nn.ReLU())
        trainedWeightMatrixOn1D=torch.load('USEMEweightMatrixactivityPunishment0CA1Size100Dropout0weight_decayFactor0rho0.7.pt')#.t()# transpose is done to make this consistant with pytorch conventions (pytorch transposes the matrix)
        
        self.CA1L2[0].weight=nn.Parameter(torch.zeros(self.CA1L2[0].weight.size()),requires_grad=False)
              
        for binCenterNo in range(self.thetaBinCenters):
            self.CA1L2[0].weight[(binCenterNo*hiddenSizeMEC):hiddenSizeMEC+(binCenterNo*hiddenSizeMEC),:]=nn.Parameter(trainedWeightMatrixOn1D,requires_grad=False)
       
        
        
        
        
        
        
     
        self.BooleanDropOut=True
        self.DropoutVal=Dropout
        self.dropout = nn.Dropout(self.DropoutVal)
        

        self.CA1ReadoutNode=nn.Sequential(
            nn.Linear(self.CA1L2Size, out_features=hiddenSizeMEC,bias=False))
        
        self.weight_decayFactor=weightDecay
        self.activityPunishment=True
        self.activityPunishmentWeight=activityPunishment
        
        
        self.train_acc = torchmetrics.Accuracy()
        self.valid_acc = torchmetrics.Accuracy()
        self.test_acc = torchmetrics.Accuracy()
        self.save_hyperparameters()
    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=learningRate,weight_decay=self.weight_decayFactor)


        
    
    def training_step(self, batch, batch_idx):
        # breakpoint()

        if self.BooleanDropOut:
            x_out1=self.dropout(torch.flatten(batch[1] , end_dim=1))
        else:
            x_out1=torch.flatten(batch[1] , end_dim=1)

        targets=torch.flatten(batch[1] , end_dim=1)       # x_out1=self.dropout(batch[0][:,:,1])

        PlaceCells1=self.CA1L2(x_out1)

        preds=self.CA1ReadoutNode(PlaceCells1)
        preds=preds.squeeze(-1)
        loss=nn.MSELoss()
        if self.activityPunishment:    

            distance_loss=loss(preds,targets)+(self.activityPunishmentWeight*(torch.sum(PlaceCells1)))#+torch.sum(torch.tril(self.CA1[0].weight,-1))*self.activityPunishmentWeight#*
        elif not(self.activityPunishment):#+torch.abs(torch.sum(self.CA1[0].weight))
            distance_loss=loss(preds,targets)#+torch.abs(torch.sum(self.CA1[0].weight))


        

        self.log('train/loss', distance_loss, on_step=False, on_epoch=True)
        return distance_loss
    def validation_step(self, batch, batch_idx):
        # breakpoint()
        if self.BooleanDropOut:
            x_out1=self.dropout(torch.flatten(batch[1] , end_dim=1))
        else:
            x_out1=torch.flatten(batch[1] , end_dim=1) 
        # xaxis=x_out1[:,:,50]
        # yaxis=x_out1[:,:,150]
        # with torch.no_grad():
        #     x_out1=torch.outer(xaxis,yaxis)
        targets=torch.flatten(batch[1] , end_dim=1)      
        # CA1Out=self.CA1L1(x_out1)
        # CA1OutMiddle=self.CA1L2(CA1Out[0])
        PlaceCells1=self.CA1L2(x_out1)
        # PlaceCells2=self.CA1L3(PlaceCells1)
        # CA1Out=self.CA1L3(CA1OutMiddle)

        # preds=self.CA1ReadoutNode(CA1Out[0])
        preds=self.CA1ReadoutNode(PlaceCells1)
        preds=preds.squeeze(-1)
        loss=nn.MSELoss()
        if self.activityPunishment:    
           
            distance_loss=loss(preds,targets)+(self.activityPunishmentWeight*(torch.sum(PlaceCells1)))#+torch.sum(torch.tril(self.CA1[0].weight,-1))*self.activityPunishmentWeight#*
        elif not(self.activityPunishment):#+torch.abs(torch.sum(self.CA1[0].weight))
            distance_loss=loss(preds,targets)#+torch.abs(torch.sum(self.CA1[0].weight))

        self.log("valid/loss_epoch",distance_loss, on_step=False, on_epoch=True) 
        return distance_loss
    
    def test_step(self, batch, batch_idx):
       
        x_out1=torch.flatten(batch[1] , end_dim=1)
        
        PlaceCells1=self.CA1L2(x_out1)
       
        PlaceCellsL1=(PlaceCells1*1).to(device='cpu').detach()
        PlaceCellsL1=torch.reshape(PlaceCellsL1,(batch[1].shape[0],batch[1].shape[1],self.CA1L2Size)) 
        nSteps=batch[1].shape[1]
        thetaBinCenters=self.thetaBinCenters
       
        angularBinCentersInRadians=np.deg2rad(torch.linspace(360/thetaBinCenters,360,thetaBinCenters))#*np.pi/180
        
        angularModulationFunctions=VonMises(angularBinCentersInRadians, torch.tensor(10.0))
        tmpActvtyLogsLoc=[]
        tmpActvtyLogsPlaceCells=[]
        sizeOfCA1L2In1D=int(self.CA1L2Size/thetaBinCenters)
        
        for batchID in range(batch[0].shape[0]):
                
                actualLocation=batch[0][batchID,:,0:].clone().detach().requires_grad_(False).cpu()

                for stepID in range(nSteps) :
                
                    currentAngle=batch[0][batchID,stepID,3].clone().detach().requires_grad_(False).cpu()
                    angularModulationValues=angularModulationFunctions.log_prob(currentAngle).exp()
                    angularMudulationVector=torch.ones(PlaceCellsL1.shape[2])
                    
                    
                    for binCenterNo in range(thetaBinCenters):
                        angularMudulationVector[(binCenterNo*sizeOfCA1L2In1D):sizeOfCA1L2In1D+(binCenterNo*sizeOfCA1L2In1D)]=angularModulationValues[binCenterNo]
        
                    # for batchNo in range(batch[1].shape[0]):
                    PlaceCellsL1[batchID,stepID,:] = PlaceCellsL1[batchID,stepID,:] *angularMudulationVector
                PlaceCellsOut=PlaceCellsL1[batchID,:,:] 
                self.activityLogs.append([actualLocation,PlaceCellsOut])
                
    
    def test_epoch_end(self, batch):
       
        torch.save(self.activityLogs,'ValDataLogsPretainedMatrix2DPolar.pt')
        
# setup data
trajData = TrajectoryDataModule()
trajData.setup()
trainer = pl.Trainer(

    gpus=-1,                # use all GPUs
    deterministic=True,     # keep it deterministic
    
    max_epochs=150)           # number of epochs



# setup model
model = EC_HPC()
trainer.test(model,trajData)
#%%

ValDataLogs=torch.load('ValDataLogsPretainedMatrix2DPolar.pt')
pretrainingMat=model.CA1L2[0].weight.detach()
plt.figure(),plt.imshow(pretrainingMat.T,cmap='bwr'),plt.colorbar()
plt.savefig('ConnectivityMatrixB4Training2D'+'.png')

# fit the model
trainer.fit(model, trajData)
trainer.test(model,trajData)

# Directory
directory = "2DLearningRate"+str(learningRate)+ "CA1Size"+str(CA1Size)+"ActivityPunishment"+str(activityPunishment)+"DropOut"+str(Dropout)+"Reconstruction"+"WeightDecay"+str(weightDecay)+"1LayerCrossed"

parent_dir = "DIRECTORY OF YOUR CODES/PlaceFieldFigs/"
  
# Path
path = os.path.join(parent_dir, directory)
os.mkdir(path)
os.chdir(path)

torch.save(model.activityLogs, 'modelActivityLogs.pt')

plotPlaceFields2D(model.activityLogs)

#%%
import torch
from resultAnalyzer import *
ValDataLogs=torch.load('ValDataLogsPretainedMatrix2DPolar.pt')
plotPlaceFields2D(ValDataLogs) 
