#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 16 10:10:01 2023

@author: panlab
"""


!pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
import pip
packages = ['numpy', 'matplotlib', 
           'torchmetrics',
           'torchvision','scipy', 'seaborn',
           'pytorch-lightning-bolts', 'pytorch-lightning',
           'plotly','nice_figures','wandb',
            'pandas','tqdm']

for package in packages:
    pip.main(['install', package])
    # pip.main(['uninstall', '-y', package])
print('All required packages installed successfully!')
