# Self-supervised learning of scale-invariant neural representations of space and time

### Description
Supplementary code for Self-supervised learning of scale-invariant neural representations of space and time
### Dependencies
1. Python 3.10.4
2. Numpy
3. Matplotlib
4. scipy
5. Pytorch-lightning 2.1.0
6. Pytorch-lightning-bolts 0.3.2.post1
7. PyTorch 2.1.0+cu118
8. Seaborn
9. nice_figures
10. wandb
11. pandas
12. tqdm


### Usage Notes:
The model was run on a PC with the following specifications:
1. CPU: AMD Ryzen 9 5950X 16-Core Processor
2. RAM: 64 GB + 100GB SSD Swap
3. GPU: GeForce RTX 3090
4. OS: Ubuntu 22.04.2 LTS

To run the model, first make sure you have all the requirement. A simple script to install all the requirements is included in main folder (InstallRequiredPackages.py). Then, download the datasets and use the ParameterSweep1D1L.py in the main folder to run the main script that runs the model and logs the results on weights and biases (you need to have a weights and biases account for that). For the 2D model, use the file named 2DPolarUsingPretrainedMatrix.py. In order to utilize this script, you need a pretrained weight matrix that is obtained from the ParameterSweep1D1L.py. A sample pretrained matrix has been added to the main file for easy usage.  
### Dataset
Used datasets are kept here:
https://drive.google.com/drive/folders/1Wa0AWuSkF3eULUVvYQ48eV9kP4teFtNp?usp=share_link

You can generate your own datasets or modify the existing ones using the scripts in this repo.

### Usage Conditions

TBD









