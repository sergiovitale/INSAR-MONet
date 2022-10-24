""" 
This is a demo of the testing phase of INSAR-MONet:
    you find the three simulated testing cases shown in the relative paper.
    
This code is under license of University Parthenope of Naples and it can be use for research purpose only.
If you use it and find useful for your research cite it as follows:

    mettere citazione
"""
#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Fri Jan  4 14:37:53 2019

@author: sergio
"""

import numpy as np
import scipy.io as sio
import matplotlib.pyplot as plt
import torch
import os

#%% Setting
"""choose testing file (uncomment the desired one)"""
testing_file = 'test_b1_g4.mat'
# testing_file = 'test_b2_g3.mat'
# testing_file = 'test_b3_g1.mat'

path_in = './data/' #select input data folder
path_out = './results/' #select output data folder
model_path = './model' #select trained model path

if not os.path.exists(path_out):
    os.makedirs(path_out)
    
id_device = 0 #select accelerator (index of the gpu, if gpu not available it will run on cpu)
device = torch.device("cuda:%d"%(id_device) if torch.cuda.is_available() else "cpu")
print('testing on '+str(device))
#%% Loading
from model import Net
blk=17
data = sio.loadmat(path_in+testing_file)

phi_n = np.float32(data['phi_n']) #loading noisy data
phi_n = np.pad(phi_n,((blk,blk),(blk,blk))) #padding for taking care of receptive field

phi = np.float32(data['phi']) #loading reference noise-free data
        

#%% Loading Network

net = Net() #define model
net.load_state_dict(torch.load(model_path)) #loading trained weights
net.to(device) #moving to accelerator
net.eval() # testing mode
        
#%% testing
with torch.no_grad():
    output = net(torch.from_numpy(phi_n[np.newaxis,np.newaxis,:,:]).to(device))
    output = output.detach().cpu() #moving to cpu
    
    #saving output
    d = {}
    d['output']=output.detach().cpu()
    sio.savemat(path_out+'out_'+testing_file,d)

#%% Visualization:
plt.close('all')
min_value =-3.14
max_value=3.14
plt.figure()
plt.subplot(131),plt.imshow(np.squeeze(phi_n[blk:-blk,blk:-blk]),vmin=min_value,vmax=max_value,cmap=plt.jet()),plt.title('noisy')
plt.subplot(132),plt.imshow(np.squeeze(phi),vmin=min_value,vmax=max_value,cmap=plt.jet()),plt.title('image- reference')
plt.subplot(133),plt.imshow(np.squeeze(output[:,:,blk:-blk,blk:-blk]),vmin=min_value,vmax=max_value,cmap=plt.jet()),plt.title('image -filtered')
       
