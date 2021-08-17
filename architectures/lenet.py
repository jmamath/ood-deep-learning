# -*- coding: utf-8 -*-
"""
Created on Mon Jun 14 07:21:47 2021

@author: JeanMichelAmath
"""

import torch.nn as nn
import torch.nn.functional as F

# @title Vanila LeNet-5
class LeNet5(nn.Module):
    def __init__(self, name="Softmax"):
        """
        You will change the filters and channels depending on the dataset to fit
        i.e MNIST or CIFAR
        """
        super(LeNet5, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5, bias=False)    
        self.pool1 = nn.MaxPool2d(2)
        self.conv2 = nn.Conv2d(6, 16, 5, bias=False)   
        self.pool2 = nn.MaxPool2d(2)
        self.fc1 = nn.Linear(400, 120, bias=False)   # 
        self.fc2 = nn.Linear(120, 84, bias=False)        
        self.fc = nn.Linear(84, 10, bias=False) 
        self.name = name       

    def forward(self, x):
        #import pdb; pdb.set_trace()
        y = F.relu(self.conv1(x))        
        y = self.pool1(y)
        y = F.relu(self.conv2(y))        
        y = self.pool2(y)        
        #import pdb; pdb.set_trace()
        y = y.view(y.shape[0], -1)        
        y = F.relu(self.fc1(y))        
        y = F.relu(self.fc2(y))        
        y = self.fc(y)        
        return y  
    
# @title MC_Dropout LeNet-5
class LeNet5Dropout(nn.Module):
    def __init__(self, name="Dropout"):
        """
        You will change the filters and channels depending on the dataset to fit
        i.e MNIST or CIFAR
        """
        super(LeNet5Dropout, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5, bias=False)    
        self.pool1 = nn.MaxPool2d(2)
        self.dropout1 = nn.Dropout(0.25)
        self.conv2 = nn.Conv2d(6, 16, 5, bias=False)   
        self.pool2 = nn.MaxPool2d(2)
        self.dropout2 = nn.Dropout(0.25)
        self.fc1 = nn.Linear(400, 120, bias=False)   
        self.dropout3 = nn.Dropout(0.5)
        self.fc2 = nn.Linear(120, 84, bias=False)        
        self.fc = nn.Linear(84, 10, bias=False) 
        self.name = name       

    def forward(self, x):
        y = F.relu(self.conv1(x))        
        y = self.pool1(y)
        y = self.dropout1(y)
        y = F.relu(self.conv2(y))        
        y = self.pool2(y)   
        y = self.dropout2(y)     
        y = y.view(y.shape[0], -1)        
        y = F.relu(self.fc1(y))        
        y = self.dropout3(y)
        y = F.relu(self.fc2(y))        
        y = self.fc(y)        
        return y    

class EnsembleNeuralNet(nn.Module):
    def __init__(self, base_neural_nets,
                 name="Standard_Item"):
        """
        You should definitely mind the features transformation depending on the dataset,
        it might be different between MNIST and CIFAR-10 for instance,
        bayesLinear1 will have respectively 16*4*4 and 16*5*5 in_features.
        """
        super(EnsembleNeuralNet, self).__init__() 
        self.ensemble = nn.ModuleList(base_neural_nets)
        
    def forward(self, x):              
        for i, l in enumerate(self.ensemble):
            pred = self.ensemble[i](x)
        return pred/len(self.ensemble)