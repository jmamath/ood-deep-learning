# -*- coding: utf-8 -*-
"""
Created on Wed Jul 28 08:41:10 2021

@author: JeanMichelAmath
"""
# perform all steps in the ISDA folder

################ 1. IMPORT FUNCTIONS AND DATA ################ 
import torch
from torchvision.transforms import transforms
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
import time
from torchvision import datasets
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np
from isda import ISDALoss
from isda_utils import train, LeNet5, Full_layer, FullLeNet5, adjust_learning_rate

def get_device():
    if torch.cuda.is_available():
        device = 'cuda:0'
    else:
        device = 'cpu'
    return device
device = get_device()

preprocess = transforms.Compose(
      [transforms.ToTensor(),
       transforms.Normalize([0.5] * 3, [0.5] * 3)])

preprocess_simple = transforms.Compose(
      [transforms.ToTensor()])
  
train_data = datasets.CIFAR10('../cifar', train=True, transform=preprocess, download=False)
test_data = datasets.CIFAR10('../cifar', train=False, transform=preprocess, download=False)

train_loader = torch.utils.data.DataLoader(
      train_data,
      batch_size=128,
      shuffle=True)

test_loader = torch.utils.data.DataLoader(
      test_data,
      batch_size=128,
      shuffle=True)

IMAGE_SIZE = 32



################ 4. MODEL TRAINING ################   
# go to the Synthetic_augment folder to easily load functions from utils
from utils import evaluate_model
from training_loops import train_model

EPOCHS = 5
## 4.1 Train with standard cross entropy loss
# Simple LeNet5 network
global_ce_acc = []
for i in range(5):
    model = LeNet5(feature_num=84).to(device)
    fc = Full_layer(int(model.feature_num), class_num=10).to(device)
    # define loss function (criterion) and optimizer
    ce_criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD([{'params': model.parameters()},
                                {'params': fc.parameters()}],
                                lr=0.1,
                                momentum=0.9,
                                nesterov=True,
                                weight_decay=1e-4)
    
    loss_ce = []
    acc_ce = []
    for epoch in range(0, EPOCHS):
        # train for one epoch
        adjust_learning_rate(optimizer, epoch + 1)
        loss, acc = train(train_loader, model, fc, ce_criterion, optimizer, epoch, loss_isda=False)
        loss_ce.append(loss.ave)
        acc_ce.append(acc.ave)
    
    final_model = nn.Sequential(model, fc)
    global_ce_acc.append(evaluate_model(final_model, test_loader, 1, 10, device))

## 4.2 Train with ISDA loss

global_isda_acc = []
for i in range(5):
    model = LeNet5(feature_num=84).to(device)
    fc = Full_layer(int(model.feature_num), class_num=10).to(device)
    isda_criterion = ISDALoss(int(model.feature_num), class_num=10)
    optimizer = torch.optim.SGD([{'params': model.parameters()},
                                {'params': fc.parameters()}],
                                lr=0.1,
                                momentum=0.9,
                                nesterov=True,
                                weight_decay=1e-4)
    
    loss_isda = []
    acc_isda = []
    for epoch in range(0, 5):
        # train for one epoch
        adjust_learning_rate(optimizer, epoch + 1)
        loss, acc = train(train_loader, model, fc, isda_criterion, optimizer, epoch)
        loss_isda.append(loss.ave)
        acc_isda.append(acc.ave)
    
    final_model_ = nn.Sequential(model, fc)
    global_isda_acc.append(evaluate_model(final_model_, test_loader, 1, 10, device))

plt.plot(loss_ce, label="CE")
plt.plot(loss_isda, label="ISDA")
plt.legend()

global_ce_acc = np.array(global_ce_acc)
global_isda_acc = np.array(global_isda_acc)
print("CE acc: {} +- {} - ISDA acc : {} +- {}".format(global_ce_acc.mean(), global_ce_acc.std(),
                                                      global_isda_acc.mean(), global_isda_acc.std()))

global_fullLenet_acc = []
for i in range(5):
    model = FullLeNet5().to(device)
    # define loss function (criterion) and optimizer
    ce_criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    _, loss = train_model(model, None, 1, train_loader, EPOCHS, ce_criterion, optimizer, 10, None, device, mixup=False, fgsm=False)
    
    global_fullLenet_acc.append(evaluate_model(model, test_loader, 1, 10, device))
    
global_fullLenet_acc = np.array(global_fullLenet_acc)
print("full lenet acc acc: {} +- {} ".format(global_fullLenet_acc.mean(), global_fullLenet_acc.std()))


optimizer = torch.optim.Adam([{'params': model.parameters()},
                                {'params': fc.parameters()}], lr=0.01)

optimizer = torch.optim.SGD([{'params': model.parameters()},
                                {'params': fc.parameters()}],
                                lr=0.1,
                                momentum=0.9,
                                nesterov=True,
                                weight_decay=1e-4)