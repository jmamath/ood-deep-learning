# -*- coding: utf-8 -*-
"""
Created on Fri Jul 30 07:54:10 2021

@author: JeanMichelAmath
"""

import time
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F

################ 2. PREPARE THE ISDA TRAINING LOOP ################ 
EPOCHS = 7
lambda_0 = 0.5

def train_isda(train_loader, model, fc, criterion, optimizer, epoch, device, scheduler, loss_isda=True):
    """Train for one epoch on the training set"""
    train_batches_num = len(train_loader)
    
    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()

    ratio = lambda_0 * (epoch / (EPOCHS))
    # switch to train mode
    model.train()
    fc.train()

    end = time.time()
    for i, (x, target) in enumerate(train_loader):
        target = target.to(device)
        x = x.to(device)
        if loss_isda:
            input_var = torch.autograd.Variable(x)
            target_var = torch.autograd.Variable(target)
            # compute output
            loss, output = criterion(model, fc, input_var, target_var, ratio)
        else:
            features = model(x)
            output = fc(features)
            loss = criterion(output, target)
        # measure accuracy and record loss
        prec1 = accuracy(output.data, target, topk=(1,))[0]
        losses.update(loss.data.item(), x.size(0))
        top1.update(prec1.item(), x.size(0))

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()

        optimizer.step()
        scheduler.step()
        batch_time.update(time.time() - end)
        end = time.time()


        if (i+1) % 391 == 0:
            # print(discriminate_weights)
            # fd = open(record_file, 'a+')
            string = ('Epoch: [{0}][{1}/{2}]\t'
                      'Time {batch_time.value:.3f} ({batch_time.ave:.3f})\t'
                      'Loss {loss.value:.4f} ({loss.ave:.4f})\t'
                      'Precision_1 {top1.value:.3f} ({top1.ave:.3f})\t'.format(
                       epoch, i+1, train_batches_num, batch_time=batch_time,
                       loss=losses, top1=top1))

            print(string)
    return losses, top1
            # print(weights)
            # fd.write(string + '\n')
            # fd.close()
            
class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()      

    def reset(self):
        self.value = 0
        self.ave = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.value = val
        self.sum += val * n
        self.count += n
        self.ave = self.sum / self.count
        
def adjust_learning_rate(optimizer, epoch):
    """Sets the learning rate"""
    for param_group in optimizer.param_groups:
        param_group['lr'] = 0.5 * 0.1\
                            * (1 + np.cos(np.pi * epoch / EPOCHS))
        
        
def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res        

################ 3. MAIN MODEL ARCHITECTURE ################
class Full_layer(torch.nn.Module):
    '''explicitly define the full connected layer'''

    def __init__(self, feature_num, class_num):
        super(Full_layer, self).__init__()
        self.class_num = class_num
        self.fc = nn.Linear(feature_num, class_num, bias=False)

    def forward(self, x):
        x = self.fc(x)
        return x
    
 