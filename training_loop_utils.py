# -*- coding: utf-8 -*-
"""
Created on Mon Aug 16 08:28:26 2021

@author: JeanMichelAmath
"""

import torch 
from tqdm import trange
import numpy as np
from torch.autograd import Variable
import torch.nn.functional as F
from torch.autograd import grad
import torch.nn as nn
from adain.adain_utils import augment

def get_device():
    if torch.cuda.is_available():
        device = 'cuda:0'
    else:
        device = 'cpu'
    return device
device = get_device()


def shuffle_minibatch(inputs, targets, no_classes, device, mixup=True):
    """Shuffle a minibatch and do linear interpolation between couple of inputs
    and targets.
    source: https://github.com/leehomyc/mixup_pytorch/blob/master/main_cifar10.py    
    Args:
        inputs: a numpy array of images with size batch_size x H x W x 3.
        targets: a numpy array of labels with size batch_size x 1.
        mixup: a boolen as whether to do mixup or not. If mixup is True, we
            sample the weight from beta distribution using parameter alpha=1,
            beta=1. If mixup is False, we set the weight to be 1 and 0
            respectively for the randomly shuffled mini-batches.
    """
    batch_size = inputs.shape[0]
    
    # We start by generating a batch of couple of inputs and targets.
    rp1 = torch.randperm(batch_size)
    inputs1 = inputs[rp1]
    targets1 = targets[rp1]
    targets1_1 = targets1.unsqueeze(1)

    rp2 = torch.randperm(batch_size)
    inputs2 = inputs[rp2]
    targets2 = targets[rp2]
    targets2_1 = targets2.unsqueeze(1)

    y_onehot = torch.FloatTensor(batch_size, no_classes).to(device)
    y_onehot.zero_()
    targets1_oh = y_onehot.scatter_(1, targets1_1, 1)

    y_onehot2 = torch.FloatTensor(batch_size, no_classes).to(device)
    y_onehot2.zero_()
    targets2_oh = y_onehot2.scatter_(1, targets2_1, 1)
    
    # For each couple of data, we have an associated alpha for mixing up.
    if mixup == True:
        alpha = np.random.beta(1, 1, [batch_size, 1])
    else:
        alpha = np.ones((batch_size, 1))
    
    
    b = np.tile(alpha[:,:, None, None], [1, 3, 32, 32]) #for images of shape [3,32,32]
    #b = np.tile(alpha[...], [2])    # We adapt to the toy dataset of shape [2]

    # We compute the linear interpolation between inputs x hat
    
    inputs1 = inputs1 * torch.from_numpy(b).float().to(device)
    inputs2 = inputs2 * torch.from_numpy(1 - b).float().to(device)
    inputs_shuffle = inputs1 + inputs2
    
    # We prepare the interpolation for each class so one dimension per class
    c = np.tile(alpha, [1, no_classes])

    # We compute the linear interpolation for each dimension of the labels
    targets1_oh = targets1_oh.float() * torch.from_numpy(c).float().to(device)
    targets2_oh = targets2_oh.float() * torch.from_numpy(1 - c).float().to(device)


    targets_shuffle = targets1_oh + targets2_oh

    return inputs_shuffle, targets_shuffle

def compute_svi_pred(posterior_model ,batch_x, MC_sample, no_classes, device): 
    # If MC_sample is equal to 1 then it is normal training, otherwise, if it is greater than one
    # then it is an implementation of Monte Carlo Dropout.
    # compute MC_sample Monte Carlo predictions and average them   
  predictions = torch.zeros((MC_sample,len(batch_x), no_classes), dtype=torch.float32).to(device)
  for j in range(MC_sample):
    # forward pass: compute predicted outputs by passing inputs to the model
    predictions[j] = posterior_model(batch_x)      
  pred = predictions.mean(0)   
  return pred

def compute_accuracy(pred, y):
  _, predicted = torch.max(F.softmax(pred), 1)
  total = len(pred)
  correct = (predicted == y).sum()
  accuracy = 100 * correct.cpu().numpy() / total 
  return accuracy  

def train_model(posterior_model, train_loader, n_epochs, criterion, optimizer, no_classes, device, scheduler=None, MC_sample=1, style_loader=None , mixup=False, fgsm=False, alpha=0.1):
    '''
    This training loop can optionally train up to 4 different methods:
    - Monte Carlo Dropout with the MC_sample parameter
    - FGSM as adversarial training
    - Mixup the data augmentation technique
    - Style transfer with a style loader and a parameter alpha
    
    Parameters
    ----------
    posterior_model : torch model
        neural network .
    train_loader : torch data loader
        training loader of images.
    n_epochs : Int
        Number of epochs for full training.
    criterion : TYPE
        loss function.
    optimizer : torch optimizer
        DESCRIPTION.
    no_classes : Int
        number of classes.
    device : string
        cpu or gpu.
    scheduler : torch scheduler, optional
       DESCRIPTION. The default is None.
    MC_sample : Int, optional
        How many times to perform a forward operation this is for Monte Carlo Dropout.
    style_loader : torch data loader
        a data loader of style images. It is used only to train with style transfer, optional, The default is None.
    mixup : Boolean, optional
        whether to use mixup during training or not
    fgsm : Boolean, optional
        whether to use fast gradient sign method during training or not   
    alpha : Float, optional
        This control the degree of synthetic transfer, 0 being without any style transfer
        and 1 being the style image. The default is 0.1.

    Returns
    -------
    None.

    '''
    
    # FGSM: https://github.com/locuslab/fast_adversarial/blob/master/MNIST/train_mnist.py
    
    # to track the training log likelihood as the model trains
    train_log_likelihood = []
    # to track the average training log likelihood per epoch as the model trains
    avg_train_log_likelihood = [] 
    
    # to track the training accuracy as the model trains
    train_acc = []
    # to track the average acc per epoch as the model trains
    avg_train_acc = [] 
    
    # for epoch in range(1, n_epochs + 1):
    with trange(n_epochs) as pbar:
      for epoch in pbar:
        ###################
        # train the model #
        ###################        
        for batch_nb, (batch_x, batch_y) in enumerate(train_loader, 1):
            #print(batch_nb)
            batch_y = batch_y.to(device)
            if style_loader is not None:
                # here batch_x will be sent to device
                batch_x = augment(style_loader, batch_x, device, alpha)
            else:
                batch_x = batch_x.to(device)
            # clear the gradients of all optimized variables
            optimizer.zero_grad()
            if (mixup == True):
                #import pdb; pdb.set_trace()
                batch_x, batch_y_mu = shuffle_minibatch(batch_x, batch_y, no_classes, device, True)
            if fgsm == True:
                epsilon=0.3
                alpha = 0.375
                delta = torch.zeros_like(batch_x).uniform_(-epsilon, epsilon).to(device)
                delta.requires_grad = True
                output = posterior_model(batch_x + delta)
                loss = F.cross_entropy(output, batch_y)
                loss.backward()
                grad = delta.grad.detach()
                delta.data = torch.clamp(delta + alpha * torch.sign(grad), -epsilon, epsilon)
                delta.data = torch.max(torch.min(1-batch_x, delta.data), 0-batch_x)
                delta = delta.detach()            
#            import pdb; pdb.set_trace()

            prediction = compute_svi_pred(posterior_model, batch_x, MC_sample, no_classes, device)                 
            #import pdb; pdb.set_trace()
            if mixup == True:
                log_likelihood = criterion(prediction, batch_y_mu)
            else:                
                log_likelihood = criterion(prediction, batch_y)
            loss = log_likelihood
           # backward pass: compute gradient of the loss with respect to model parameters
            loss.backward()                 
            # perform a single optimization step (parameter update)
            optimizer.step() 
            scheduler.step()
            acc = compute_accuracy(prediction, batch_y) 
            train_acc.append(acc)
            # record training log likelihood, KL and accuracy
            train_log_likelihood.append(log_likelihood.item())          

        # Get descriptive statistics of the training log likelihood, the training accuracy and the KL over MC_sample                       
        # Store the descriptive statistics to display the learning behavior 
        avg_train_log_likelihood.append( np.average(train_log_likelihood) )
        avg_train_acc.append(np.average(train_acc))
                
        # print training/validation statistics 
        pbar.set_postfix(train_log_likelihood=avg_train_log_likelihood[-1], acc=avg_train_acc[-1])
        
        # clear lists to track the monte carlo estimation for the next epoch
        train_log_likelihood = []  
        train_acc = []                     
        # if epoch % 20 == 0:
        #   print("Saving model at epoch ", epoch)
        #   torch.save(posterior_model.state_dict(), './'+'{}_state_{}.pt'.format(model.name, epoch))              
                  
    return avg_train_log_likelihood


