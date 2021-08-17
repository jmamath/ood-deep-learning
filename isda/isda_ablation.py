# -*- coding: utf-8 -*-
"""
Created on Tue Aug 10 08:43:21 2021

@author: JeanMichelAmath
"""


import augmentations
import torch
import torch.nn.functional as F
from torchvision import datasets
import torch.nn as nn
from torchvision import transforms
import numpy as np
from isda import ISDALoss, ISDALossFull, ISDALossPosNeg
from isda_utils import train_isda, Full_layer
from resnet import CifarResNet, BasicBlock

def get_device():
    if torch.cuda.is_available():
        device = 'cuda:0'
    else:
        device = 'cpu'
    return device
device = get_device()

CORRUPTIONS = [
    'gaussian_noise', 'shot_noise', 'impulse_noise', 'defocus_blur',
    'glass_blur', 'motion_blur', 'zoom_blur', 'snow', 'frost', 'fog',
    'brightness', 'contrast', 'elastic_transform', 'pixelate',
    'jpeg_compression'
]

# Load datasets
preprocess = transforms.Compose(
      [transforms.ToTensor(),
       transforms.Normalize([0.5] * 3, [0.5] * 3)])

preprocess_simple = transforms.Compose(
      [transforms.ToTensor()])

train_data = datasets.CIFAR10('../data/cifar', train=True, transform=preprocess, download=False)
test_data = datasets.CIFAR10('../data/cifar', train=False, transform=preprocess, download=False)

train_loader = torch.utils.data.DataLoader(
      train_data,
      batch_size=128,
      shuffle=True)

test_loader = torch.utils.data.DataLoader(
      test_data,
      batch_size=128,
      shuffle=True)

base_c_path = '../data/CIFAR-10-C/'

def test(net, test_loader):
  """Evaluate network on given dataset."""
  net.eval()
  total_loss = 0.
  total_correct = 0
  with torch.no_grad():
    for images, targets in test_loader:
      images, targets = images.to(device), targets.to(device)
      logits = net(images)
      loss = F.cross_entropy(logits, targets)
      pred = logits.data.max(1)[1]
      total_loss += float(loss.data)
      total_correct += pred.eq(targets.data).sum().item()

  return total_loss / len(test_loader.dataset), total_correct / len(
      test_loader.dataset)


def test_c(net, test_data, base_path):
  """Evaluate network on given corrupted dataset."""
  corruption_accs = []
  for corruption in CORRUPTIONS:
    # Reference to original data is mutated
    test_data.data = np.load(base_path + corruption + '.npy')
    test_data.targets = torch.LongTensor(np.load(base_path + 'labels.npy'))

    test_loader = torch.utils.data.DataLoader(
        test_data,
        batch_size=128,
        shuffle=True)

    test_loss, test_acc = test(net, test_loader)
    corruption_accs.append(test_acc)
    print('{}\n\tTest Loss {:.3f} | Test Error {:.3f}'.format(
        corruption, test_loss, 100 - 100. * test_acc))

  return np.mean(corruption_accs)

def evaluate_loss_function(nb_evaluation,criterion, device, epochs, train_loader, test_loader, test_data, CE=False):
    iid_acc = []
    ood_acc = []
    for i in range(nb_evaluation):
        model = CifarResNet(BasicBlock, [1,1,1]).to(device)
        feature_nb = model.fc.in_features
        fc = Full_layer(feature_nb, class_num=10).to(device)
        # we empty the last layer with a sequential layer doing nothing (pass-through)
        model.fc = nn.Sequential() 
        
        lr_steps = epochs * len(train_loader)
        optimizer = torch.optim.SGD([{'params': model.parameters()},
                                        {'params': fc.parameters()}],
                                        lr=0.2,
                                        momentum=0.9,
                                        weight_decay=5e-4)
        
        scheduler = torch.optim.lr_scheduler.CyclicLR(optimizer, base_lr=0, max_lr=0.2,
                    step_size_up=lr_steps / 2, step_size_down=lr_steps / 2)
        
        loss_isda = []
        acc_isda = []
        for epoch in range(0, epochs):
            # train for one epoch
            if CE:
                loss, acc = train_isda(train_loader, model, fc, criterion, optimizer, epoch, device, scheduler, loss_isda=False)
            else:
                loss, acc = train_isda(train_loader, model, fc, criterion, optimizer, epoch, device, scheduler)
            loss_isda.append(loss.ave)
            acc_isda.append(acc.ave)
        
        model.fc = fc
        iid_acc.append(test(model, test_loader))
        ood_acc.append(test_c(model, test_data, base_c_path))
    return np.array(iid_acc), np.array(ood_acc)



epoch=15
# device='cpu'
nb_evaluation = 1

ce_criterion = nn.CrossEntropyLoss()
print('CE Loss')
ce_acc, ce_ood_acc = evaluate_loss_function(nb_evaluation, ce_criterion, device, epoch, train_loader, test_loader, test_data, CE=True)

isda_loss = ISDALoss(64, class_num=10)
print('ISDA Loss')
isda_acc, isda_ood_acc = evaluate_loss_function(nb_evaluation, isda_loss, device, epoch, train_loader, test_loader, test_data)

isda_loss_full = ISDALossFull(64, class_num=10, rank=[0,1,2,3], use_mu=True)
print('ISDA Full Rank 1 - 3')
isda_full_acc, isda_full_ood_acc = evaluate_loss_function(nb_evaluation, isda_loss_full, device, epoch, train_loader, test_loader, test_data)

isda_loss_pn = ISDALossPosNeg(64, class_num=10, rank=[0,1,2,3])
print('ISDA Pos Neg Rank 1-3')
isda_loss_pn_acc, isda_loss_pn_ood_acc = evaluate_loss_function(nb_evaluation, isda_loss_pn, device, epoch, train_loader, test_loader, test_data)

isda_loss_pn_full = ISDALossPosNeg(64, class_num=10, rank=[0,1,2,3], use_mu=True)
print('ISDA Pos Neg Full Rank 1-3')
isda_loss_pn_acc, isda_loss_pn_ood_acc = evaluate_loss_function(nb_evaluation, isda_loss_pn_full, device, epoch, train_loader, test_loader, test_data)


print('-------------------')
print('Cifar-10-Acc')
print('CE : {} +- {}'.format(ce_acc.mean(0)[1], ce_acc.std(0)[1]))
print('ISDA: {} +- {}'.format(isda_acc.mean(0)[1], isda_acc.std(0)[1]))
print('ISDA PosNeg : {} +- {}'.format(isda_loss_pn_acc.mean(0)[1], isda_loss_pn_acc.std(0)[1]))
print('ISDA Full {} +- {}: '.format(isda_full_acc.mean(0)[1], isda_full_acc.std(0)[1]))
print('\n')
print('Cifar-10-C Acc')
print('CE : {} +- {}'.format(ce_ood_acc.mean(), ce_ood_acc.std()))
print('ISDA: {} +- {}'.format(isda_ood_acc.mean(), isda_ood_acc.std()))
print('ISDA PosNeg : {} +- {}'.format(isda_loss_pn_ood_acc.mean(), isda_loss_pn_ood_acc.std()))
print('ISDA Full : {} +- {}'.format(isda_full_ood_acc.mean(), isda_full_ood_acc.std()))
print('-------------------')

