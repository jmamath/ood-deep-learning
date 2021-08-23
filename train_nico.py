from architectures.resnet import CifarResNet, BasicBlock
import augmix.augmentations
import torch
from torch.utils.data import DataLoader
import torch.nn.functional as F
from torchvision import datasets
from torchvision import transforms
import numpy as np
import torch.nn as nn
from isda.isda import ISDALoss, ISDALossFull, ISDALossPosNeg
from isda.isda_utils import train_isda
import matplotlib.pyplot as plt
from pathlib import Path
import torchvision
from isda.isda_utils import Full_layer
from architectures.lenet import EnsembleNeuralNet
from training_loop_utils import train_model

# !!!!!!!!!!!!!
# !!!!!!!!!!!!!
# !!!!!!!!!!!!!
# Some things you should read before running this file:
# You need to wget -c http://nico.thumedialab.com/dataset/NICO/Animal.zip and extract in current dir
# Then you can run this file, which includes some code below to reorganize the data for torch dataloader
# The code below reorganizes the NICO dataset in the form of root/class/image.jpg
import os
import shutil
path = "./Animal"
# I need to check whether the dataset has already been processed
if len(os.listdir(path)) < 20:
    for i in os.listdir(path):
        t = os.path.join(path,i)
        if not os.path.isdir(t):
            os.remove(t)
        else:
            for j in os.listdir(t):
                temp = os.path.join(path,i,j)
                os.mkdir(t+"_"+j.strip().replace(" ",""))
                for k in os.listdir(temp):
                    shutil.move(os.path.join(temp,k), os.path.join(t+"_"+j.strip().replace(" ",""), k))
                os.rmdir(temp)
            os.rmdir(t)


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

from augmix.augmix_utils import test_c, AugMixDataset, train_augmix, test

train_transform = transforms.Compose(
    [transforms.RandomHorizontalFlip(),
     transforms.RandomCrop(32, padding=4)])
preprocess = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize([0.5] * 3, [0.5] * 3)])
test_transform = preprocess

n_epochs = 15

######################## TRAIN AUGMIX ########################
# Load datasets

train_data = datasets.ImageFolder(path, transform=train_transform)
test_data = datasets.ImageFolder(path, transform=test_transform)

# the specifics of AugMix happens here, in the custom dataset but also later in the train_augmix function
# where the mixture is computed
train_data = AugMixDataset(train_data, preprocess, no_jsd=False)
train_loader = torch.utils.data.DataLoader(
    train_data,
    batch_size=128,
    shuffle=True)

test_loader = torch.utils.data.DataLoader(
    test_data,
    batch_size=128,
    shuffle=True)

base_c_path = '../data/CIFAR-10-C/'

print('TRAIN AUGMIX')
model = CifarResNet(BasicBlock, [1, 1, 1]).to(device)
# model.fc = nn.Sequential()

lr_steps = n_epochs * len(train_loader)
optimizer = torch.optim.SGD(model.parameters(), lr=0.2, momentum=0.9, weight_decay=5e-4)
scheduler = torch.optim.lr_scheduler.CyclicLR(optimizer, base_lr=0, max_lr=0.2,
                                              step_size_up=lr_steps / 2, step_size_down=lr_steps / 2)

augmix_loss = []
for epoch in range(n_epochs):
    # begin_time = time.time()
    train_loss_ema = train_augmix(model, train_loader, optimizer, scheduler, no_jsd=False)
    test_loss, test_acc = test(model, test_loader)
    augmix_loss.append(train_loss_ema)
print('Epoch {} - AugMix loss: {} - Test loss: {} - Test Acc: {}'.format(epoch, train_loss_ema, test_loss, test_acc))

test_acc_augmix = test(model, test_loader)
test_c_acc_augmix = test_c(model, test_data, base_c_path)

######################## TRAIN ISDA, ISDA FULL, ISDA POSNEG, CROSS-ENTROPY ########################

train_data = datasets.CIFAR10('../data/cifar', train=True, transform=preprocess, download=False)
test_data = datasets.CIFAR10('../data/cifar', train=False, transform=test_transform, download=False)

train_loader = torch.utils.data.DataLoader(
    train_data,
    batch_size=128,
    shuffle=True)

test_loader = torch.utils.data.DataLoader(
    test_data,
    batch_size=128,
    shuffle=True)


def evaluate_loss_function(criterion, device, epochs, train_loader, test_loader, test_data, CE=False):
    model = CifarResNet(BasicBlock, [1, 1, 1]).to(device)
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
            loss, acc = train_isda(train_loader, model, fc, criterion, optimizer, epoch, device, scheduler,
                                   loss_isda=False)
        else:
            loss, acc = train_isda(train_loader, model, fc, criterion, optimizer, epoch, device, scheduler)
        loss_isda.append(loss.ave)
        acc_isda.append(acc.ave)

    model.fc = fc
    iid_acc = test(model, test_loader)
    ood_acc = test_c(model, test_data, base_c_path)
    return iid_acc, ood_acc


ce_criterion = nn.CrossEntropyLoss()
print('CE Loss')
ce_acc, ce_ood_acc = evaluate_loss_function(ce_criterion, device, n_epochs, train_loader, test_loader, test_data,
                                            CE=True)

isda_loss = ISDALoss(64, class_num=10)
print('ISDA Loss')
isda_acc, isda_ood_acc = evaluate_loss_function(isda_loss, device, n_epochs, train_loader, test_loader, test_data)

isda_loss_full = ISDALossFull(64, class_num=10, rank=[0, 1, 2, 3], use_mu=True)
print('ISDA Full Rank 1 - 3')
isda_full_acc, isda_full_ood_acc = evaluate_loss_function(isda_loss_full, device, n_epochs, train_loader, test_loader,
                                                          test_data)

isda_loss_pn = ISDALossPosNeg(64, class_num=10, rank=[0, 1, 2, 3])
print('ISDA Pos Neg Rank 1-3')
isda_loss_pn_acc, isda_loss_pn_ood_acc = evaluate_loss_function(isda_loss_pn, device, n_epochs, train_loader,
                                                                test_loader, test_data)

######################## TRAIN  ########################

train_data = datasets.CIFAR10('../data/cifar', train=True, transform=preprocess, download=False)
test_data = datasets.CIFAR10('../data/cifar', train=False, transform=test_transform, download=False)

train_loader = torch.utils.data.DataLoader(
    train_data,
    batch_size=128,
    shuffle=True)

test_loader = torch.utils.data.DataLoader(
    test_data,
    batch_size=128,
    shuffle=True)

print('TRAIN DEEP ENSEMBLES')


## Train an ensemble of NN
def train_ensemble(N, n_epochs, trainLoader, no_classes):
    # Here our goal is to train a neural net N times from scratch, and return the list of trained neural net
    ensembles = []
    for i in range(N):
        model = CifarResNet(BasicBlock, [1, 1, 1]).to(device)
        lr_steps = n_epochs * len(train_loader)
        optimizer = torch.optim.SGD(model.parameters(), lr=0.2, momentum=0.9, weight_decay=5e-4)
        scheduler = torch.optim.lr_scheduler.CyclicLR(optimizer, base_lr=0, max_lr=0.2,
                                                      step_size_up=lr_steps / 2, step_size_down=lr_steps / 2)
        crit = nn.CrossEntropyLoss()
        _ = train_model(model, trainLoader, n_epochs, crit, optimizer, no_classes, device, scheduler)
        ensembles.append(model)
    return ensembles


ensembles = train_ensemble(5, n_epochs, train_loader, no_classes=10)
ensemble_nn = EnsembleNeuralNet(ensembles)

ens_acc = test(ensemble_nn, test_loader)
ens_ood_acc = test_c(ensemble_nn, test_data, base_c_path)

######################## TRAIN FAST GRADIENT SIGN METHOD ########################
print('TRAIN FAST GRADIENT SIGN METHOD')
model = CifarResNet(BasicBlock, [1, 1, 1]).to(device)
lr_steps = n_epochs * len(train_loader)
optimizer = torch.optim.SGD(model.parameters(), lr=0.2, momentum=0.9, weight_decay=5e-4)
scheduler = torch.optim.lr_scheduler.CyclicLR(optimizer, base_lr=0, max_lr=0.2,
                                              step_size_up=lr_steps / 2, step_size_down=lr_steps / 2)
crit = nn.CrossEntropyLoss()

_ = train_model(model, train_loader, n_epochs, crit, optimizer, 10, device, scheduler, fgsm=True)

fgsm_acc = test(model, test_loader)
fgsm_ood_acc = test_c(model, test_data, base_c_path)

######################## TRAIN MIXUP ########################
print('TRAIN MIXUP')
model = CifarResNet(BasicBlock, [1, 1, 1]).to(device)
lr_steps = n_epochs * len(train_loader)
optimizer = torch.optim.SGD(model.parameters(), lr=0.2, momentum=0.9, weight_decay=5e-4)
scheduler = torch.optim.lr_scheduler.CyclicLR(optimizer, base_lr=0, max_lr=0.2,
                                              step_size_up=lr_steps / 2, step_size_down=lr_steps / 2)


def mixup_log_loss(prediction, label):
    loss = nn.LogSoftmax()
    log_loss = loss(prediction) * label
    return -log_loss.mean()


crit = mixup_log_loss

_ = train_model(model, train_loader, n_epochs, crit, optimizer, 10, device, scheduler, mixup=True)

mixup_acc = test(model, test_loader)
mixup_ood_acc = test_c(model, test_data, base_c_path)

######################## TRAIN SYNTHETIC AUGMENTATION ########################
print('TRAIN SYNTHETIC AUGMENTATION')
transf = transforms.ToTensor()  # Turn PIL Image to torch.Tensor

current_dir = os.getcwd()
style_set_path = Path(os.path.join(current_dir, 'adain/style/style_set/'))

style_dataset = torchvision.datasets.ImageFolder(style_set_path, transform=transf)
style_loader = DataLoader(style_dataset, batch_size=128, shuffle=True, )

model = CifarResNet(BasicBlock, [1, 1, 1]).to(device)
lr_steps = n_epochs * len(train_loader)
optimizer = torch.optim.SGD(model.parameters(), lr=0.2, momentum=0.9, weight_decay=5e-4)
scheduler = torch.optim.lr_scheduler.CyclicLR(optimizer, base_lr=0, max_lr=0.2,
                                              step_size_up=lr_steps / 2, step_size_down=lr_steps / 2)

crit = nn.CrossEntropyLoss()

_ = train_model(model, train_loader, n_epochs, crit, optimizer, 10, device, scheduler, style_loader=style_loader)

adain_acc = test(model, test_loader)
adain_ood_acc = test_c(model, test_data, base_c_path)
######################## EVALUATE RESULTS ########################

print('-------------------')
print('Cifar-10-Acc')
print('\n')
print('CE :', ce_acc[1])
print('Deep Ensembles :', ens_acc[1])
print('FGSM :', fgsm_acc[1])
print('AugMix: ', test_acc_augmix[1])
print('Mixup: ', mixup_acc[1])
print('Style Transfer: ', adain_acc[1])
print('ISDA : ', isda_acc[1])
print('ISDA_Full : ', isda_full_acc[1])
print('ISDA_PosNeg : ', isda_loss_pn_acc[1])
print('\n')
print('-------------------')
print('Cifar-10-C Acc')
print('\n')
print('CE :', ce_ood_acc)
print('Deep Ensembles :', ens_ood_acc)
print('FGSM :', fgsm_ood_acc)
print('AugMix: ', test_c_acc_augmix)
print('Mixup: ', mixup_ood_acc)
print('Style Transfer: ', adain_ood_acc)
print('ISDA : ', isda_ood_acc)
print('ISDA_Full : ', isda_full_ood_acc)
print('ISDA_PosNeg : ', isda_loss_pn_ood_acc)
print('\n')
print('-------------------')