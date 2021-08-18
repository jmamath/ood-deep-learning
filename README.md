# ood-deep-learning
Repository for the project proposal "A Thorough Analysis of Deep Neural Networks under Distribution Shift"

Note that to run the experiments you have to have a data folder with CIFAR-10 that you can download by specifying download=True, in the *fight_* script.

## Download CIFAR-10-C
```
mkdir -p ./data/cifar
curl -O https://zenodo.org/record/2535967/files/CIFAR-10-C.tar
tar -xvf CIFAR-10-C.tar -C data/cifar/
```

## Requirements
```
torch==1.7.1
numpy==1.19.2
matplotlib==3.3.2
torchvision==0.8.2
tqdm==4.50.2
Pillow==8.3.1
```

## Contributing
As you try to implement a new method follow the following guideline:
clone the repo and modify the *fight_* script, then add a comenting line after the import
``` ######################## NEW METHOD ######################## 
# 1. load data and create dataloaders
# 2. instantiate a model
# 3. train (you can create your own custom training loss if required)
# 4. evaluate on the iid and ood test set
```
