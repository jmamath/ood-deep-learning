# ood-deep-learning
Repository for the project proposal "A Thorough Analysis of Deep Neural Networks under Distribution Shift"

Note that to run the experiments you have to have a data folder with CIFAR-10 that you can download by specifying download=True, in the *fight_* script.

#### Download CIFAR-10-C
```
mkdir -p ./data/cifar
curl -O https://zenodo.org/record/2535967/files/CIFAR-10-C.tar
tar -xvf CIFAR-10-C.tar -C data/cifar/
```

