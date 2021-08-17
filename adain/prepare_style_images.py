# -*- coding: utf-8 -*-
"""
Created on Tue Aug 17 08:51:24 2021

@author: JeanMichelAmath

The goal of this script is to uniformise the dimension of style images so that they match the same dimension of 
the training data. Indeed, style transfer is done during training at the batch level. For each batch of data
a batch of style image is selected and they are merged into the training data. To achieve that we need a folder
of formated style images 

art images are downloaded from the wikiart dataset: https://www.kaggle.com/c/painter-by-numbers/data
"""

import os
import shutil # Moving file to different folders.
from pathlib import Path
import random
 
# Improting Image class from PIL module 
from PIL import Image 

"""
We suppose that all the style images are in the style/art folder, then we organize a new folder
style_set
"""

current_dir = os.getcwd()
style_dir = Path(os.path.join(current_dir,'style'))
art_dir = Path(os.path.join(current_dir,'style/art'))

l = [f for f in os.listdir(art_dir)] 

random.shuffle(l)
num_envs = len(l)
number_train_envs = int((num_envs*.7))        
train_envs = l[:number_train_envs]



"""
We create a new directory for our style images. If the folders already exist
then we remove all images within, this is in case we change the dataset of images to train on.
Indeed for instance in the case of cifar we resize the images into (32,32) but in case 
the images shape where different we would resize accordingly.

The folder is style_set/style_set because it is the format accepted by torchvision.datasets.ImageFolder
"""
dirname = Path(os.path.join(style_dir, 'style_set/style_set'))
if os.path.exists(dirname):
    for filename in os.listdir(art_dir):
        filepath = os.path.join(art_dir, filename)
        try:
            shutil.rmtree(filepath)
        except OSError:
            os.remove(filepath)
    print(dirname,"cleared")
else:
    os.makedirs(dirname )

style_folder = Path(os.path.join(style_dir, dirname ))
    

"""
Now we resize images with pil, it might exhaust cpu resources while performing it. Finally we save them in 
the relevant folder.
"""

size=(32,32) 
 
for file in train_envs:
    #import pdb; pdb.set_trace()
    im = Image.open((Path(os.path.join(art_dir,file))))
    #resize image
    out = im.resize(size)
    out = out.convert('RGB')  
    #save resized image
    out.save(os.path.join(style_folder, file))
    #shutil.move(os.path.join(art_dir, file), meta_train_folder)
    
 


