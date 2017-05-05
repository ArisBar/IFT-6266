 #!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 22 18:53:53 2017

@author: aristidebaratin
"""

from six.moves import cPickle as pkl
import numpy as np
import os, sys
import glob
import PIL
from PIL import Image
#from skimage.transform import resize
from tqdm import tqdm
from collections import OrderedDict
import theano
import theano.tensor as T

data_path = "/u/baratina/Documents/IFT6266/Datasets/inpainting"
train_path = "train2014"
val_path= "val2014"
caption = "dict_key_imgID_value_caps_train_and_valid.pkl"



train_data_path = os.path.join(data_path, train_path)
val_data_path = os.path.join(data_path, val_path)
caption_path = os.path.join(data_path, caption)

new_train_path = os.path.join(data_path, "train2014.pkl")
new_val_path = os.path.join(data_path, "val2014.pkl")

# Remove the grayscale images; save the RGB images into pkl files
def pkl_images():
    '''
    Save RGB images as ordered dictionary  into pkl files
    Keys: image IDs
    Items: images converted to numpy arrays

    '''
    dict_train = {}
    dict_val = {}
    imgs_train = glob.glob(train_data_path + "/*.jpg")
    imgs_val = glob.glob(val_data_path + "/*.jpg")

    print("looping over training images...")
    for img_path in tqdm(imgs_train):
        img = Image.open(img_path)
        img_array = np.array(img)
        if len(img_array.shape) == 3:
            cap_id = os.path.basename(img_path)[:-4]
            dict_train[cap_id] = img_array

    print("looping over validation images...")
    for img_path in tqdm(imgs_val):
        img = Image.open(img_path)
        img_array = np.array(img)
        if len(img_array.shape) == 3:
            cap_id = os.path.basename(img_path)[:-4]
            dict_val[cap_id] = img_array

    dict_train = OrderedDict(sorted(dict_train.items(), key=lambda t: t[0]))
    dict_val = OrderedDict(sorted(dict_val.items(), key=lambda t: t[0]))

    new_train_path = os.path.join(data_path, "train2014.pkl")
    new_val_path = os.path.join(data_path, "val2014.pkl")

    with open(new_train_path, 'wb') as train:
        pkl.dump(dict_train, train, protocol=pkl.HIGHEST_PROTOCOL)

    with open(new_val_path, 'wb') as val:
        pkl.dump(dict_val, val, protocol=pkl.HIGHEST_PROTOCOL)



def load_caption():
    '''
    load the image captions into an ordered dictionary (keys: imageID)
    '''

    with open(caption, 'rb') as fd:
        caption_dict = pkl.load(fd)
    return caption_dict


def load_images(n='all'):
    '''
    load the images into numpy arrays
    '''

    with open(new_train_path, 'rb') as tr:
        train_set = pkl.load(tr)
    with open(new_val_path, 'rb') as val:
        valid_set = pkl.load(val)

    if n=='all':
        tr = np.concatenate([im[None,...] for im in list(train_set.values())], axis=0)
        val = np.concatenate([im[None,...] for im in list(valid_set.values())], axis=0)
    else:
        tr = np.concatenate([im[None,...] for im in list(train_set.values())[:n]], axis=0)
        val = np.concatenate([im[None,...] for im in list(valid_set.values())[:n]], axis=0)

    return tr, val

if __name__ == '__main__':
    pkl_images()
