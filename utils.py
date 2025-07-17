import tensorflow as tf, numpy as np, pandas as pd
import os, sys
from pathlib import Path
import torch
from torch import nn
import matplotlib.pyplot as plt
import importlib.util

data_dir = str(Path(__file__).resolve().parent)

hspec = importlib.util.spec_from_file_location("utils", data_dir + '\\HIGGS\\HIGGS_utils.py')
uh = importlib.util.module_from_spec(hspec)
hspec.loader.exec_module(uh)

sspec = importlib.util.spec_from_file_location("utils", data_dir + '\\SUSY\\SUSY_utils.py')
us = importlib.util.module_from_spec(sspec)
sspec.loader.exec_module(us)

# %%

def accuracy(corr, pred):
    return 1 - np.sum(np.abs(corr - pred))/len(corr)

def get_HIGGS_labels():
    train_files = tf.io.gfile.glob(data_dir + '\\HIGGS data\\training' + '\\*.tfrecord')
    valid_files = tf.io.gfile.glob(data_dir + '\\HIGGS data\\validation' + '\\*.tfrecord')
    
    training_size = int(1.05e7)
    validation_size = int(5e5)
    batch_size = int(5e5)
    
    total_train_steps = training_size / batch_size
    total_valid_steps = validation_size / batch_size
    
    ds_train = uh.make_ds(train_files, batch=batch_size, shuffle=False)
    ds_train_np = ds_train.as_numpy_iterator()

    ds_valid = us.make_ds(valid_files, batch=batch_size, shuffle=False)
    ds_valid_np = ds_valid.as_numpy_iterator()
    
    train_labels = []
    valid_labels = []
    
    for train_step, (features, labels) in enumerate(ds_train_np):
        if train_step == total_train_steps:
            break
        
        train_labels.append(labels)
        
    for valid_step, (features, labels) in enumerate(ds_valid_np):
        if valid_step == total_valid_steps:
            break
        
        valid_labels.append(labels)
    
    return train_labels, valid_labels

def get_SUSY_labels():
    train_files = tf.io.gfile.glob(data_dir + '\\SUSY\\SUSY data\\training' + '\\*.tfrecord')
    valid_files = tf.io.gfile.glob(data_dir + '\\SUSY\\SUSY data\\validation' + '\\*.tfrecord')
    
    training_size = int(4.5e6)
    validation_size = int(5e5)
    batch_size = int(5e5)
    
    total_train_steps = training_size / batch_size
    total_valid_steps = validation_size / batch_size
    
    ds_train = uh.make_ds(train_files, batch=batch_size, shuffle=False)
    ds_train_np = ds_train.as_numpy_iterator()

    ds_valid = us.make_ds(valid_files, batch=batch_size, shuffle=False)
    ds_valid_np = ds_valid.as_numpy_iterator()
    
    train_labels = []
    valid_labels = []
    
    for train_step, (features, labels) in enumerate(ds_train_np):
        if train_step == total_train_steps:
            break
        
        train_labels.append(labels)
        
    for valid_step, (features, labels) in enumerate(ds_valid_np):
        if valid_step == total_valid_steps:
            break
        
        valid_labels.append(labels)
    
    return train_labels, valid_labels