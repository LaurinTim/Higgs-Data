import tensorflow as tf, numpy as np, pandas as pd
import os, sys
from pathlib import Path
import torch
from torch import nn
import matplotlib.pyplot as plt
import importlib.util
import json, glob
from tqdm import tqdm

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
    train_files = tf.io.gfile.glob(data_dir + '\\HIGGS\\HIGGS data\\training' + '\\*.tfrecord')
    valid_files = tf.io.gfile.glob(data_dir + '\\HIGGS\\HIGGS data\\validation' + '\\*.tfrecord')
    
    training_size = int(1.05e7)
    validation_size = int(5e5)
    train_batch_size = int(1.05e7)
    valid_batch_size = int(5e5)
    
    total_train_steps = training_size / train_batch_size
    total_valid_steps = validation_size / valid_batch_size
    
    ds_train = uh.make_ds(train_files, batch=train_batch_size, shuffle=False)
    ds_train_np = ds_train.as_numpy_iterator()

    ds_valid = uh.make_ds(valid_files, batch=valid_batch_size, shuffle=False)
    ds_valid_np = ds_valid.as_numpy_iterator()
    
    train_labels = []
    valid_labels = []
    
    #print(type(next(iter(ds_train_np))))
    
    for train_step, (features, labels) in enumerate(ds_train_np):
        if train_step == total_train_steps:
            break
        
        train_labels.extend(labels)
        
    for valid_step, (features, labels) in enumerate(ds_valid_np):
        if valid_step == total_valid_steps:
            break
        
        valid_labels.extend(labels)
    
    return train_labels, valid_labels

def get_SUSY_labels():
    train_files = tf.io.gfile.glob(data_dir + '\\SUSY\\SUSY data\\training' + '\\*.tfrecord')
    valid_files = tf.io.gfile.glob(data_dir + '\\SUSY\\SUSY data\\validation' + '\\*.tfrecord')
    
    training_size = int(4.5e6)
    validation_size = int(5e5)
    train_batch_size = int(4.5e6)
    valid_batch_size = int(5e5)
    
    total_train_steps = training_size / train_batch_size
    total_valid_steps = validation_size / valid_batch_size
    
    ds_train = us.make_ds(train_files, batch=train_batch_size, shuffle=False)
    ds_train_np = ds_train.as_numpy_iterator()

    ds_valid = us.make_ds(valid_files, batch=valid_batch_size, shuffle=False)
    ds_valid_np = ds_valid.as_numpy_iterator()
    
    train_labels = []
    valid_labels = []
    
    for train_step, (features, labels) in enumerate(ds_train_np):
        if train_step == total_train_steps:
            break
        
        train_labels.extend(labels)
        
    for valid_step, (features, labels) in enumerate(ds_valid_np):
        if valid_step == total_valid_steps:
            break
        
        valid_labels.extend(labels)
    
    return train_labels, valid_labels


def plot_training_info(train_df, valid_df, n=300) -> None:
    train_loss = train_df.loss_history.tolist()
    train_auc = train_df.auc_history.tolist()
    valid_loss = valid_df.loss_history.tolist()
    valid_auc = valid_df.auc_history.tolist()
    
    total_epochs = len(valid_loss)
    
    train_loss_truncated = np.array(train_loss[:(len(train_loss) - (len(train_loss) % n))]).reshape(-1, n).mean(axis=1)
    train_auc_truncated = np.array(train_auc[:(len(train_auc) - (len(train_auc) % n))]).reshape(-1, n).mean(axis=1)

    x_train = np.linspace(0, total_epochs-1, len(train_loss_truncated))
    x_valid = np.linspace(0, total_epochs-1, total_epochs)

    plt.figure(figsize=(15,8))

    plt.plot(x_train, train_loss_truncated, c='k', label='Training loss')
    plt.plot(x_valid, valid_loss, c='r', linestyle='--', label='Validation loss')

    plt.legend(loc='best')
    plt.show()
    
    plt.figure(figsize=(15,8))

    plt.plot(x_train, train_auc_truncated, c='k', label='Training auc')
    plt.plot(x_valid, valid_auc, c='r', linestyle='--', label='Validation auc')

    plt.legend(loc='best')
    plt.show()