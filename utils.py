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


def plot_training_info(train_df, valid_df, n=300, start_epoch=0) -> None:
    train_loss = train_df.loss_history.tolist()
    train_auc = train_df.auc_history.tolist()
    valid_loss = valid_df.loss_history.tolist()
    valid_auc = valid_df.auc_history.tolist()
    
    total_epochs = len(valid_loss)
    
    train_steps_per_epoch = int(len(train_loss) / total_epochs)
    
    if n == "epoch":
        n = train_steps_per_epoch
    
    if start_epoch > 0:
        train_loss = train_loss[start_epoch * train_steps_per_epoch:]
        train_auc = train_auc[start_epoch * train_steps_per_epoch:]
        valid_loss = valid_loss[start_epoch:]
        valid_auc = valid_auc[start_epoch:]
    
    train_loss_truncated = np.array(train_loss[:(len(train_loss) - (len(train_loss) % n))]).reshape(-1, n).mean(axis=1)
    train_auc_truncated = np.array(train_auc[:(len(train_auc) - (len(train_auc) % n))]).reshape(-1, n).mean(axis=1)
    
    x_train = np.linspace(start_epoch, total_epochs-1, len(train_loss_truncated))
    x_valid = np.linspace(start_epoch, total_epochs-1, total_epochs-start_epoch)
    
    fig, (ax1, ax2) = plt.subplots(nrows=1, ncols=2, figsize=(16,5))

    ax1.plot(x_train, train_loss_truncated, c='k', label='Training loss')
    ax1.plot(x_valid, valid_loss, c='r', linestyle='--', label='Validation loss')

    ax1.legend(loc='best')
    ax1.set_title("Loss per Epoch")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss")
    
    ax2.plot(x_train, train_auc_truncated, c='k', label='Training auc')
    ax2.plot(x_valid, valid_auc, c='r', linestyle='--', label='Validation auc')

    ax2.legend(loc='best')
    ax2.set_title("AUC Score per Epoch")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("AUC Score")
    
    plt.show()

def CosineAnnealing_lr(epoch, start_lr, end_lr, dec_epochs):    
    return end_lr + 1/2 * (start_lr - end_lr) * (1 + np.cos(np.pi * epoch/dec_epochs)) if epoch < dec_epochs else end_lr

def plot_func(func, x, title, legend, xlabel, ylabel, sci=True):
    
    y = func(x)
    
    fig, ax = plt.subplots(figsize=(7, 4))
    
    ax.plot(x, y, c='k', label=legend, linewidth=2)
    
    ax.legend(loc='best')
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    
    if sci:
        ax.ticklabel_format(axis="y", style="sci", scilimits=(0,0))
    
    plt.show()
    
def plot_data(x, y, title, legend, xlabel, ylabel, sci=True):
        
    fig, ax = plt.subplots(figsize=(7, 4))
    
    ax.plot(x, y, c='k', label=legend, linewidth=2)
    
    ax.legend(loc='best')
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    
    if sci:
        ax.ticklabel_format(axis="y", style="sci", scilimits=(0,0))
    
    plt.show()
    
def roc_curve(y, p):
    
    y = y.astype(bool) if y.dtype != bool else y
        
    desc_score_indices = np.argsort(-p)
    sorted_y = y[desc_score_indices]
    sorted_p = p[desc_score_indices]

    
    tp = np.cumsum(sorted_y)
    fp = np.cumsum(~sorted_y)
    
    p_total = tp[-1]
    p_total = fp[-1]
    
    distinct_value_indices = np.where(np.diff(sorted_p))[0]
    threshold_idxs = np.r_[distinct_value_indices, sorted_y.size - 1]
    
    tpr = tp[threshold_idxs] / p_total
    fpr = fp[threshold_idxs] / p_total
    
    return fpr, tpr

def auc(y, p):
    fpr, tpr = roc_curve(y, p)
    
    tpr_diff = tpr[1:] - tpr[:-1]
    fpr_diff = fpr[1:] - fpr[:-1]
    
    auc_rect_arr = tpr[:-1] * fpr_diff
    auc_tri_arr = tpr_diff/2 * fpr_diff
        
    return float(np.sum(auc_rect_arr + auc_tri_arr))

def auc_from_roc(fpr, tpr):
    
    tpr_diff = tpr[1:] - tpr[:-1]
    fpr_diff = fpr[1:] - fpr[:-1]
    
    auc_rect_arr = tpr[:-1] * fpr_diff
    auc_tri_arr = tpr_diff/2 * fpr_diff
        
    return float(np.sum(auc_rect_arr + auc_tri_arr))



















































































