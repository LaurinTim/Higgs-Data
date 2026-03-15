import tensorflow as tf, numpy as np, pandas as pd
import os, sys
from pathlib import Path
import torch
from torch import nn
import matplotlib.pyplot as plt
import json, glob

data_dir = str(Path(__file__).resolve().parent)

# %%
    
def plot_training_info(train_loss, valid_loss, train_auc, valid_auc, n=300) -> None:
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























































































