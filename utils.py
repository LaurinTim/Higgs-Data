import tensorflow as tf, numpy as np, pandas as pd
import os, sys
from pathlib import Path
import torch
from torch import nn
import matplotlib.pyplot as plt

data_dir = str(Path(__file__).resolve().parent)

# %%

def make_decoder(feature_description):
    def decoder(example):
        example = tf.io.parse_single_example(example, feature_description)
        features = tf.io.parse_tensor(example['features'], tf.float32)
        features = tf.reshape(features, [28])
        label = example['label']
        return features, label
    return decoder

def load_dataset(filenames, decoder, ordered=False):
    AUTO = tf.data.experimental.AUTOTUNE
    ignore_order = tf.data.Options()
    if not ordered:
        ignore_order.experimental_deterministic = False
    dataset = (
        tf.data
        .TFRecordDataset(filenames, num_parallel_reads=AUTO)
        .with_options(ignore_order)
        .map(decoder, AUTO)
    )
    return dataset

class EarlyStopping:
    """Stops training when a monitored metric has stopped improving."""
    def __init__(self, patience=7, min_delta=0, path='checkpoint.pt'):
        """
        Args:
            patience (int): How long to wait after last time the monitored metric improved.
                            Default: 7
            min_delta (float): Minimum change in the monitored metric to qualify as an improvement.
                               Default: 0
            path (str): Path to save the best model file.
                        Default: 'checkpoint.pt'
        """
        self.patience = patience
        self.min_delta = min_delta
        self.path = str(Path(__file__).resolve().parent) + '\\EarlyStopping model\\' + path
        self.counter = 0
        self.best_loss = None
        self.early_stop = False

    def __call__(self, val_loss, model):
        if self.best_loss is None:
            self.best_loss = val_loss
            self.save_checkpoint(val_loss, model)
        elif val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.save_checkpoint(val_loss, model)
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True

    def save_checkpoint(self, val_loss, model):
        '''Saves model when validation loss decrease.'''
        torch.save(model.state_dict(), self.path)
        
def count_samples(files):
    '''
    Get the number of samples in files.

    Parameters
    ----------
    files : list
        List containing the paths to the files in which the samples should 
        be counted.

    Returns
    -------
    n : int
        Number of samples found in the files.

    '''
    feature_description = {
        'features': tf.io.FixedLenFeature([], tf.string),
        'label': tf.io.FixedLenFeature([], tf.float32),
    }
    decoder = make_decoder(feature_description)
    ds = load_dataset(files, decoder, ordered=False).cache()
    n = sum(1 for _ in ds)   # ~0.5 s per million examples
    
    return n

class DenseBlock(nn.Module):
    def __init__(self, input_size, output_size, activation, dropout_rate=0.1):
        super().__init__()
        self.stack = nn.Sequential(
            nn.Linear(input_size, output_size),
            nn.BatchNorm1d(num_features=output_size),
            activation,
            nn.Dropout(p=dropout_rate)
        )
        
    def forward(self, x):
        logits = self.stack(x)
        return logits
    
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
    























































































