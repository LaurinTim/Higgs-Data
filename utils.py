import tensorflow as tf, numpy as np, pandas as pd
import os, sys
from pathlib import Path
import torch

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
        
def count_data(files):
    feature_description = {
        'features': tf.io.FixedLenFeature([], tf.string),
        'label': tf.io.FixedLenFeature([], tf.float32),
    }
    decoder = make_decoder(feature_description)
    ds = load_dataset(files, decoder, ordered=False).cache()
    n = sum(1 for _ in ds)   # ~0.5 s per million examples
    
    return n

























































































