import tensorflow as tf, numpy as np, pandas as pd
import os, sys
from pathlib import Path
import torch
from torch import nn
import matplotlib.pyplot as plt
import json, glob
from sklearn.metrics import roc_curve

data_dir = str(Path(__file__).resolve().parent)

# %%

class AsimovLoss(nn.Module):
    def __init__(self, s_tot, b_tot, sig_b_tot):
        super(AsimovLoss, self).__init__()
        self.s_tot = s_tot
        self.b_tot = b_tot
        self.sig_b_tot = sig_b_tot
        self.epsilon = 1e-8
    
    def AsimovSignificance(self, s, b, sig_b):
        term1 = (s + b) * torch.log((s + b) * (b + sig_b * sig_b) / (torch.square(b) + (s + b) * torch.square(sig_b) + self.epsilon) + self.epsilon)
        term2 = torch.square(b) * torch.log(1 + torch.square(sig_b) * s / (b * (b + torch.square(sig_b)) + self.epsilon)) / (torch.square(sig_b) + self.epsilon)
        Z = torch.sqrt(2 * (term1 - term2))
        return Z
    
    def forward(self, y_pred, y_true):
        sWeight = self.s_tot / torch.sum(y_true)
        bWeight = self.b_tot / torch.sum(1 - y_true)

        s = sWeight * torch.sum(y_pred * y_true)
        b = bWeight * torch.sum(y_pred * (1 - y_true))
        sig_b = self.sig_b_tot * b

        return 1 / self.AsimovSignificance(s, b, sig_b)

def calculate_asimov_significance(s, b, sigma_b, b_min=1):
    """
    Calculates the Asimov discovery significance (Z_A).
    
    Parameters:
        s (float or np.array): Expected signal events.
        b (float or np.array): Expected background events.
        sigma_b (float or np.array): Systematic uncertainty on background events.
        b_min (float): Minimum number of background events when calculating the asimov significance.
        
    Returns:
        Z (float or np.array): Significance in sigmas.
    """
    # Initialize Z as zeros
    if np.isscalar(b):
        if b <= 0 or sigma_b <= 0:
            return s / np.sqrt(b) if b > 0 else 0.0
    
    # Calculate terms with safe division for arrays
    # Always use at least b=b_min to avoid spikes in the significance for high thresholds
    b_safe = np.maximum(b, b_min) #np.maximum(b, 1e-9)
    
    if np.max(sigma_b) == 0:
        term = 2*((s + b_safe) * np.log(1 + s / b_safe) - s)
        Z = np.sqrt(np.maximum(term, 0))
        return Z
    
    # Add small epsilon to avoid division by 0 if sigma_b is very small
    sigma_b_safe = np.maximum(sigma_b, 1e-9)
    
    term1 = (s + b_safe) * np.log(
        ((s + b_safe) * (b_safe + sigma_b_safe**2)) / 
        (b_safe**2 + (s + b_safe) * sigma_b_safe**2)
    )
    term2 = (b_safe**2 / sigma_b_safe**2) * np.log(
        1 + (sigma_b_safe**2 * s) / (b_safe * (b_safe + sigma_b_safe**2))
    )
    
    # Z squared
    Z2 = 2 * (term1 - term2)
    
    # Handle potential negative values due to precision issues near zero
    Z = np.sqrt(np.maximum(Z2, 0))
    
    return Z

def roc_custom_thresholds(y_true, y_pred, thresholds):
    # Ensure inputs are numpy arrays
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    
    # Sort predictions for the positive and negative classes
    pos_scores = np.sort(y_pred[y_true == 1])
    neg_scores = np.sort(y_pred[y_true == 0])
    
    n_pos = len(pos_scores)
    n_neg = len(neg_scores)

    # Use searchsorted to find how many scores are BELOW each threshold
    # The number of True Positives is (Total Positives - Positives below threshold)
    tp_below_threshold = np.searchsorted(pos_scores, thresholds, side='left')
    fp_below_threshold = np.searchsorted(neg_scores, thresholds, side='left')
    
    tp = n_pos - tp_below_threshold
    fp = n_neg - fp_below_threshold
    
    # Calculate rates
    tpr = tp / n_pos if n_pos > 0 else np.zeros_like(thresholds)
    fpr = fp / n_neg if n_neg > 0 else np.zeros_like(thresholds)
    
    return fpr, tpr

def find_optimal_asimov_threshold(y_true, y_pred, weight_s=100, weight_b=1000, sys_uncertainty=0.05, b_min=1, num_thresholds=101, ret_full=True):
    """
    Scans all thresholds to find the one that maximizes discovery significance.
    
    Parameters:
        y_true (array-like): Ground truth labels (1 for signal, 0 for background).
        y_pred (array-like): Classifier probability scores.
        weight_s (float): Total expected signal events (Default: 100).
        weight_b (float): Total expected background events (Default: 1000).
        sys_uncertainty (float): Relative systematic uncertainty (Default: 0.05).
        b_min (float): Minimum number of background events when calculating the asimov significance.
        logits (bool): Whether the predicted values are given as logits (Defauls: False).
        
    Returns:
        results (dict): Dictionary containing max significance, optimal threshold, etc.
    """
    # 1. Compute ROC curve to get efficiency (tpr) and rejection (1-fpr) at all thresholds
    # fpr = False Positive Rate (Background Efficiency)
    # tpr = True Positive Rate (Signal Efficiency)
    #thresholds = np.linspace(0, 1, num_thresholds)
    #fpr, tpr = roc_custom_thresholds(y_true, y_pred, thresholds)
    fpr, tpr, thresholds = roc_curve(y_true, y_pred)
    
    # 2. Scale efficiencies to event counts
    # s = Total Signal * Signal Efficiency
    s_counts = weight_s * tpr
    
    # b = Total Background * Background Efficiency
    b_counts = weight_b * fpr
    
    # 3. Calculate systematic uncertainty for each point
    # sigma_b = b * 5%
    sigma_b_counts = b_counts * sys_uncertainty
    
    # 4. Calculate significance for all thresholds at once
    significances = calculate_asimov_significance(s_counts, b_counts, sigma_b_counts, b_min=b_min)
    
    # 5. Find the maximum
    best_idx = np.argmax(significances)
    max_sig = significances[best_idx]
    best_threshold = thresholds[best_idx]
    
    # Corresponding signal and background events at this threshold
    best_s = s_counts[best_idx]
    best_b = b_counts[best_idx]
    
    if ret_full:
        return {
            "optimal_threshold": best_threshold,
            "max_significance": max_sig,
            "signal_events": best_s,
            "background_events": best_b,
            "significances": significances,
            "thresholds": thresholds,
            "true_positive_rates": tpr,
            "false_positive_rates": fpr
        }
    
    return {
        "optimal_threshold": best_threshold,
        "max_significance": max_sig,
        "signal_events": best_s,
        "background_events": best_b,
    }

class EarlyStopping:
    """Stops training when a monitored metric has stopped improving."""
    def __init__(self, patience=7, min_delta=0, path='checkpoint.pt'):
        # Check self.counter if the loss was lower (better) than for the current iteration
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
        self.path = data_dir + '\\EarlyStopping model\\' + path
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
    ds = make_ds(files, shuffle=False).cache()
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
    
def plot_training_info(train_loss, valid_loss, train_auc, valid_auc, train_ads, valid_ads, n=300) -> None:
    figsize = (10, 7)

    total_epochs = len(valid_loss)
    
    valid_ads_sigs = [val["max_significance"] for val in valid_ads]
    valid_ads_threshs = [val["optimal_threshold"] for val in valid_ads]

    train_loss_truncated = np.array(train_loss[:(len(train_loss) - (len(train_loss) % n))]).reshape(-1, n).mean(axis=1)
    train_auc_truncated = np.array(train_auc[:(len(train_auc) - (len(train_auc) % n))]).reshape(-1, n).mean(axis=1)
    train_ads_truncated = np.array(train_ads[:(len(train_ads) - (len(train_ads) % n))]).reshape(-1, n).mean(axis=1)

    x_train = np.linspace(0, total_epochs-1, len(train_loss_truncated))
    x_valid = np.linspace(0, total_epochs-1, total_epochs)

    plt.figure(figsize=figsize)

    plt.plot(x_train, train_loss_truncated, c='k', linewidth=2, label='Training loss')
    plt.plot(x_valid, valid_loss, c='r', linewidth=2, linestyle='--', label='Validation loss')

    plt.legend(loc='best')
    plt.show()
    
    plt.figure(figsize=figsize)

    plt.plot(x_train, train_auc_truncated, c='k', linewidth=2, label='Training auc')
    plt.plot(x_valid, valid_auc, c='r', linewidth=2, linestyle='--', label='Validation auc')

    plt.legend(loc='best')
    plt.show()

    plt.figure(figsize=figsize)

    plt.plot(x_train, train_ads_truncated, c='k', linewidth=2, label='Training ADS')
    plt.plot(x_valid, valid_ads_sigs, c='r', linewidth=2, linestyle='--', label='Validation ADS')

    plt.legend(loc='best')
    plt.show()

    plt.figure(figsize=figsize)

    plt.plot(x_valid, valid_ads_threshs, c='k', linewidth=2, label='Validation Thresholds')

    plt.legend(loc='best')
    plt.title("Validation best Thresholds for ADS")
    plt.show()

    curr_valid_ads = valid_ads[-1]
    plt.figure(figsize=figsize)

    plt.plot(curr_valid_ads["thresholds"], curr_valid_ads["significances"], c='k', linewidth=2, label='Validation ADS')

    plt.legend(loc='best')
    plt.title("Last Epoch Validation ADS per Threshold")
    plt.show()

    plt.figure(figsize=figsize)

    plt.plot(curr_valid_ads["false_positive_rates"], curr_valid_ads["true_positive_rates"], c='k', linewidth=2, label='ROC Curve')

    plt.legend(loc='best')
    plt.title("Last Epoch Validation ROC Curve")
    plt.show()
    
def get_feature_spec():
    return {
        "label": tf.io.FixedLenFeature([], tf.int64),
        **{f"f{i}": tf.io.FixedLenFeature([], tf.float32) for i in range(28)}
    }

def parse_fn(ex_proto):
    ex = tf.io.parse_single_example(ex_proto, get_feature_spec())
    label = ex.pop("label")
    features = tf.stack([ex[f"f{i}"] for i in range(28)], axis=0)
    return features, label

def make_ds(files, batch=2**11, shuffle=False):
    #files = glob.glob(pattern)
    ds = tf.data.TFRecordDataset(files, compression_type="GZIP")
    if shuffle: ds = ds.shuffle(1_000_000, reshuffle_each_iteration=True)
    return ds.map(parse_fn, num_parallel_calls=tf.data.AUTOTUNE).batch(batch).prefetch(tf.data.AUTOTUNE).cache().repeat()























































































