import tensorflow as tf, numpy as np, pandas as pd
import os, sys
from pathlib import Path
import torch
from torch import nn
import matplotlib.pyplot as plt
import importlib.util
import json, glob
from tqdm import tqdm
from sklearn.linear_model import LogisticRegression
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import log_loss, brier_score_loss
from sklearn.metrics import roc_curve as sklearn_roc_curve
from torch.nn import Sigmoid

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

    ax1.plot(x_train, train_loss_truncated, c='k', label='Training loss', linewidth=2)
    ax1.plot(x_valid, valid_loss, c='r', linestyle='--', label='Validation loss', linewidth=2)

    ax1.legend(loc='best')
    ax1.set_title("Loss per Epoch")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss")
    
    ax2.plot(x_train, train_auc_truncated, c='k', label='Training AUC', linewidth=2)
    ax2.plot(x_valid, valid_auc, c='r', linestyle='--', label='Validation AUC', linewidth=2)

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
    
def plot_data(x, y, title, legend=None, xlabel=None, ylabel=None, sci=True):
        
    fig, ax = plt.subplots(figsize=(5, 4))
    
    ax.plot(x, y, c='k', label=legend, linewidth=2)
    
    if legend: ax.legend(loc='best')
    ax.set_title(title)
    if xlabel: ax.set_xlabel(xlabel)
    if ylabel: ax.set_ylabel(ylabel)
    
    if sci:
        ax.ticklabel_format(axis="y", style="sci", scilimits=(0,0))
    
    plt.show()
    
def plot_roc_curve(y_train, p_train, y_valid, p_valid):
    fpr_train, tpr_train = roc_curve(y_train, p_train)
    fpr_valid, tpr_valid = roc_curve(y_valid, p_valid)
    
    fig, ax = plt.subplots(figsize=(5, 4))
    
    ax.plot(fpr_train, tpr_train, c='k', label="Training ROC curve", linewidth=2)
    ax.plot(fpr_valid, tpr_valid, c='r', label="Validation ROC curve", linestyle='--', linewidth=2)
    
    ax.legend(loc='best')
    ax.set_title("ROC curve")
    ax.set_xlabel("False positive rate")
    ax.set_ylabel("True positive rate")
    
    ax.ticklabel_format(axis="y", style="sci", scilimits=(0,0))
    
    plt.show()
    
def roc_curve(y, p):
    
    y = y.astype(bool) if y.dtype != bool else y
        
    desc_score_indices = np.argsort(-p)
    sorted_y = y[desc_score_indices]
    sorted_p = p[desc_score_indices]

    tp = np.cumsum(sorted_y)
    fp = np.cumsum(~sorted_y)
    
    tp_total = tp[-1]
    fp_total = fp[-1]
    
    distinct_value_indices = np.where(np.diff(sorted_p))[0]
    threshold_idxs = np.r_[distinct_value_indices, sorted_y.size - 1]
    
    tpr = tp[threshold_idxs] / tp_total
    fpr = fp[threshold_idxs] / fp_total
    
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

def prob_from_logits(logits):
    return nn.Sigmoid()(torch.tensor(logits))

def get_HIGGS_pred():
    ret = dict()
    
    ret["DL_valid"] = pd.read_csv(data_dir + "\\HIGGS\\predictions\\DL_prediction_best.csv").to_numpy().reshape(-1)
    ret["DL_train"] = pd.concat([pd.read_csv(data_dir + "\\HIGGS\\predictions\\DL_prediction_train_part1_best.csv"), 
                                 pd.read_csv(data_dir + "\\HIGGS\\predictions\\DL_prediction_train_part2_best.csv")], 
                                 axis=0, ignore_index=True).to_numpy().reshape(-1)
    
    ret["XGB_valid"] = pd.read_csv(data_dir + "\\HIGGS\\predictions\\XGB_prediction_best.csv").to_numpy().reshape(-1)
    ret["XGB_train"] = pd.concat([pd.read_csv(data_dir + "\\HIGGS\\predictions\\XGB_prediction_train_part1_best.csv"), 
                                 pd.read_csv(data_dir + "\\HIGGS\\predictions\\XGB_prediction_train_part2_best.csv")], 
                                 axis=0, ignore_index=True).to_numpy().reshape(-1)
    
    ret["RFC_valid"] = pd.read_csv(data_dir + "\\HIGGS\\predictions\\RFC_prediction_best.csv").to_numpy().reshape(-1)
    ret["RFC_train"] = pd.concat([pd.read_csv(data_dir + "\\HIGGS\\predictions\\RFC_prediction_train_part1_best.csv"), 
                                 pd.read_csv(data_dir + "\\HIGGS\\predictions\\RFC_prediction_train_part2_best.csv")], 
                                 axis=0, ignore_index=True).to_numpy().reshape(-1)
    
    return ret

def get_SUSY_pred():
    ret = dict()
    
    ret["DL_valid"] = pd.read_csv(data_dir + "\\SUSY\\predictions\\DL_prediction_best.csv").to_numpy().reshape(-1)
    ret["DL_train"] = pd.concat([pd.read_csv(data_dir + "\\SUSY\\predictions\\DL_prediction_best_train_part1.csv"), 
                                 pd.read_csv(data_dir + "\\SUSY\\predictions\\DL_prediction_best_train_part2.csv")], 
                                 axis=0, ignore_index=True).to_numpy().reshape(-1)
    
    ret["XGB_valid"] = pd.read_csv(data_dir + "\\SUSY\\predictions\\XGB_prediction_best.csv").to_numpy().reshape(-1)
    ret["XGB_train"] = pd.read_csv(data_dir + "\\SUSY\\predictions\\XGB_prediction_train_best.csv").to_numpy().reshape(-1)
    #ret["XGB_train"] = pd.concat([pd.read_csv(data_dir + "\\SUSY\\predictions\\XGB_prediction_train_part1_best.csv"), 
    #                             pd.read_csv(data_dir + "\\SUSY\\predictions\\XGB_prediction_train_part2_best.csv")], 
    #                             axis=0, ignore_index=True).to_numpy().reshape(-1)
    
    ret["RFC_valid"] = pd.read_csv(data_dir + "\\SUSY\\predictions\\RFC_prediction_best.csv").to_numpy().reshape(-1)
    ret["RFC_train"] = pd.read_csv(data_dir + "\\SUSY\\predictions\\RFC_prediction_train_best.csv").to_numpy().reshape(-1)
    #ret["RFC_train"] = pd.concat([pd.read_csv(data_dir + "\\SUSY\\predictions\\RFC_prediction_train_part1_best.csv"), 
    #                             pd.read_csv(data_dir + "\\SUSY\\predictions\\RFC_prediction_train_part2_best.csv")], 
    #                             axis=0, ignore_index=True).to_numpy().reshape(-1)
    
    return ret

def plot_roc_curves(data, valid_labels, train_labels):
    fpr_DL_valid, tpr_DL_valid = roc_curve(valid_labels, data["DL_valid"])
    fpr_XGB_valid, tpr_XGB_valid = roc_curve(valid_labels, data["XGB_valid"])
    fpr_RFC_valid, tpr_RFC_valid = roc_curve(valid_labels, data["RFC_valid"])
    
    auc_DL_valid = auc_from_roc(fpr_DL_valid, tpr_DL_valid)
    auc_XGB_valid = auc_from_roc(fpr_XGB_valid, tpr_XGB_valid)
    auc_RFC_valid = auc_from_roc(fpr_RFC_valid, tpr_RFC_valid)
    
    fpr_DL_train, tpr_DL_train = roc_curve(train_labels, data["DL_train"])
    fpr_XGB_train, tpr_XGB_train = roc_curve(train_labels, data["XGB_train"])
    fpr_RFC_train, tpr_RFC_train = roc_curve(train_labels, data["RFC_train"])
    
    auc_DL_train = auc_from_roc(fpr_DL_train, tpr_DL_train)
    auc_XGB_train = auc_from_roc(fpr_XGB_train, tpr_XGB_train)
    auc_RFC_train = auc_from_roc(fpr_RFC_train, tpr_RFC_train)
    
    ret_aucs = {
        "DL_valid": auc_DL_valid,
        "XGB_valid": auc_XGB_valid,
        "RFC_valid": auc_RFC_valid,
        "DL_train": auc_DL_train,
        "XGB_train": auc_XGB_train,
        "RFC_train": auc_RFC_train
    }
    
    fig, (ax1, ax2) = plt.subplots(ncols=2, figsize=(12, 4), sharey=False)
    
    ax1.plot(fpr_DL_valid, tpr_DL_valid, c='k', label="DL ROC curve", linewidth=2)
    ax1.plot(fpr_XGB_valid, tpr_XGB_valid, c='r', label="XGB ROC curve", linewidth=2)
    ax1.plot(fpr_RFC_valid, tpr_RFC_valid, c='b', label="RFC ROC curve", linewidth=2)
    
    ax1.legend(loc='best')
    ax1.set_title("ROC curves for the validation set")
    ax1.set_xlabel("False positive rate")
    ax1.set_ylabel("True positive rate")
    ax1.ticklabel_format(axis="y", style="sci", scilimits=(0,0))
    
    ax2.plot(fpr_DL_train, tpr_DL_train, c='k', label="DL ROC curve", linewidth=2)
    ax2.plot(fpr_XGB_train, tpr_XGB_train, c='r', label="XGB ROC curve", linewidth=2)
    ax2.plot(fpr_RFC_train, tpr_RFC_train, c='b', label="RFC ROC curve", linewidth=2)
    
    ax2.legend(loc='best')
    ax2.set_title("ROC curves for the training set")
    ax2.set_xlabel("False positive rate")
    ax2.set_ylabel("True positive rate")
    ax2.ticklabel_format(axis="y", style="sci", scilimits=(0,0))
    
    plt.show()
    
    return ret_aucs

def platt(y_true, p_pred, logits=True):
    if not logits:
        eps = 1e-6
        p_pred = np.clip(p_pred, eps, 1 - eps)
        p_pred = np.log(p_pred / (1 - p_pred))

    p_pred = p_pred.reshape(-1, 1)
    lr = LogisticRegression(max_iter=1000).fit(p_pred, y_true)
    return lr.predict_proba(p_pred)[:, 1]

def expected_calibration_error(y_true, p_pred, n_bins=20, logits=True):
    if logits:
        p_pred = torch.nn.Sigmoid()(torch.tensor(p_pred)).numpy()
    
    y_true = np.asarray(y_true).astype(int)
    p_pred = np.asarray(p_pred).astype(float)
    bins = np.linspace(0.0, 1.0, n_bins+1)
    idx = np.digitize(p_pred, bins) - 1
    ece = 0.0
    for b in range(n_bins):
        mask = idx == b
        if not np.any(mask): 
            continue
        conf = p_pred[mask].mean()
        acc = y_true[mask].mean()
        w = mask.mean()
        ece += w * abs(acc - conf)
    return ece

def model_calibration_df(y_true, p_pred, logits=True):
    y_true = np.array(y_true)
    r = 5
    
    if logits:
        p_pred = Sigmoid()(torch.tensor(p_pred)).numpy()
    
    auc_norm = round(auc(y_true, p_pred), r)
    ece_norm = round(expected_calibration_error(y_true, p_pred, logits=False), r)
    logl_norm = round(log_loss(y_true, p_pred), r)
    brier_norm = round(brier_score_loss(y_true, p_pred), r)
    norm_scores = [auc_norm, ece_norm, logl_norm, brier_norm]

    iso = IsotonicRegression(out_of_bounds="clip")
    iso.fit(p_pred, y_true)
    p_iso = iso.transform(p_pred)
    auc_iso = round(auc(y_true, p_iso), r)
    ece_iso = round(expected_calibration_error(y_true, p_iso, logits=False), r)
    logl_iso = round(log_loss(y_true, p_iso), r)
    brier_iso = round(brier_score_loss(y_true, p_iso), r)
    iso_scores = [auc_iso, ece_iso, logl_iso, brier_iso]

    p_platt = platt(y_true, p_pred, logits=False)
    auc_platt = round(auc(y_true, p_platt), r)
    ece_platt = round(expected_calibration_error(y_true, p_platt, logits=False), r)
    logl_platt = round(log_loss(y_true, p_platt), r)
    brier_platt = round(brier_score_loss(y_true, p_platt), r)
    platt_scores = [auc_platt, ece_platt, logl_platt, brier_platt]
    
    ret = pd.DataFrame([norm_scores, iso_scores, platt_scores], columns=["AUC", "ECE", "log_loss", "brier_loss"], index=["original", "isotonic", "platt"])
    
    return ret

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

def find_optimal_threshold(y_true, y_pred, weight_s=100, weight_b=1000, sys_uncertainty=0.05, b_min=1, logits=False):
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
    # If the predicted values are logits, convert them to probabilities with the Sigmoid function
    if logits:
        y_pred = Sigmoid()(torch.tensor(y_pred)).numpy()
    
    # 1. Compute ROC curve to get efficiency (tpr) and rejection (1-fpr) at all thresholds
    # fpr = False Positive Rate (Background Efficiency)
    # tpr = True Positive Rate (Signal Efficiency)
    fpr, tpr, thresholds = sklearn_roc_curve(y_true, y_pred)
    
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
    
    return {
        "max_significance": max_sig,
        "optimal_threshold": best_threshold,
        "signal_events": best_s,
        "background_events": best_b,
        "significances": significances,
        "thresholds": thresholds
    }

def plot_sig(ax, sig_DL, sig_XGB, sig_RFC, min_thresh=None):
    sig = [sig_DL["significances"].copy(), sig_XGB["significances"].copy(), sig_RFC["significances"].copy()]
    thresh = [sig_DL["thresholds"].copy(), sig_XGB["thresholds"].copy(), sig_RFC["thresholds"].copy()]
    
    if min_thresh:
        for i in range(3):
            sig[i] = [val for val,bal in zip(sig[i], thresh[i]) if bal>= min_thresh]
            thresh[i] = [val for val in thresh[i] if val>=min_thresh]
    
    ax.plot(thresh[0], sig[0], c='k', label="DL model", linewidth=2)
    ax.plot(thresh[1], sig[1], c='r', label="XGB model", linewidth=2)
    ax.plot(thresh[2], sig[2], c='b', label="RFC model", linewidth=2)

    ax.legend(loc="best")
    ax.set_title("Asimov Discovery Significances at different Thresholds")
    ax.set_xlabel("Threshold")
    ax.set_ylabel("Discovery Significance [σ]")
    
    return ax
    


















































































