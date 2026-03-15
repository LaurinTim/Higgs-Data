import tensorflow as tf, numpy as np
from pathlib import Path
import torch
from torch import nn
import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from sklearn.metrics import roc_curve
import logging
from typing import Any

data_dir = str(Path(__file__).resolve().parent)

# %%

class AsimovLoss(nn.Module):
    def __init__(self, s_tot: float, b_tot: float, sys_uncertainty: float):
        super().__init__()
        self.s_tot: float = s_tot
        self.b_tot: float = b_tot
        self.sys_uncertainty: float = sys_uncertainty
        self.epsilon: float = 1e-8

    def AsimovSignificance(self, s: torch.Tensor, b: torch.Tensor, sig_b: torch.Tensor) -> torch.Tensor:
        # If s/b is very small (smaller than 0) use the taylor expansion to get the ADS
        # Otherwise floating point errors can occur when s<<b
        if False and s/b < 1e-4:
            Z = s / torch.sqrt(b + torch.square(sig_b))
        
        else:
            term1 = (s + b) * torch.log((s + b) * (b + torch.square(sig_b)) / (torch.square(b) + (s + b) * torch.square(sig_b) + self.epsilon) + self.epsilon)
            term2 = (torch.square(b) / (torch.square(sig_b) + self.epsilon)) * torch.log(1 + torch.square(sig_b) * s / (b * (b + torch.square(sig_b)) + self.epsilon))
            Z = torch.sqrt(2 * (term1 - term2))

        return Z
    
    def forward(self, y_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
        sWeight = self.s_tot / torch.sum(y_true)
        bWeight = self.b_tot / torch.sum(1 - y_true)

        s = sWeight * torch.sum(y_pred * y_true)
        b = bWeight * torch.sum(y_pred * (1 - y_true))
        sig_b = self.sys_uncertainty * b

        s = s.double()
        b = b.double()
        sig_b = sig_b.double()

        Z = self.AsimovSignificance(s, b, sig_b)
        """
        print(sWeight)
        print(bWeight)
        print(s)
        print(b)
        print(sig_b)
        print(Z)
        print()
        """
        return 1 / Z

class SignificanceLoss(nn.Module):
    def __init__(self, s_tot: float, b_tot: float):
        super().__init__()
        self.s_tot: float = s_tot
        self.b_tot: float = b_tot
        self.epsilon: float = 1e-8
    
    def forward(self, y_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
        assert torch.sum(y_true) != 0, "ERROR: BATCH CONTAINS ONLY BACKGROUND EVENTS"
        assert torch.sum(y_true) != len(y_true), "ERROR: BATCH CONTAINS ONLY SIGNAL EVENTS"
        

        sWeight = self.s_tot / torch.sum(y_true)
        bWeight = self.b_tot / torch.sum(1 - y_true)

        s = sWeight * torch.sum(y_pred * y_true)
        b = bWeight * torch.sum(y_pred * (1 - y_true))

        s = s.double()
        b = b.double()

        return (b) / (torch.square(s) + self.epsilon)

def calculate_asimov_significance(s: float | np.ndarray, b: float | np.ndarray, sigma_b: float | np.ndarray, b_min: float = 1) -> float | np.ndarray:
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

def find_optimal_asimov_threshold(y_true: np.ndarray, y_pred: np.ndarray, weight_s: float = 100, weight_b: float = 1000, sys_uncertainty: float = 0.05, b_min: float = 1, ret_full: bool = True) -> dict[str, Any]:
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
    mask = (tpr * sum(y_true) >= 5)
    significances = significances * mask
    
    # 5. Find the maximum
    best_idx = np.argmax(significances)
    max_sig = significances[best_idx]
    best_threshold = thresholds[best_idx]
    
    # Corresponding signal and background events at this threshold
    best_s = s_counts[best_idx]
    best_b = b_counts[best_idx]
    """
    print(best_threshold)
    print(max_sig)
    print(best_s)
    print(best_b)
    print(list(zip(thresholds, tpr, fpr)))
    """
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
    def __init__(self, patience: int = 7, min_delta: float = 0, path: str = 'checkpoint.pt'):
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
        self.patience: int = patience
        self.min_delta: float = min_delta
        self.path: str = data_dir + '\\EarlyStopping model\\' + path
        self.counter: int = 0
        self.best_loss: float | None = None
        self.early_stop: bool = False

    def __call__(self, val_loss: float, model: nn.Module) -> None:
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

    def save_checkpoint(self, model: nn.Module) -> None:
        '''Saves model'''
        torch.save(model.state_dict(), self.path)
        
def count_samples(files: list[str]) -> int:
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
    def __init__(self, input_size: int, output_size: int, activation: nn.Module, dropout_rate: float = 0.1):
        super().__init__()
        self.stack = nn.Sequential(
            nn.Linear(input_size, output_size),
            nn.BatchNorm1d(num_features=output_size),
            activation,
            nn.Dropout(p=dropout_rate)
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        logits = self.stack(x)
        return logits

def plot_training_info(train_loss: list[float], valid_loss: list[float], train_auc: list[float], valid_auc: list[float], train_ads: list[float], valid_ads: list[dict[str, Any]], valid_labels: np.ndarray, valid_outputs: np.ndarray, n: int = 300, compare_prev_epoch: int = 0, save_fig: str | None = None) -> None:
    fig = plt.figure(figsize=(12, 16), constrained_layout=True)
    gs = fig.add_gridspec(13, 2)
    ax1 = fig.add_subplot(gs[:4, :])
    ax2 = fig.add_subplot(gs[4:7, 0])
    ax3 = fig.add_subplot(gs[4:7, 1])
    ax4 = fig.add_subplot(gs[7:10, 0])
    ax5 = fig.add_subplot(gs[7:10, 1])
    ax6 = fig.add_subplot(gs[10:, 0])
    ax7 = fig.add_subplot(gs[10:, 1])

    total_epochs = len(valid_loss)
    
    curr_valid_ads = valid_ads[-1]
    valid_ads_sigs = [val["max_significance"] for val in valid_ads]
    valid_ads_threshs = [val["optimal_threshold"] for val in valid_ads]

    train_loss_truncated = np.array(train_loss[:(len(train_loss) - (len(train_loss) % n))]).reshape(-1, n).mean(axis=1)
    train_auc_truncated = np.array(train_auc[:(len(train_auc) - (len(train_auc) % n))]).reshape(-1, n).mean(axis=1)
    train_ads_truncated = np.array(train_ads[:(len(train_ads) - (len(train_ads) % n))]).reshape(-1, n).mean(axis=1)

    x_train = np.linspace(0, total_epochs-1, len(train_loss_truncated))
    x_valid = np.linspace(0, total_epochs-1, total_epochs)

    def single_plot(ax, x, y, label, title=None, xlabel=None, ylabel=None):
        ax.plot(x, y, c='k', linewidth=2, label=label)
        ax.legend(loc='best')
        if title:
            ax.set_title(title)
        if xlabel:
            ax.set_xlabel(xlabel)
        if ylabel:
            ax.set_ylabel(ylabel)
    
    def double_plot(ax, x1, x2, y1, y2, label1, label2, title=None, xlabel=None, ylabel=None):
        ax.plot(x1, y1, c='k', linewidth=2, label=label1)
        ax.plot(x2, y2, c='r', linewidth=2, linestyle='--', label=label2)
        ax.legend(loc='best')
        if title:
            ax.set_title(title)
        if xlabel:
            ax.set_xlabel(xlabel)
        if ylabel:
            ax.set_ylabel(ylabel)

    double_plot(ax1, x_train, x_valid, train_loss_truncated, valid_loss, 'Training loss', 'Validation loss', 'Loss per Epoch', 'Epoch', 'Loss')
    double_plot(ax2, x_train, x_valid, train_auc_truncated, valid_auc, 'Training AUC', 'Validation AUC', 'AUC per Epoch', 'Epoch', 'AUC')
    double_plot(ax3, x_train, x_valid, train_ads_truncated, valid_ads_sigs, 'Training ADS', 'Validation ADS', 'ADS per Epoch', 'Epoch', 'ADS')
    single_plot(ax4, x_valid, valid_ads_threshs, 'Validation Thresholds', 'Validation best Thresholds for ADS', 'Epoch', 'Threshold')

    if compare_prev_epoch > 0 and total_epochs > compare_prev_epoch:
        compare_valid_ads = valid_ads[-(compare_prev_epoch+1)]
        double_plot(ax5, curr_valid_ads["thresholds"], compare_valid_ads["thresholds"], curr_valid_ads["significances"], compare_valid_ads["significances"], 
                    f'Epoch {total_epochs} Validation ADS', f'Epoch {total_epochs - compare_prev_epoch} Validation ADS', 'ADS per Threshold', 'Threshold', 'ADS [σ]')
        double_plot(ax6, curr_valid_ads["false_positive_rates"], compare_valid_ads["false_positive_rates"], curr_valid_ads["true_positive_rates"], compare_valid_ads["true_positive_rates"], 
                    f'Epoch {total_epochs} Validation ADS', f'Epoch {total_epochs - compare_prev_epoch} Validation ADS', 'ROC Curve', 'False Positive Rate', 'True Positive Rate')

    else:
        single_plot(ax5, curr_valid_ads["thresholds"], curr_valid_ads["significances"], 'Validation ADS', 'ADS per Threshold', 'Threshold', 'ADS [σ]')
        single_plot(ax6, curr_valid_ads["false_positive_rates"], curr_valid_ads["true_positive_rates"], 'Validation ADS', 'ROC Curve', 'False Positive Rate', 'True Positive Rate')

    plot_output_histogram(ax7, valid_labels, valid_outputs)

    if save_fig:
        fig.savefig(Path(save_fig), bbox_inches="tight")
        plt.close(fig)
    
    else:
        plt.show()

def plot_output_histogram(ax: Axes, labels: np.ndarray, outputs: np.ndarray) -> None:
    outputs_background = outputs[labels == 0]
    outputs_signal = outputs[labels == 1]

    bins = np.linspace(0, 1, 101)

    ax.hist(outputs_background, bins=bins, density=True, histtype='bar', color='b', label='Background')
    ax.hist(outputs_signal, bins=bins, density=True, histtype='step', color='r', linewidth=2, label='Signal')

    ax.set_title("Histograms of the predicted values")
    ax.set_xlabel("Predicted Value")
    ax.set_ylabel("Prevalence")
    ax.legend(loc="best")    

def get_feature_spec() -> dict[str, tf.io.FixedLenFeature]:
    return {
        "label": tf.io.FixedLenFeature([], tf.int64),
        **{f"f{i}": tf.io.FixedLenFeature([], tf.float32) for i in range(28)}
    }

def parse_fn(ex_proto: tf.Tensor) -> tuple[tf.Tensor, tf.Tensor]:
    ex = tf.io.parse_single_example(ex_proto, get_feature_spec())
    label = ex.pop("label")
    features = tf.stack([ex[f"f{i}"] for i in range(28)], axis=0)
    return features, label

def make_ds(files: list[str], batch: int = 2**11, shuffle: bool = False) -> tf.data.Dataset:
    #files = glob.glob(pattern)
    ds = tf.data.TFRecordDataset(files, compression_type="GZIP")
    if shuffle: ds = ds.shuffle(1_000_000, reshuffle_each_iteration=True)
    return ds.map(parse_fn, num_parallel_calls=tf.data.AUTOTUNE).batch(batch).prefetch(tf.data.AUTOTUNE).cache().repeat()

def setup_logging(filemode: str = "a") -> logging.Logger:
    logger = logging.getLogger("train")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()

    fmt = logging.Formatter("%(asctime)s %(levelname)s %(message)s")

    out_dir = Path("logs")
    fh = logging.FileHandler(out_dir / "HIGGS.log", mode=filemode)
    fh.setFormatter(fmt)
    logger.addHandler(fh)

    logger.propagate = False

    return logger






















































































# %%
