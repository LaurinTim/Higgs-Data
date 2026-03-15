import torch
from torch import nn

class EarlyStopping:
    """Stops training when a monitored metric has stopped improving."""
    def __init__(self, save_path, patience=7, min_delta=0):
        # Check self.counter if the loss was lower (better) than for the current iteration
        """
        Args:
            save_path (str): Path to save the best model file.
            patience (int): How long to wait after last time the monitored metric improved.
                            Default: 7
            min_delta (float): Minimum change in the monitored metric to qualify as an improvement.
                               Default: 0
        """
        self.patience = patience
        self.min_delta = min_delta
        self.save_path = save_path
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
        torch.save(model.state_dict(), self.save_path)
    
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

class Deep(nn.Module):
    def __init__(self, units=28, p=0.1, num_blocks=8):
        super().__init__()
        assert num_blocks >= 2, "At least two blocks are required"
        self.stack = nn.Sequential(
            DenseBlock(28, units, nn.GELU(), p)
        )

        for i in range(num_blocks-2):
            self.stack.append(DenseBlock(units, units, nn.GELU(), p))
        
        self.stack.append(DenseBlock(units, units, nn.Tanh(), p))
        self.stack.append(nn.Linear(units, 1))

    def forward(self, x):
        logits = self.stack(x)
        return logits
    
class Wide(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear_stack = nn.Sequential(
            nn.Linear(28, 1),
        )

    def forward(self, x):
        logits = self.linear_stack(x)
        return logits

class DeepWide(nn.Module):
    def __init__(self, deep, wide, deep_ratio=0.5):
        super().__init__()
        self.deep = deep
        self.wide = wide
        self.deep_ratio = deep_ratio

    def forward(self, x):
        deep_logits = self.deep(x)
        wide_logits = self.wide(x)
        logits = self.deep_ratio * deep_logits + (1 - self.deep_ratio) * wide_logits
        return logits