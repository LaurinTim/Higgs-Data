import numpy as np, pandas as pd
import torch, copy
from torch import nn
import os, sys
from pathlib import Path
import importlib.util
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score
import time
import xgboost as xgb
from sklearn.ensemble import RandomForestClassifier

os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1' 

import tensorflow as tf

data_dir = str(Path(__file__).resolve().parent)

spec = importlib.util.spec_from_file_location("utils", data_dir + '\\HIGGS_utils.py')
u = importlib.util.module_from_spec(spec)
spec.loader.exec_module(u)

AUTO = tf.data.experimental.AUTOTUNE

SEED = 0
torch.manual_seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)

torch.set_default_device("cuda" if torch.cuda.is_available() else "cpu")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# %%

feature_description = {
    'features': tf.io.FixedLenFeature([], tf.string),
    'label': tf.io.FixedLenFeature([], tf.float32),
}

train_files = tf.io.gfile.glob(data_dir + '\\HIGGS data\\training' + '\\*.tfrecord')#[:2]
valid_files = tf.io.gfile.glob(data_dir + '\\HIGGS data\\validation' + '\\*.tfrecord')#[:2]

# Count the number of samples in the train and validation datasets
# This takes a long time, so this was run once and it is not manually defined below
#training_size = u.count_samples(train_files)
#validation_size = u.count_samples(valid_files)

training_size = int(1.05e7/21)
validation_size = int(5e5)
training_size = int(1.05e7)
validation_size = int(5e5)
BATCH_SIZE_PER_REPLICA = 2 ** 11
batch_size = BATCH_SIZE_PER_REPLICA
steps_per_epoch = training_size // batch_size
validation_steps = validation_size // batch_size

print(f"steps_per_epoch: {steps_per_epoch}, validation_steps: {validation_steps}")

# %%

ds_train = u.make_ds(train_files, batch=batch_size, shuffle=True)
ds_train_np = ds_train.as_numpy_iterator()

ds_valid = u.make_ds(valid_files, batch=batch_size, shuffle=False)
ds_valid_np = ds_valid.as_numpy_iterator()

ds_valid_all = u.make_ds(valid_files, batch=validation_size, shuffle=False)
ds_valid_all_np = ds_valid_all.as_numpy_iterator()

# %%

class Deep(nn.Module):
    def __init__(self, units=28, p=0.1):
        super().__init__()
        self.linear_stack = nn.Sequential(
            u.DenseBlock(28, units, nn.ReLU(), p),
            u.DenseBlock(units, units, nn.ReLU(), p),
            u.DenseBlock(units, units, nn.ReLU(), p),
            u.DenseBlock(units, units, nn.ReLU(), p),
            u.DenseBlock(units, units, nn.ReLU(), p),
            u.DenseBlock(units, units, nn.Tanh(), p),
            nn.Linear(units, 1)
        )

    def forward(self, x):
        logits = self.linear_stack(x)
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

deep = Deep(units=2**11, p=0.3)
wide = Wide()
model = DeepWide(deep, wide, deep_ratio=0.5)

# %%

class ConvModel(nn.Module):
    def __init__(self, p=0.3):
        super().__init__()
        self.reorder = u.DenseBlock(28, 2**8, nn.ReLU(), p)
        self.conv_stack = nn.Sequential(
            nn.Conv1d(1, 8, 3, stride=1, padding='same'), # 8 x 256
            nn.BatchNorm1d(8),
            nn.ReLU(),
            nn.Dropout1d(p=p),
            nn.MaxPool1d(2), # 8 x 128
            nn.Conv1d(8, 32, 4, stride=2), # 32 x 63
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Dropout1d(p=p),
            nn.MaxPool1d(3), # 32 x 21
            nn.Conv1d(32, 128, 3, stride=1), # 128 x 19
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout1d(p=p),
            nn.Flatten(),
        )
        self.head = nn.Sequential(
            nn.Linear(128*19, 1),
        )

    def forward(self, x):
        features = self.reorder(x)
        features = features.unsqueeze(1)
        features = self.conv_stack(features)
        logits = self.head(features)
        return logits
    
model = ConvModel(p=0.0)

# %%

class ConvModel(nn.Module):
    def __init__(self, p=0.3):
        super().__init__()
        self.reorder = u.DenseBlock(28, 2**10, nn.ReLU(), p)
        self.conv_stack = nn.Sequential(
            nn.Conv1d(1, 8, 3, stride=1, padding='same'), # 8 x 1024
            nn.BatchNorm1d(8),
            nn.ReLU(),
            nn.Dropout1d(p=p),
            nn.MaxPool1d(2), # 8 x 512
            nn.Conv1d(8, 32, 4, stride=2), # 32 x 255
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Dropout1d(p=p),
            nn.MaxPool1d(3), # 32 x 85
            nn.Conv1d(32, 128, 3, stride=2), # 128 x 42
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout1d(p=p),
            nn.Conv1d(128, 256, 4, stride=2), # 256 x 20
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout1d(p=p),
            nn.Flatten(),
        )
        self.head = nn.Sequential(
            nn.Linear(256*20, 1),
        )

    def forward(self, x):
        features = self.reorder(x)
        features = features.unsqueeze(1)
        features = self.conv_stack(features)
        logits = self.head(features)
        return logits
    
model = ConvModel(p=0.0)

# %%

class ConvModel(nn.Module):
    def __init__(self, p=0.3):
        super().__init__()
        self.reorder = u.DenseBlock(28, 2**10, nn.ReLU(), p)
        self.conv_stack = nn.Sequential(
            nn.Conv1d(1, 8, 5, stride=1, padding='same'), # 8 x 1024
            nn.BatchNorm1d(8),
            nn.ReLU(),
            nn.Dropout1d(p=p),
            nn.AvgPool1d(4), # 8 x 256
            nn.Conv1d(8, 32, 4, stride=2), # 32 x 127
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Dropout1d(p=p),
            nn.MaxPool1d(3), # 32 x 42
            nn.Conv1d(32, 128, 3, stride=3), # 128 x 14
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout1d(p=p),
            nn.Flatten(),
        )
        self.head = nn.Sequential(
            nn.Linear(128*14, 1),
        )

    def forward(self, x):
        features = self.reorder(x)
        features = features.unsqueeze(1)
        features = self.conv_stack(features)
        logits = self.head(features)
        return logits
    
model = ConvModel(p=0.0)

# %%

class ConvModel(nn.Module):
    def __init__(self, p=0.3):
        super().__init__()
        self.reorder = u.DenseBlock(28, 2**10, nn.ReLU(), p)
        self.conv_stack = nn.Sequential(
            nn.Conv1d(1, 8, 5, stride=1, padding='same'), # 8 x 1024
            nn.BatchNorm1d(8),
            nn.ReLU(),
            nn.Dropout1d(p=p),
            nn.AvgPool1d(4), # 8 x 256
            nn.Conv1d(8, 32, 4, stride=2), # 32 x 127
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Dropout1d(p=p),
            nn.MaxPool1d(3), # 32 x 42
            nn.Conv1d(32, 128, 3, stride=2), # 128 x 20
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout1d(p=p),
            nn.Conv1d(128, 256, 3, stride=2), # 128 x 9
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout1d(p=p),
            nn.Flatten(),
        )
        self.head = nn.Sequential(
            nn.Linear(256*9, 1),
        )

    def forward(self, x):
        features = self.reorder(x)
        features = features.unsqueeze(1)
        features = self.conv_stack(features)
        logits = self.head(features)
        return logits
    
model = ConvModel(p=0.3)

# %%

class ConvModel(nn.Module):
    def __init__(self, p=0.3):
        super().__init__()
        self.reorder = u.DenseBlock(28, 2**5, nn.ReLU(), p)
        self.conv_stack = nn.Sequential(
            nn.Conv1d(1, 8, 3, stride=1, padding='same'), # 8 x 32
            nn.BatchNorm1d(8),
            nn.ReLU(),
            nn.Dropout1d(p=p),
            nn.MaxPool1d(2), # 8 x 16
            nn.Conv1d(8, 32, 2, stride=1), # 32 x 15
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Dropout1d(p=p),
            nn.MaxPool1d(3), # 32 x 5
            nn.Conv1d(32, 128, 3, stride=1), # 128 x 3
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout1d(p=p),
            nn.Flatten(),
        )
        self.head = nn.Sequential(
            nn.Linear(128*3, 1),
        )

    def forward(self, x):
        features = self.reorder(x)
        features = features.unsqueeze(1)
        features = self.conv_stack(features)
        logits = self.head(features)
        return logits
    
model = ConvModel(p=0.4)

# %%

class HIGGSConvNet(nn.Module):
    """
    A lightweight 1-D convolutional network for the HIGGS dataset
    (28 numeric features → binary label).

    Notes
    -----
    • 1-D convolutions view the feature vector as a length-28
      sequence with one channel.
    • BatchNorm + GELU + Dropout give good regularisation.
    • Change `hidden_channels`, `kernel_size`, etc. to experiment.
    """

    def __init__(
        self,
        hidden_channels: int = 32, # C
        kernel_size: int = 3,
        fc_hidden: int = 64, # F
        p: float = 0.25,
    ):
        super().__init__()

        # 28 features → (B, 1, 28)
        self.backbone = nn.Sequential(
            nn.Conv1d(in_channels=1, out_channels=hidden_channels, kernel_size=kernel_size, padding="same"),
            nn.BatchNorm1d(hidden_channels),
            nn.GELU(),
            nn.Conv1d(hidden_channels, hidden_channels, kernel_size=kernel_size, padding="same"),
            nn.BatchNorm1d(hidden_channels),
            nn.GELU(),
            nn.AdaptiveAvgPool1d(1),      # (B, C, 1)
            nn.Flatten(),                 # (B, C)
            nn.Dropout(p),
        )

        self.head = nn.Sequential(
            nn.Linear(hidden_channels, fc_hidden), # (B, F)
            nn.GELU(),
            nn.Dropout(p),
            nn.Linear(fc_hidden, 1),      # binary output logit
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        x : torch.Tensor
            Shape (batch, 28) -- already float32 and feature-scaled.

        Returns
        -------
        torch.Tensor
            Shape (batch,) – raw logits.  
            Apply `torch.sigmoid` or use `BCEWithLogitsLoss` during training.
        """
        x = x.unsqueeze(1)        # (B, 1, 28)
        features = self.backbone(x)
        logits = self.head(features)
        return logits
    
    
class DeepWideConv(nn.Module):
    def __init__(self, deep, wide, conv, deep_ratio=1/3, wide_ratio=1/3):
        super().__init__()
        self.deep = deep
        self.wide = wide
        self.conv = conv
        self.deep_ratio = deep_ratio
        self.wide_ratio = wide_ratio

    def forward(self, x):
        deep_logits = self.deep(x)
        wide_logits = self.wide(x)
        conv_logits = self.conv(x)
        logits = self.deep_ratio * deep_logits + self.wide_ratio * wide_logits + (1 - self.deep_ratio - self.wide_ratio) * conv_logits
        return logits
    
deep = Deep(units=2**11, p=0.3)
wide = Wide()
conv = HIGGSConvNet(hidden_channels=32, kernel_size=3, fc_hidden=64, p=0.3)
model = DeepWideConv(deep, wide, conv, deep_ratio=1/3, wide_ratio=1/3)

# %%

class Deep(nn.Module):
    def __init__(self, units=28, p=0.1):
        super().__init__()
        self.linear_stack = nn.Sequential(
            u.DenseBlock(28, units, nn.GELU(), p),
            u.DenseBlock(units, units, nn.GELU(), p),
            u.DenseBlock(units, units, nn.GELU(), p),
            u.DenseBlock(units, units, nn.GELU(), p),
            u.DenseBlock(units, units, nn.GELU(), p),
            u.DenseBlock(units, units, nn.GELU(), p),
            u.DenseBlock(units, units, nn.GELU(), p),
            u.DenseBlock(units, units, nn.Tanh(), p),
            nn.Linear(units, 1)
        )

    def forward(self, x):
        logits = self.linear_stack(x)
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

deep = Deep(units=2**11, p=0.2)
wide = Wide()
model = DeepWide(deep, wide, deep_ratio=0.5)

# %%

model.to(device)

optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.05)
#optimizer = torch.optim.AdamW(model.parameters(), lr=0.0001, weight_decay=0.1)
#lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.2, patience=1, threshold=0.0001, cooldown=0, min_lr=0.000001, eps=1e-08)
#lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.2, patience=0, threshold=0.00003, cooldown=0, min_lr=0.000001, eps=1e-08)
lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, 100, 1e-10, -1)
loss_fn = nn.BCEWithLogitsLoss()
early_stopping = u.EarlyStopping(patience=10, min_delta=0.000, path='best_model.pth')

lr_div = (1e-2 / 1e-6)**(1 / 30)

# %%

train_history = []
valid_history = []
train_history_auc = []
valid_history_auc = []


def train_loop(data, model, loss_fn, optimizer):    
    losses = []
    aucs = []
    
    # Set the model to training mode - important for batch normalization and dropout layers
    # Unnecessary in this situation but added for best practices
    model.train()
    for train_step, (features, labels) in enumerate(data):
        if train_step == steps_per_epoch:
            break
        
        features = torch.from_numpy(copy.copy(features)).to(device)
        labels = torch.from_numpy(copy.copy(labels)).to(device)
        
        # Compute prediction and loss
        optimizer.zero_grad()
        outputs = model(features)
        outputs = torch.squeeze(outputs)
        loss = loss_fn(outputs, labels.float())
        losses.append(loss.cpu().detach().numpy())
        aucs.append(roc_auc_score(labels.detach().cpu().numpy(), outputs.detach().cpu().numpy()))

        # Backpropagation
        loss.backward()
        optimizer.step()

        if train_step % 10000 == -1:
            loss = loss.item()
            print(f"loss: {loss:.5f}")
            print(f'Auc: {aucs[-1]:.5f}')
            
    print(f'Training average loss: {sum(losses)/len(losses):.5f}')
    print(f'Training average auc: {sum(aucs)/len(aucs):.5f}')
        
    return losses, aucs
            
def valid_loop(data, model, loss_fn):
    # Set the model to evaluation mode - important for batch normalization and dropout layers
    # Unnecessary in this situation but added for best practices
    model.eval()
    sum_loss = 0
    sum_count = 0
    val_labels = []
    val_preds = []
    
    # Evaluating the model with torch.no_grad() ensures that no gradients are computed during test mode
    # also serves to reduce unnecessary gradient computations and memory usage for tensors with requires_grad=True
    with torch.no_grad():
        for valid_step, (features, labels) in enumerate(data):
            if valid_step == validation_steps:
                break
            
            features = torch.from_numpy(copy.copy(features)).to(device)
            labels = torch.from_numpy(copy.copy(labels)).to(device)
            
            outputs = model(features)
            outputs = torch.squeeze(outputs)
            loss = loss_fn(outputs, labels.float()).item()
            sum_loss += loss
            sum_count += 1
            
            val_labels.extend(labels.detach().cpu().numpy())
            val_preds.extend(outputs.detach().cpu().numpy())
            
        avg_loss = sum_loss / max(sum_count, 1)
        auc = roc_auc_score(val_labels, val_preds)
        
    print(f"Validation average loss: {avg_loss:.5f}")
    print(f'Validation auc: {auc:.5f}')
    
    return avg_loss, auc

def valid_prediction(data, model, loss_fn):
    # Set the model to evaluation mode - important for batch normalization and dropout layers
    # Unnecessary in this situation but added for best practices
    model.eval()
    sum_loss = 0
    sum_count = 0
    val_labels = []
    val_preds = []
    
    # Evaluating the model with torch.no_grad() ensures that no gradients are computed during test mode
    # also serves to reduce unnecessary gradient computations and memory usage for tensors with requires_grad=True
    with torch.no_grad():
        features, labels = next(iter(data))
        
        features = torch.from_numpy(copy.copy(features)).to(device)
        labels = torch.from_numpy(copy.copy(labels)).to(device)
        
        outputs = model(features)
        #outputs = nn.Sigmoid(outputs)
        outputs = torch.squeeze(outputs)
        loss = loss_fn(outputs, labels.float()).item()
        sum_loss += loss
        sum_count += 1
        
        val_labels.extend(labels.detach().cpu().numpy())
        val_preds.extend(outputs.detach().cpu().numpy())
            
        avg_loss = sum_loss / max(sum_count, 1)
        auc = roc_auc_score(val_labels, val_preds)
        
    print(f"Validation average loss: {avg_loss:.6f}")
    print(f'Validation auc: {auc:.5f}')
    
    #val_labels = val_labels[:validation_size]
    #val_preds = val_preds[:validation_size]
    
    return val_labels, val_preds

# %%

epochs = 200
total_start = time.time()
cont = True
optimizer.param_groups[0]['lr'] = 1e-6
for t in range(epochs):
    t += 107
    print(f"Epoch {t+1}\n-------------------------------")
    curr_lr = optimizer.param_groups[0]['lr']
    #print(f'Current learning rate: {curr_lr}')
    
    start_time = time.time()
    
    train_losses, train_aucs = train_loop(ds_train_np, model, loss_fn, optimizer)
    valid_loss, valid_auc = valid_loop(ds_valid_np, model, loss_fn)
    
    duration = time.time()-start_time
    print(f'Epoch {t+1} finished in {duration:.2f} seconds and with learning rate {curr_lr:.8f}')
    
    train_history.extend(train_losses)
    valid_history.append(valid_loss)
    train_history_auc.extend(train_aucs)
    valid_history_auc.append(valid_auc) 
    early_stopping(valid_loss, model)
        
    if (t + 1) % 10 == 0:
        u.plot_training_info(train_history, valid_history, train_history_auc, valid_history_auc, n=100)
        
    if early_stopping.early_stop and curr_lr <= 1e-5 and t>=120:
        print('Early stopping triggered')
        break
    
    #optimizer.param_groups[0]['lr'] /= lr_div
    
    #lr_scheduler.step(valid_history[-1])
    if optimizer.param_groups[0]['lr'] >= 1e-8 and False:
        lr_scheduler.step()
        
    if optimizer.param_groups[0]['lr'] == 1e-10:
        cont = False
    #else:
    #    optimizer.param_groups[0]['lr'] /= 1.1
    
    #if curr_lr >= 0.000001 and t > 10 and (valid_history[-10] - ((valid_history[-1] + valid_history[-2]) / 2)) <= lr_thresh:
    #    lr_thresh /= 10
    #    optimizer.param_groups['lr'] = max(optimizer.param_groups['lr'] * 0.2, 0.000001)
        
    print()
    
total_duration = time.time() - total_start
print(f"Done! Total elapsed time is {total_duration:.2f} seconds.")

# %%

u.plot_training_info(train_history, valid_history, train_history_auc, valid_history_auc, n=int(5126/8))

# %%

best_model = copy.deepcopy(model)
best_model.load_state_dict(torch.load(data_dir + '\\EarlyStopping model\\best_model.pth'))

# %%

val_labels, val_pred = valid_prediction(ds_valid_all_np, best_model, loss_fn)
pred_df = pd.DataFrame(val_pred, columns=['pred']).T

# %%

train_info = pd.DataFrame([train_history, train_history_auc], index=['loss_history', 'auc_history']).T
valid_info = pd.DataFrame([valid_history, valid_history_auc], index=['loss_history', 'auc_history']).T

# %%

train_info.to_csv(data_dir + "\\DL info\\train_info.csv", index=False)
valid_info.to_csv(data_dir + "\\DL info\\valid_info.csv", index=False)

# %%

pred_df.to_csv(data_dir + '\\predictions\\DL_prediction.csv', index=False)

# %%













































































