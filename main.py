import numpy as np, pandas as pd
import torch
from torch import nn
import os, sys
from pathlib import Path
import importlib.util
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score

os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1' 

import tensorflow as tf

data_dir = str(Path(__file__).resolve().parent)

spec = importlib.util.spec_from_file_location("utils", data_dir + '\\utils.py')
u = importlib.util.module_from_spec(spec)
spec.loader.exec_module(u)

AUTO = tf.data.experimental.AUTOTUNE

SEED = 0
torch.manual_seed(SEED)
np.random.seed(SEED)

torch.set_default_device("cuda" if torch.cuda.is_available() else "cpu")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# %%

feature_description = {
    'features': tf.io.FixedLenFeature([], tf.string),
    'label': tf.io.FixedLenFeature([], tf.float32),
}
decoder = u.make_decoder(feature_description)

train_files = tf.io.gfile.glob(data_dir + '\\training' + '\\*.tfrecord')#[:1]
valid_files = tf.io.gfile.glob(data_dir + '\\validation' + '\\*.tfrecord')#[:1]

dataset_size = int(11e6)
validation_size = int(5e5)
BATCH_SIZE_PER_REPLICA = 2 ** 11
training_size = dataset_size - validation_size
batch_size = BATCH_SIZE_PER_REPLICA
steps_per_epoch = training_size // batch_size
validation_steps = validation_size // batch_size

print(f"steps_per_epoch: {steps_per_epoch}, validation_steps: {validation_steps}")

# %%

ds_train = u.load_dataset(train_files, decoder, ordered=False)
ds_train = (
    ds_train
    .cache()
    .repeat()
    .shuffle(2 ** 19)
    .batch(batch_size)
    .prefetch(AUTO)
)
ds_train_np = ds_train.as_numpy_iterator()

ds_valid = u.load_dataset(valid_files, decoder, ordered=False)
ds_valid = (
    ds_valid
    .cache()
    .repeat()
    .batch(batch_size)
    .prefetch(AUTO)
)
ds_valid_np = ds_valid.as_numpy_iterator()

# %%

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
    
class NeuralNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear_stack = nn.Sequential(
            DenseBlock(28, 56, nn.Tanh(), 0.1),
            DenseBlock(56, 56, nn.Tanh(), 0.1),
            DenseBlock(56, 56, nn.Tanh(), 0.1),
            DenseBlock(56, 56, nn.Tanh(), 0.1),
            DenseBlock(56, 56, nn.Tanh(), 0.0),
            DenseBlock(56, 56, nn.Tanh(), 0.0),
            nn.Linear(56, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        logits = self.linear_stack(x)
        return logits

model = NeuralNetwork()

# %%

class Deep(nn.Module):
    def __init__(self, p=0.1):
        super().__init__()
        self.linear_stack = nn.Sequential(
            DenseBlock(28, 56, nn.Tanh(), p),
            DenseBlock(56, 56, nn.Tanh(), p),
            DenseBlock(56, 56, nn.Tanh(), p),
            DenseBlock(56, 56, nn.Tanh(), p),
            DenseBlock(56, 56, nn.Tanh(), p),
            DenseBlock(56, 56, nn.Tanh(), p),
            nn.Linear(56, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        logits = self.linear_stack(x)
        return logits
    
class Wide(nn.Module):
    def __init__(self, p=0.1):
        super().__init__()
        self.linear_stack = nn.Sequential(
            DenseBlock(28, 2**11, nn.Tanh(), p),
            nn.Linear(2**11, 1),
            nn.Sigmoid()
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

deep = Deep(p=0.1)
wide = Wide(p=0.2)
model = DeepWide(deep, wide, deep_ratio=0.5)

# %%

model.to(device)

optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, mode='min', factor=0.2, patience=0, threshold=0.0001, cooldown=0, min_lr=0.0001, eps=1e-08)
loss_fn = nn.BCEWithLogitsLoss()

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
        
        features = torch.from_numpy(features).to(device)
        labels = torch.from_numpy(labels).to(device)
        
        # Compute prediction and loss
        optimizer.zero_grad()
        outputs = model(features)
        outputs = torch.squeeze(outputs)
        loss = loss_fn(outputs, labels)
        losses.append(loss.cpu().detach().numpy())
        aucs.append(roc_auc_score(labels.detach().cpu().numpy(), outputs.detach().cpu().numpy()))

        # Backpropagation
        loss.backward()
        optimizer.step()

        if train_step % 10000 == 0:
            loss = loss.item()
            print(f"loss: {loss:4f}")
            print(f'Auc: {aucs[-1]:4f}')
            
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
            
            features = torch.from_numpy(features).to(device)
            labels = torch.from_numpy(labels).to(device)
            
            outputs = model(features)
            outputs = torch.squeeze(outputs)
            loss = loss_fn(outputs, labels).item()
            sum_loss += loss
            sum_count += 1
            
            val_labels.extend(labels.detach().cpu().numpy())
            val_preds.extend(outputs.detach().cpu().numpy())
            
        avg_loss = sum_loss / max(sum_count, 1)
        auc = roc_auc_score(val_labels, val_preds)

    print(f"Validation average loss: {avg_loss:4f}\n")
    print(f'Validation auc: {auc:4f}')
    
    return avg_loss, auc

# %%

epochs = 20
for t in range(epochs):
    print(f"Epoch {t+1}\n-------------------------------")
    curr_lr = optimizer.param_groups[0]['lr']
    print(f'Current learning rate: {curr_lr}')
    train_losses, train_aucs = train_loop(ds_train_np, model, loss_fn, optimizer)
    valid_loss, valid_auc = valid_loop(ds_valid_np, model, loss_fn)
    train_history.extend(train_losses)
    valid_history.append(valid_loss)
    train_history_auc.extend(train_aucs)
    valid_history_auc.append(valid_auc)
    lr_scheduler.step(valid_history[-1])
print("Done!")

# %%

x_train = np.linspace(0, epochs-1, len(train_history))
x_valid = np.linspace(0, epochs-1, epochs)

plt.figure(figsize=(15,8))

plt.plot(x_train, train_history, c='k', label='Train loss')
plt.plot(x_valid, valid_history, c='r', linestyle='--', label='Validation loss')

plt.legend(loc='best')
plt.show()

# %%

n = 300

train_history_truncated = np.array(train_history[:(len(train_history) - (len(train_history) % n))]).reshape(-1, n).mean(axis=1)

x_train = np.linspace(0, epochs-1, len(train_history_truncated))
x_valid = np.linspace(0, epochs-1, epochs)

plt.figure(figsize=(15,8))

plt.plot(x_train, train_history_truncated, c='k', linewidth=2, label='Train loss')
plt.plot(x_valid, valid_history, c='r', linewidth=2, linestyle='--', label='Validation loss')

plt.legend(loc='best')
plt.show()

# %%

x_train = np.linspace(0, epochs-1, len(train_history_auc))
x_valid = np.linspace(0, epochs-1, epochs)

plt.figure(figsize=(15,8))

plt.plot(x_train, train_history_auc, c='k', label='Train auc')
plt.plot(x_valid, valid_history_auc, c='r', linestyle='--', label='Validation auc')

plt.legend(loc='best')
plt.show()

# %%

n = 300

train_history_auc_truncated = np.array(train_history_auc[:(len(train_history_auc) - (len(train_history_auc) % n))]).reshape(-1, n).mean(axis=1)

x_train = np.linspace(0, epochs-1, len(train_history_auc_truncated))
x_valid = np.linspace(0, epochs-1, epochs)

plt.figure(figsize=(15,8))

plt.plot(x_train, train_history_auc_truncated, c='k', linewidth=2, label='Train auc')
plt.plot(x_valid, valid_history_auc, c='r', linewidth=2, linestyle='--', label='Validation auc')

plt.legend(loc='best')
plt.show()

# %%












































































