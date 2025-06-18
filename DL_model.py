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

spec = importlib.util.spec_from_file_location("utils", data_dir + '\\utils.py')
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
decoder = u.make_decoder(feature_description)

train_files = tf.io.gfile.glob(data_dir + '\\training' + '\\*.tfrecord')#[:1]
valid_files = tf.io.gfile.glob(data_dir + '\\validation' + '\\*.tfrecord')#[:1]

# Count the number of samples in the train and validation datasets
# This takes a long time, so this was run once and it is not manually defined below
#training_size = u.count_samples(train_files)
#validation_size = u.count_samples(valid_files)

training_size = int(1.05e7)
validation_size = int(5e5)
BATCH_SIZE_PER_REPLICA = 2 ** 11
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
    .shuffle(2 ** 19, seed=SEED)
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

ds_valid_all = u.load_dataset(valid_files, decoder, ordered=True)
ds_valid_all = (
    ds_valid_all
    .cache()
    .repeat()
    .batch(validation_size)
    .prefetch(AUTO)
)
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

deep = Deep(units=2**11, p=0.15)
wide = Wide()
model = DeepWide(deep, wide, deep_ratio=0.5)

# %%

model.to(device)

optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, mode='min', factor=0.2, patience=1, threshold=0.0001, cooldown=0, min_lr=0.00001, eps=1e-08)
loss_fn = nn.BCEWithLogitsLoss()
early_stopping = u.EarlyStopping(patience=5, min_delta=0.000, path='best_model.pth')

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
        loss = loss_fn(outputs, labels)
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
            loss = loss_fn(outputs, labels).item()
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
        outputs = torch.squeeze(outputs)
        loss = loss_fn(outputs, labels).item()
        sum_loss += loss
        sum_count += 1
        
        val_labels.extend(labels.detach().cpu().numpy())
        val_preds.extend(outputs.detach().cpu().numpy())
            
        avg_loss = sum_loss / max(sum_count, 1)
        auc = roc_auc_score(val_labels, val_preds)
        
    print(f"Validation average loss: {avg_loss:.5f}")
    print(f'Validation auc: {auc:.5f}')
    
    #val_labels = val_labels[:validation_size]
    #val_preds = val_preds[:validation_size]
    
    return val_labels, val_preds

# %%

epochs = 100
total_start = time.time()

for t in range(epochs):
    print(f"Epoch {t+1}\n-------------------------------")
    curr_lr = optimizer.param_groups[0]['lr']
    #print(f'Current learning rate: {curr_lr}')
    
    start_time = time.time()
    
    train_losses, train_aucs = train_loop(ds_train_np, model, loss_fn, optimizer)
    valid_loss, valid_auc = valid_loop(ds_valid_np, model, loss_fn)
    
    duration = time.time()-start_time
    print(f'Epoch {t+1} finished with duration {duration:.2f}s and using learning rate {curr_lr:.5f}')
    
    train_history.extend(train_losses)
    valid_history.append(valid_loss)
    train_history_auc.extend(train_aucs)
    valid_history_auc.append(valid_auc)
    early_stopping(valid_loss, model)
        
    if (t + 1) % 10 == 0:
        u.plot_training_info(train_history, valid_history, train_history_auc, valid_history_auc, n=500)
        
    if early_stopping.early_stop:
        print('Early stopping triggered')
        break
    lr_scheduler.step(valid_history[-1])
        
    print()
    
total_duration = time.time() - total_start
print(f"Done! Total elapsed time is {total_duration:.2f}s.")

# %%

u.plot_training_info(train_history, valid_history, train_history_auc, valid_history_auc, n=500)

# %%

best_model = copy.deepcopy(model)
best_model.load_state_dict(torch.load(data_dir + '\\EarlyStopping model\\best_model.pth'))

# %%

train_info = pd.DataFrame([train_history, train_history_auc], index=['loss_history', 'auc_history']).T
valid_info = pd.DataFrame([valid_history, valid_history_auc], index=['loss_history', 'auc_history']).T

# %%

train_info.to_csv(data_dir + "\\DL info\\train_info.csv", index=False)
valid_info.to_csv(data_dir + "\\DL info\\valid_info.csv", index=False)

# %%

val_labels, val_pred = valid_prediction(ds_valid_all_np, best_model, loss_fn)
pred_df = pd.DataFrame(val_pred, columns=['pred']).T

# %%

pred_df.to_csv(data_dir + '\\predictions\\DL_prediction.csv')

# %%













































































