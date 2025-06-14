import tensorflow as tf, numpy as np, pandas as pd
import torch
from torch import nn
import os, sys
from pathlib import Path
import importlib.util
import matplotlib.pyplot as plt
#from sklearn.metrics import roc_auc_score

os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

data_dir = str(Path(__file__).resolve().parent)

spec = importlib.util.spec_from_file_location("utils", data_dir + '\\utils.py')
u = importlib.util.module_from_spec(spec)
spec.loader.exec_module(u)

AUTO = tf.data.experimental.AUTOTUNE

SEED = 0
torch.manual_seed(SEED)
np.random.seed(SEED)

device = torch.set_default_device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# %%

feature_description = {
    'features': tf.io.FixedLenFeature([], tf.string),
    'label': tf.io.FixedLenFeature([], tf.float32),
}
decoder = u.make_decoder(feature_description)

train_files = tf.io.gfile.glob(data_dir + '\\training' + '\\*.tfrecord')[:1]
valid_files = tf.io.gfile.glob(data_dir + '\\validation' + '\\*.tfrecord')[:1]

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

class NeuralNetworkDeep(nn.Module):
    def __init__(self):
        super(NeuralNetworkDeep).__init__()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(28, 56),
            nn.BatchNorm1d(num_features=56),
            nn.ReLU(),
            nn.Linear(56, 12),
            nn.BatchNorm1d(num_features=12),
            nn.ReLU(),
            nn.Linear(12, 1),
            nn.Sigmoid
        )

    def forward(self, x):
        logits = self.linear_relu_stack(x)
        return logits
    
class NeuralNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(28, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        logits = self.linear_relu_stack(x)
        return logits

model = NeuralNetwork()

# %%

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

model.to(device)

optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, mode='min', factor=0.2, patience=0, threshold=0.0001, cooldown=0, min_lr=0.0001, eps=1e-08)
loss_fn = nn.BCEWithLogitsLoss()

# %%

train_history = []
valid_history = []

def train_loop(data, model, loss_fn, optimizer):    
    losses = []
    
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

        # Backpropagation
        loss.backward()
        optimizer.step()

        if train_step % 1000 == 0:
            loss = loss.item()
            print(f"loss: {loss:4f}")
            
    return losses
            
def valid_loop(data, model, loss_fn):    
    # Set the model to evaluation mode - important for batch normalization and dropout layers
    # Unnecessary in this situation but added for best practices
    model.eval()
    sum_loss = 0
    sum_count = 0
    
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
            
        avg_loss = sum_loss / max(sum_count, 1)

    print(f"Validation average loss: {avg_loss:4f}\n")
    
    return avg_loss

# %%

epochs = 5
for t in range(epochs):
    print(f"Epoch {t+1}\n-------------------------------")
    curr_lr = optimizer.param_groups[0]['lr']
    print(f'Current learning rate: {curr_lr}')
    train_losses = train_loop(ds_train_np, model, loss_fn, optimizer)
    valid_loss = valid_loop(ds_valid_np, model, loss_fn)
    train_history.extend(train_losses)
    valid_history.append(valid_loss)
    lr_scheduler.step(valid_history[-1])
print("Done!")

# %%

x_train = np.linspace(0, epochs-1, len(train_history))
x_valid = np.linspace(0, epochs-1, epochs)

plt.figure(figsize=(15,8))

plt.plot(x_train, train_history, c='k', label='Train')
plt.plot(x_valid, valid_history, c='r', linestyle='--', label='Validation')

plt.legend(loc='best')
plt.show()

# %%












































































