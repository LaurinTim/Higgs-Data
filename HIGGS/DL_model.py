# %%
import numpy as np, pandas as pd
import torch, copy
from torch import nn
import os
from pathlib import Path
from sklearn.metrics import roc_auc_score
import time
#from torchinfo import summary
import utils.HIGGS_utils as u
from utils.Model import Deep, Wide, DeepWide
from utils.TrainTest import train_loop, valid_loop, get_prediction, get_prediction_train

os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1' 

import tensorflow as tf

data_dir = str(Path(__file__).resolve().parent)

logger = u.setup_logging(filemode="w")

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

#training_size = int(1.05e7/21)
#validation_size = int(5e5)
training_size = int(1.05e7)
validation_size = int(5e5)
BATCH_SIZE_PER_REPLICA = 2 ** 12
batch_size = BATCH_SIZE_PER_REPLICA
steps_per_epoch = training_size // batch_size
validation_steps = validation_size // batch_size

print(f"steps_per_epoch: {steps_per_epoch}, validation_steps: {validation_steps}")

# %%

ds_train = u.make_ds(train_files, batch=batch_size, shuffle=True)
ds_train_np = ds_train.as_numpy_iterator()

ds_valid = u.make_ds(valid_files, batch=batch_size, shuffle=False)
ds_valid_np = ds_valid.as_numpy_iterator()

ds_valid_all = u.make_ds(valid_files, batch=500000, shuffle=False)
ds_valid_all_np = ds_valid_all.as_numpy_iterator()

# %%

s_tot = 100
b_tot = 1000
sys_uncertainty = 0.05

l = 20.3
sig_s = 3.2
sig_b = 252900
efficiency_s = 5e-2
efficiency_b = 1e-3

#s_tot = l*sig_s*efficiency_s
#b_tot = l*sig_b*efficiency_b

units = 2**11
dropout_rate = 0.2

deep = Deep(units=units, p=dropout_rate)
wide = Wide()
model = DeepWide(deep, wide, deep_ratio=0.5)
model.to(device)

logger.info("Model created with parameters:")
logger.info("device: %s", device)
logger.info("units: %d", units)
logger.info("dropout rate: %f", dropout_rate)
logger.info("units: %s", units)

lr_pretrain = 0.001
epochs_pretrain = 5

weight_decay = 0.005
lr_start = 5e-4
lr_end = 1e-5
epochs_lr_scheduler = 100
epochs = 200

early_stopping_patience = 10
early_stopping_min_delta = 0.0

optimizer = torch.optim.AdamW(model.parameters(), lr=lr_pretrain, weight_decay=weight_decay)
lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, epochs_lr_scheduler, lr_end, -1)
loss_fn_pretrain =  u.SignificanceLoss(s_tot, b_tot) # nn.BCEWithLogitsLoss()
loss_fn_asimov = u.AsimovLoss(s_tot, b_tot, sys_uncertainty)
early_stopping = u.EarlyStopping(patience=early_stopping_patience, min_delta=early_stopping_min_delta, path='loss_test_model.pth')

# %%

train_history = []
valid_history = []
train_history_auc = []
valid_history_auc = []
train_history_ads = []
valid_history_ads = []

loss_fn = loss_fn_pretrain
pretraining = True

logger.info("Total Epochs: %d", epochs)
logger.info("Pretrain Epochs: %d\n", epochs_pretrain)

total_start = time.time()
cont = True
#optimizer.param_groups[0]['lr'] = 1e-6
for t in range(epochs):
    if t == epochs_pretrain:
        loss_fn = loss_fn_asimov
        pretraining = False
        optimizer.param_groups[0]['lr'] = lr
        if epochs_pretrain > 0:
            print('Pretraining complete, switching to Asimov Loss\n')
            logger.info("Pretraining finished\n")

    print(f'Epoch {t+1}\n--------------------------------------------------')
    logger.info('EPOCH %d\n--------------------------------------------------', t + 1)

    curr_lr = optimizer.param_groups[0]['lr']
    #print(f'Current learning rate: {curr_lr}')
    
    start_time = time.time()
    
    train_losses, train_aucs, train_adss = train_loop(ds_train_np, model, loss_fn, optimizer)
    valid_labels, valid_outputs, valid_loss, valid_auc, valid_ads = valid_loop(ds_valid_np, model, loss_fn)
    
    duration = time.time()-start_time
    print(f'Epoch {t+1} finished in {duration:.2f} seconds and with learning rate {curr_lr:.8f}')
    print(f'Early stopping counter: {early_stopping.counter}')

    logger.info('Epoch %d finished in %f seconds and with learning rate %f', t + 1, round(duration, 2), round(curr_lr, 8))
    logger.info('Early stopping counter: %d\n', early_stopping.counter)

    if not pretraining:
        train_history.extend(train_losses)
        valid_history.append(valid_loss)
        train_history_auc.extend(train_aucs)
        valid_history_auc.append(valid_auc)
        train_history_ads.extend(train_adss)
        valid_history_ads.append(valid_ads)
        
    if not pretraining and (t + 1) % 5 == 0:
        u.plot_training_info(train_history, valid_history, train_history_auc, valid_history_auc, train_history_ads, valid_history_ads, 
                             valid_labels, valid_outputs, n=100, compare_prev_epoch=5, save_fig=f"{data_dir}\\logs\\figs\\{t+1}_fig.png")
    
    if not pretraining:
        early_stopping(valid_loss, model)
        if early_stopping.early_stop: # and curr_lr <= 1e-5 and t>=120:
            print('Early stopping triggered')
            logger.info('Early stopping triggered')
            break
    
    #optimizer.param_groups[0]['lr'] /= lr_div
    
    #lr_scheduler.step(valid_history[-1])
    if not pretraining and t < epochs_lr_scheduler + epochs_pretrain: # optimizer.param_groups[0]['lr'] >= 1e-8:
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
print(f'Done! Total elapsed time is {total_duration:.2f} seconds.')
logger.info('Done! Total elapsed time is %f seconds.', round(total_duration, 2))

# %%

labels, inputs, outputs, ads = get_prediction(ds_valid_all_np, model, loss_fn)

# %%

u.plot_output_histogram(labels, outputs)

# %%

u.plot_training_info(train_history, valid_history, train_history_auc, valid_history_auc, n=int(5126/8))

# %%

best_model = copy.deepcopy(model)
best_model.load_state_dict(torch.load(data_dir + '\\EarlyStopping model\\best_model.pth'))

# %%

ds_valid_all = u.make_ds(valid_files, batch=validation_size, shuffle=False)
ds_valid_all_np = ds_valid_all.as_numpy_iterator()

val_labels, val_pred = get_prediction(ds_valid_all_np, best_model, loss_fn)
pred_df = pd.DataFrame(val_pred, columns=['pred'])

# %%

ds_train_all = u.make_ds(train_files, batch=int(5e4), shuffle=False)
ds_train_all_np = ds_train_all.as_numpy_iterator()

train_labels, train_pred = get_prediction_train(ds_train_all_np, best_model, loss_fn)
pred_train_df = pd.DataFrame(train_pred, columns=['pred'])

# %%

pred_train_df1 = pred_train_df[:int(training_size/2)]
pred_train_df2 = pred_train_df[int(training_size/2):]

# %%

pred_df.to_csv(data_dir + '\\predictions\\DL_prediction.csv', index=False)
pred_train_df1.to_csv(data_dir + '\\predictions\\DL_prediction_train_part1.csv', index=False)
pred_train_df2.to_csv(data_dir + '\\predictions\\DL_prediction_train_part2.csv', index=False)

# %%

train_info = pd.DataFrame([train_history, train_history_auc], index=['loss_history', 'auc_history']).T
valid_info = pd.DataFrame([valid_history, valid_history_auc], index=['loss_history', 'auc_history']).T

# %%

train_info.to_csv(data_dir + "\\DL info\\train_info.csv", index=False)
valid_info.to_csv(data_dir + "\\DL info\\valid_info.csv", index=False)

# %%

val_labels, val_pred = get_prediction(ds_valid_all_np, best_model, loss_fn)
pred_df = pd.DataFrame(val_pred, columns=['pred'])

# %%

pred_df.to_csv(data_dir + '\\predictions\\DL_prediction.csv', index=False)

# %%

model.to(device)

optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.05)
lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, 100, 1e-5, -1)
loss_fn = nn.BCEWithLogitsLoss()
early_stopping = u.EarlyStopping(patience=10, min_delta=0.000, path='loss_test_model.pth')











































































