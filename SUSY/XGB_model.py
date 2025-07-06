import numpy as np, pandas as pd
from pathlib import Path
import importlib.util
from sklearn.metrics import roc_auc_score
import xgboost as xgb
import os

os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1' 

import tensorflow as tf

data_dir = str(Path(__file__).resolve().parent)

spec = importlib.util.spec_from_file_location("utils", data_dir + '\\SUSY_utils.py')
u = importlib.util.module_from_spec(spec)
spec.loader.exec_module(u)

AUTO = tf.data.experimental.AUTOTUNE

SEED = 0
np.random.seed(SEED)
tf.random.set_seed(SEED)

# %%

feature_description = {
    'features': tf.io.FixedLenFeature([], tf.string),
    'label': tf.io.FixedLenFeature([], tf.float32),
}

train_files = tf.io.gfile.glob(data_dir + '\\SUSY data\\training' + '\\*.tfrecord')#[:2]
valid_files = tf.io.gfile.glob(data_dir + '\\SUSY data\\validation' + '\\*.tfrecord')#[:2]

# Count the number of samples in the train and validation datasets
# This takes a long time, so this was run once and it is not manually defined below
#training_size = u.count_samples(train_files)
#validation_size = u.count_samples(valid_files)

#training_size = int(4.5e6/21)
#validation_size = int(5e5)
training_size = int(4.5e6)
validation_size = int(5e5)
BATCH_SIZE_PER_REPLICA = 2 ** 11
batch_size = BATCH_SIZE_PER_REPLICA
steps_per_epoch = training_size // batch_size
validation_steps = validation_size // batch_size

print(f"steps_per_epoch: {steps_per_epoch}, validation_steps: {validation_steps}")

# %%

ds_train = u.make_ds(train_files, batch=training_size, shuffle=True)
ds_train_np = ds_train.as_numpy_iterator()
arr_train = next(iter(ds_train_np))

ds_valid = u.make_ds(valid_files, batch=validation_size, shuffle=False)
ds_valid_np = ds_valid.as_numpy_iterator()
arr_valid = next(iter(ds_valid_np))

# %%

modelXGB = xgb.XGBClassifier(n_estimators=50, max_depth=6, max_leaves=42, 
                             objective='binary:logistic', n_jobs=-1, seed=42)

# %%

modelXGB = xgb.XGBClassifier(n_estimators=300, max_depth=7, max_leaves=100, 
                             objective='binary:logistic', n_jobs=-1, seed=42)

modelXGB = xgb.XGBClassifier(n_estimators=50, max_depth=8, max_leaves=None, 
                             max_bin=256, eta=0.239, gamma=0.83, min_child_weight=6,
                             objective='binary:logistic', n_jobs=-1, seed=42)
modelXGB.fit(arr_train[0], arr_train[1])

pred = modelXGB.predict_proba(arr_valid[0])[:, 1]
score = roc_auc_score(arr_valid[1], pred)

pred_train = modelXGB.predict_proba(arr_train[0])[:, 1]
score_train = roc_auc_score(arr_train[1], pred_train)

print(f'Score: {score:.5f}')
print(f'Train score: {score_train:.5f}')

# %%

# 0.87709
modelXGB = xgb.XGBClassifier(n_estimators=50, max_depth=8, max_leaves=None, 
                             max_bin=256, eta=0.3, gamma=0, min_child_weight=1,
                             objective='binary:logistic', n_jobs=-1, seed=42)

# 0.87714
modelXGB = xgb.XGBClassifier(n_estimators=50, max_depth=8, max_leaves=None, 
                             max_bin=256, eta=0.239, gamma=0, min_child_weight=1,
                             objective='binary:logistic', n_jobs=-1, seed=42)

# 0.87718
modelXGB = xgb.XGBClassifier(n_estimators=50, max_depth=8, max_leaves=None, 
                             max_bin=256, eta=0.239, gamma=0.83, min_child_weight=1,
                             objective='binary:logistic', n_jobs=-1, seed=42)

# 0.87719
modelXGB = xgb.XGBClassifier(n_estimators=50, max_depth=8, max_leaves=None, 
                             max_bin=256, eta=0.239, gamma=0.83, min_child_weight=6, 
                             max_delta_step=0,
                             objective='binary:logistic', n_jobs=-1, seed=42)

# %%

best = 0.87719
best_val = None
found_new_best = False
new_best_score = None

for val in range(21):
    val = 0.1*val
        
    modelXGB = xgb.XGBClassifier(n_estimators=50, max_depth=8, max_leaves=None, 
                                 max_bin=256, eta=0.239, gamma=0.83, min_child_weight=6, 
                                 max_delta_step=val,
                                 objective='binary:logistic', n_jobs=-1, seed=42)
    modelXGB.fit(arr_train[0], arr_train[1])
    
    pred = modelXGB.predict_proba(arr_valid[0])[:, 1]
    score = roc_auc_score(arr_valid[1], pred)
    
    pred_train = modelXGB.predict_proba(arr_train[0])[:, 1]
    score_train = roc_auc_score(arr_train[1], pred_train)
    
    new_best = round(score, 5)>best
    if new_best:
        best_val = val
        found_new_best = True
        new_best_score = score
    
    print('-'*30 + f'\nIteration with Value {val}:')
    print(f'Score: {score:.5f}')
    print(f'Train score: {score_train:.5f}')
    print(f'New Best: {new_best}')
    print('-'*30)
    print()
    
    

if found_new_best:
    print(f'New best was found with Value {best_val} and score {new_best_score:.5f}')

else:
    print('No new best was found')

# %%

pred_df = pd.DataFrame(pred, columns=['pred'])

# %%

pred_df.to_csv(data_dir + '\\predictions\\XGB_prediction.csv', index=False)

# %%























































































