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

spec = importlib.util.spec_from_file_location("utils", data_dir + '\\HIGGS_utils.py')
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

train_files = tf.io.gfile.glob(data_dir + '\\HIGGS data\\training' + '\\*.tfrecord')#[:1]
valid_files = tf.io.gfile.glob(data_dir + '\\HIGGS data\\validation' + '\\*.tfrecord')#[:1]

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

ds_train = u.make_ds(train_files, batch=training_size, shuffle=True)
ds_train_np = ds_train.as_numpy_iterator()
arr_train = next(iter(ds_train_np))

ds_valid = u.make_ds(valid_files, batch=validation_size, shuffle=False)
ds_valid_np = ds_valid.as_numpy_iterator()
arr_valid = next(iter(ds_valid_np))

# %%

modelXGB = xgb.XGBClassifier(n_estimators=300, max_depth=6, max_leaves=42, 
                             objective='binary:logistic', n_jobs=-1, seed=42)
modelXGB.fit(arr_train[0], arr_train[1])

# %%

pred = modelXGB.predict_proba(arr_valid[0])[:, 1]
score = roc_auc_score(arr_valid[1], pred)

pred_train = modelXGB.predict_proba(arr_train[0])[:, 1]
score_train = roc_auc_score(arr_train[1], pred_train)

print(f'Score: {score:.4f}')
print(f'Train score: {score_train:.4f}')

# %%

pred = modelXGB.predict_proba(arr_valid[0])[:, 1]
pred_df = pd.DataFrame(pred, columns=['pred'])

pred_train = modelXGB.predict_proba(arr_train[0])[:, 1]
pred_train_df = pd.DataFrame(pred_train, columns=['pred'])

# %%

pred_df.to_csv(data_dir + '\\predictions\\XGB_prediction.csv', index=False)
pred_train_df.to_csv(data_dir + '\\predictions\\XGB_prediction_train.csv', index=False)

# %%























































































