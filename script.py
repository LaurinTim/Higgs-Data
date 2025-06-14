import tensorflow as tf, numpy as np, pandas as pd
import os, sys
from pathlib import Path

os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

sys.path.insert(0, str(Path(__file__).resolve().parent))
os.chdir(Path(__file__).resolve().parent)

data_dir = os.getcwd()

# %%

import utils as u

# %%

files = [data_dir + '\\training\\' + val for val in os.listdir(data_dir + '\\training')]

# %%

train_files = tf.io.gfile.glob(data_dir + '\\training' + '\*.tfrecord')
valid_files = tf.io.gfile.glob(data_dir + '\\validation' + '\*.tfrecord')



# %%

raw_df = tf.data.TFRecordDataset(data_dir + '\\training\\shard_00.tfrecord')

# %%














































































