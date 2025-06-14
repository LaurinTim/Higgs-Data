import tensorflow as tf, numpy as np, pandas as pd
import os, sys
from pathlib import Path
import importlib.util

os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

data_dir = str(Path(__file__).resolve().parent)

spec = importlib.util.spec_from_file_location("utils", data_dir + '\\utils.py')
u = importlib.util.module_from_spec(spec)
spec.loader.exec_module(u)

# %%

train_files = tf.io.gfile.glob(data_dir + '\\training' + '\*.tfrecord')
valid_files = tf.io.gfile.glob(data_dir + '\\validation' + '\*.tfrecord')

# %%

raw_df = tf.data.TFRecordDataset(data_dir + '\\training\\shard_00.tfrecord')

# %%














































































