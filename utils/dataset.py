import os
import copy
from pathlib import Path

os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'

import tensorflow as tf
from pathlib import Path 

def get_files(data_dir):
    train_files = tf.io.gfile.glob(str(data_dir / 'training' / '*.tfrecord'))[:2]
    val_files = tf.io.gfile.glob(str(data_dir / 'validation' / '*.tfrecord'))[:2]
    return train_files, val_files

def make_ds(name, batch=2**11, shuffle=False):
    BASE_DIR = Path(__file__).resolve().parent.parent
    if name.lower().strip() == 'higgs':
        from .HIGGS import dataset
        data_dir = BASE_DIR / "data" / "HIGGS data"
    elif name.lower().strip() == 'susy':
        from .SUSY import dataset
        data_dir = BASE_DIR / "data" / "SUSY data"
    else:
        raise ValueError(f"Unknown dataset name: {name}")

    files_train, files_val = get_files(data_dir)
    ds_train = tf.data.TFRecordDataset(files_train, compression_type="GZIP")
    ds_val = tf.data.TFRecordDataset(files_val, compression_type="GZIP")

    if shuffle:
        ds_train = ds_train.shuffle(1_000_000, reshuffle_each_iteration=True)

    ret_train = ds_train.map(dataset.parse_fn, num_parallel_calls=tf.data.AUTOTUNE).batch(batch).prefetch(tf.data.AUTOTUNE).cache().repeat()
    ret_val = ds_val.map(dataset.parse_fn, num_parallel_calls=tf.data.AUTOTUNE).batch(batch).prefetch(tf.data.AUTOTUNE).cache().repeat()

    return ret_train, ret_val

def count_samples(name):
    BASE_DIR = Path(__file__).resolve().parent.parent
    if name.lower().strip() == 'higgs':
        from .HIGGS import dataset
        data_dir = BASE_DIR / "data" / "HIGGS data"
    elif name.lower().strip() == 'susy':
        from .SUSY import dataset
        data_dir = BASE_DIR / "data" / "SUSY data"
    else:
        raise ValueError(f"Unknown dataset name: {name}")
    
    files_train = tf.io.gfile.glob(str(data_dir / 'training' / '*.tfrecord'))[:2]
    files_val = tf.io.gfile.glob(str(data_dir / 'validation' / '*.tfrecord'))[:2]
    
    train_count = 0
    for filename in files_train:
        train_count += sum(1 for _ in tf.data.TFRecordDataset(filename, compression_type="GZIP"))
    
    val_count = 0
    for filename in files_val:
        val_count += sum(1 for _ in tf.data.TFRecordDataset(filename, compression_type="GZIP"))
    
    return train_count, val_count
