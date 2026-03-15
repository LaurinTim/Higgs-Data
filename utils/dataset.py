import os

os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'

import tensorflow as tf
from pathlib import Path 

def get_files(data_dir):
    train_files = (data_dir / "training").glob("*.tfrecord")
    val_files = (data_dir / "validation").glob("*.tfrecord")
    return train_files, val_files

def make_ds(name, batch=2**11, shuffle=False):
    BASE_DIR = Path(__file__).resolve().parent.parent
    if name == 'HIGGS':
        from .HIGGS import dataset
        data_dir = BASE_DIR / "data" / "HIGGS data"
    elif name == 'SUSY':
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

def count_samples(files):
    '''
    Get the number of samples in files.

    Parameters
    ----------
    files : list
        List containing the paths to the files in which the samples should 
        be counted.

    Returns
    -------
    n : int
        Number of samples found in the files.

    '''
    ds = make_ds(files, shuffle=False).cache()
    n = sum(1 for _ in ds)   # ~0.5 s per million examples
    
    return n