import tensorflow as tf

def get_feature_spec():
    return {
        "label": tf.io.FixedLenFeature([], tf.int64),
        **{f"f{i}": tf.io.FixedLenFeature([], tf.float32) for i in range(18)}
    }

def parse_fn(ex_proto):
    ex = tf.io.parse_single_example(ex_proto, get_feature_spec())
    label = ex.pop("label")
    features = tf.stack([ex[f"f{i}"] for i in range(18)], axis=0)
    return features, label
