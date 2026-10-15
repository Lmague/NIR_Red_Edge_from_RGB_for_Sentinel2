import numpy as np
import tensorflow as tf

BANDS = ('B2', 'B3', 'B4', 'B5', 'B6', 'B7', 'B8')
INPUT_BANDS = ('B4', 'B3', 'B2')
TARGET_BANDS = ('B5', 'B6', 'B7', 'B8')

FEATURE_DESCRIPTION = {b: tf.io.VarLenFeature(tf.float32) for b in BANDS}
FEATURE_DESCRIPTION['valid'] = tf.io.VarLenFeature(tf.float32)

def _find_dimensions(num_values):
    num_values_f = tf.cast(num_values, tf.float32)
    sqrt_n = tf.sqrt(num_values_f)
    sqrt_n_int = tf.cast(sqrt_n, tf.int32)
    is_square = tf.equal(sqrt_n_int * sqrt_n_int, num_values)

    def square_dims():
        return sqrt_n_int, sqrt_n_int

    def fallback_dims():
        candidates = tf.constant([64, 65, 66, 128, 129, 130, 256], dtype=tf.int32)
        remainders = num_values % candidates
        valid = tf.equal(remainders, 0)
        idx = tf.argmax(tf.cast(valid, tf.int32))
        height = candidates[idx]
        width = num_values // height
        any_valid = tf.reduce_any(valid)
        height = tf.cond(any_valid, lambda: height, lambda: sqrt_n_int + 1)
        width = tf.cond(any_valid, lambda: width, lambda: num_values // (sqrt_n_int + 1))
        return height, width

    return tf.cond(is_square, square_dims, fallback_dims)

def parse_tfrecord(example, patch_size=128):
    parsed = tf.io.parse_single_example(example, FEATURE_DESCRIPTION)

    def extract_band(band_name):
        dense = tf.sparse.to_dense(parsed[band_name])
        dense = tf.where(tf.math.is_nan(dense) | tf.math.is_inf(dense),
                         tf.zeros_like(dense), dense)
        n = tf.shape(dense)[0]
        h, w = _find_dimensions(n)
        return tf.reshape(dense, [h, w])

    rgb = tf.stack([extract_band(b) for b in INPUT_BANDS], axis=-1)
    target = tf.stack([extract_band(b) for b in TARGET_BANDS], axis=-1)
    valid_mask = extract_band('valid')

    rgb = tf.clip_by_value(rgb, 0.0, 10000.0) / 10000.0
    target = tf.clip_by_value(target, 0.0, 10000.0) / 10000.0

    non_black = tf.cast(
        (tf.reduce_sum(rgb, axis=-1) > 0.01) | (tf.reduce_sum(target, axis=-1) > 0.01),
        tf.float32,
    )
    mask = valid_mask * non_black

    return tf.cast(rgb, tf.float32), tf.cast(target, tf.float32), tf.cast(mask, tf.float32)

def _crop_pair(rgb, target, mask, crop_size, mode='random'):
    h = tf.shape(rgb)[0]
    w = tf.shape(rgb)[1]
    if mode == 'random':
        off_h = tf.random.uniform((), 0, tf.maximum(h - crop_size, 0) + 1, dtype=tf.int32)
        off_w = tf.random.uniform((), 0, tf.maximum(w - crop_size, 0) + 1, dtype=tf.int32)
    else:
        off_h = tf.maximum((h - crop_size) // 2, 0)
        off_w = tf.maximum((w - crop_size) // 2, 0)
    rgb = rgb[off_h:off_h + crop_size, off_w:off_w + crop_size, :]
    target = target[off_h:off_h + crop_size, off_w:off_w + crop_size, :]
    mask = mask[off_h:off_h + crop_size, off_w:off_w + crop_size]
    rgb = tf.image.resize_with_crop_or_pad(rgb, crop_size, crop_size)
    target = tf.image.resize_with_crop_or_pad(target, crop_size, crop_size)
    mask = tf.image.resize_with_crop_or_pad(mask[..., tf.newaxis], crop_size, crop_size)[..., 0]
    return rgb, target, mask

def augment(rgb, target, mask, patch_size=128):
    rgb, target, mask = _crop_pair(rgb, target, mask, patch_size, mode='random')
    if tf.random.uniform(()) > 0.5:
        rgb = tf.image.flip_left_right(rgb)
        target = tf.image.flip_left_right(target)
        mask = tf.image.flip_left_right(mask[..., tf.newaxis])[..., 0]
    if tf.random.uniform(()) > 0.5:
        rgb = tf.image.flip_up_down(rgb)
        target = tf.image.flip_up_down(target)
        mask = tf.image.flip_up_down(mask[..., tf.newaxis])[..., 0]
    k = tf.random.uniform((), 0, 4, dtype=tf.int32)
    rgb = tf.image.rot90(rgb, k)
    target = tf.image.rot90(target, k)
    mask = tf.image.rot90(mask[..., tf.newaxis], k)[..., 0]
    return rgb, target, mask

def center_crop(rgb, target, mask, patch_size=128):
    return _crop_pair(rgb, target, mask, patch_size, mode='center')

def build_dataset(pattern, batch_size, patch_size=128, is_training=True,
                  compression='GZIP', shuffle_buffer=1000):
    files = tf.io.gfile.glob(pattern)
    if not files:
        raise FileNotFoundError(f"No files matched pattern: {pattern}")
    print(f"Found {len(files)} TFRecord files for pattern: {pattern}")

    ds = tf.data.TFRecordDataset(
        files,
        compression_type=compression if compression else None,
        num_parallel_reads=tf.data.AUTOTUNE,
    )
    ds = ds.map(lambda x: parse_tfrecord(x, patch_size), num_parallel_calls=tf.data.AUTOTUNE)

    if is_training:
        ds = ds.shuffle(shuffle_buffer)
        ds = ds.map(lambda r, t, m: augment(r, t, m, patch_size),
                    num_parallel_calls=tf.data.AUTOTUNE)
    else:
        ds = ds.map(lambda r, t, m: center_crop(r, t, m, patch_size),
                    num_parallel_calls=tf.data.AUTOTUNE)

    return ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)

def build_train_val_split(train_pattern, batch_size, patch_size=128,
                          val_split=0.1, seed=42, compression='GZIP'):
    files = tf.io.gfile.glob(train_pattern)
    if not files:
        raise FileNotFoundError(f"No files matched pattern: {train_pattern}")

    rng = np.random.default_rng(seed)
    files = list(files)
    rng.shuffle(files)
    n_val = max(1, int(len(files) * val_split))
    val_files, train_files = files[:n_val], files[n_val:]
    print(f"Split {len(files)} files: {len(train_files)} train / {len(val_files)} val")

    def _make(file_list, training):
        ds = tf.data.TFRecordDataset(
            file_list,
            compression_type=compression if compression else None,
            num_parallel_reads=tf.data.AUTOTUNE,
        )
        ds = ds.map(lambda x: parse_tfrecord(x, patch_size), num_parallel_calls=tf.data.AUTOTUNE)
        if training:
            ds = ds.shuffle(1000)
            ds = ds.map(lambda r, t, m: augment(r, t, m, patch_size),
                        num_parallel_calls=tf.data.AUTOTUNE)
        else:
            ds = ds.map(lambda r, t, m: center_crop(r, t, m, patch_size),
                        num_parallel_calls=tf.data.AUTOTUNE)
        return ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)

    return _make(train_files, training=True), _make(val_files, training=False)
