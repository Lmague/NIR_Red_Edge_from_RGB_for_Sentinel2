"""Visualize TFRecord patches and report NDVI statistics.

GEE exports patches at 129-130 px depending on rounding, so we parse the bands
as variable-length features and crop to 128 before display.

Run inside Colab: it authenticates against GCS and copies one shard locally
(TensorFlow's direct GCS reader has flaky permissions in Colab).
"""
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

BUCKET = 'YOUR_BUCKET'
FOLDER = 'dataset_training_mst'
TARGET_SIZE = 128

BANDS = ('B2', 'B3', 'B4', 'B5', 'B6', 'B7', 'B8')


def parse_tfrecord(example):
    """Parse one GEE-exported example. Patches may be 129x129, 129x130, or 130x130."""
    feature_description = {b: tf.io.VarLenFeature(tf.float32) for b in BANDS}
    feature_description['valid'] = tf.io.VarLenFeature(tf.float32)
    parsed = tf.io.parse_single_example(example, feature_description)

    bands = {}
    for band in BANDS:
        dense = tf.sparse.to_dense(parsed[band])
        num_values = tf.shape(dense)[0]
        size_sqrt = tf.cast(tf.sqrt(tf.cast(num_values, tf.float32)), tf.int32)
        is_square = tf.equal(size_sqrt * size_sqrt, num_values)
        # Non-square patches are assumed to be 129 x (N/129).
        height = tf.cond(is_square, lambda: size_sqrt, lambda: tf.constant(129))
        width = tf.cond(is_square, lambda: size_sqrt, lambda: num_values // 129)
        bands[band] = tf.reshape(dense, [height, width])
    return bands


def crop_to_target(bands):
    return {k: v[:TARGET_SIZE, :TARGET_SIZE] for k, v in bands.items()}


def normalize_for_display(arr, percentile=2):
    low = np.percentile(arr, percentile)
    high = np.percentile(arr, 100 - percentile)
    return np.clip((arr - low) / (high - low + 1e-6), 0, 1)


def visualize_sample(tfrecord_path, num_samples=3):
    """Plot RGB / NIR / NDVI / false-color for the first `num_samples` records."""
    compression = 'GZIP' if tfrecord_path.endswith('.gz') else None
    dataset = tf.data.TFRecordDataset(tfrecord_path, compression_type=compression).map(parse_tfrecord)

    fig, axes = plt.subplots(num_samples, 4, figsize=(16, 4 * num_samples))

    for idx, bands in enumerate(dataset.take(num_samples)):
        bands = {k: v.numpy() for k, v in crop_to_target(bands).items()}

        rgb = normalize_for_display(np.stack([bands['B4'], bands['B3'], bands['B2']], axis=-1))
        nir = normalize_for_display(bands['B8'])
        ndvi = (bands['B8'] - bands['B4']) / (bands['B8'] + bands['B4'] + 1e-6)
        false_color = normalize_for_display(np.stack([bands['B8'], bands['B4'], bands['B3']], axis=-1))

        axes[idx, 0].imshow(rgb)
        axes[idx, 0].set_title(f'RGB (Input) #{idx + 1}')
        axes[idx, 0].axis('off')

        axes[idx, 1].imshow(nir, cmap='gray')
        axes[idx, 1].set_title(f'NIR B8 (Target) #{idx + 1}')
        axes[idx, 1].axis('off')

        axes[idx, 2].imshow(ndvi, cmap='RdYlGn', vmin=-0.2, vmax=0.9)
        axes[idx, 2].set_title(f'NDVI #{idx + 1}')
        axes[idx, 2].axis('off')

        axes[idx, 3].imshow(false_color)
        axes[idx, 3].set_title(f'False Color NIR-R-G #{idx + 1}')
        axes[idx, 3].axis('off')

    plt.tight_layout()
    plt.savefig('verification_samples.png', dpi=150)
    plt.show()


def check_statistics(tfrecord_path, num_samples=100):
    """Report NDVI mean/min/max over the first `num_samples` records."""
    compression = 'GZIP' if tfrecord_path.endswith('.gz') else None
    dataset = tf.data.TFRecordDataset(tfrecord_path, compression_type=compression).map(parse_tfrecord)

    all_ndvi = []
    for bands in dataset.take(num_samples):
        bands = {k: v.numpy() for k, v in bands.items()}
        ndvi = (bands['B8'] - bands['B4']) / (bands['B8'] + bands['B4'] + 1e-6)
        all_ndvi.append(np.mean(ndvi))

    print(f"Samples analyzed: {len(all_ndvi)}")
    print(f"NDVI mean: {np.mean(all_ndvi):.3f}")
    print(f"NDVI min:  {np.min(all_ndvi):.3f}")
    print(f"NDVI max:  {np.max(all_ndvi):.3f}")


if __name__ == "__main__":
    import subprocess
    from google.colab import auth

    GCS_PATH = "gs://YOUR_BUCKET/dataset_training_mst/train/data_0.tfrecord.gz"
    LOCAL_PATH = "/tmp/data_0.tfrecord.gz"

    auth.authenticate_user()

    result = subprocess.run(
        ["gsutil", "cp", GCS_PATH, LOCAL_PATH],
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        raise RuntimeError(f"gsutil cp failed: {result.stderr}")

    visualize_sample(LOCAL_PATH, num_samples=3)
    check_statistics(LOCAL_PATH, num_samples=100)
