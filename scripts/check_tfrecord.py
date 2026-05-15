"""Quick TFRecord sanity check. Run inside Colab.

Downloads one shard from GCS via gsutil (TensorFlow's direct GCS reader
trips on permissions in Colab) and prints the feature shapes and ranges
of the first few records.
"""
import subprocess

import tensorflow as tf

BUCKET_PATH = "gs://YOUR_BUCKET/dataset_training_mst/train/data_0.tfrecord.gz"
LOCAL_PATH = "/tmp/data_0.tfrecord.gz"

if __name__ == "__main__":
    from google.colab import auth

    auth.authenticate_user()

    print("Downloading file...")
    result = subprocess.run(
        ["gsutil", "cp", BUCKET_PATH, LOCAL_PATH],
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        raise RuntimeError(f"gsutil cp failed: {result.stderr}")
    print("File downloaded")

    dataset = tf.data.TFRecordDataset(LOCAL_PATH, compression_type='GZIP')

    count = 0
    for record in dataset:
        count += 1
        if count <= 3:
            example = tf.train.Example()
            example.ParseFromString(record.numpy())
            print(f"\n--- Record {count} ---")
            for key, value in example.features.feature.items():
                if value.float_list.value:
                    vals = list(value.float_list.value)
                    print(f"  {key}: {len(vals)} values, range [{min(vals):.1f}, {max(vals):.1f}]")
                elif value.int64_list.value:
                    vals = list(value.int64_list.value)
                    print(f"  {key}: {len(vals)} values, range [{min(vals)}, {max(vals)}]")

    print(f"\nTotal: {count} records (expected ~500, fewer if many points were filtered out)")
