import os
import shutil
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import tensorflow as tf

from src.data.data_loader import build_dataset
from src.models.micronir import build_micronir, Config as MicroNIRConfig
from src.training.trainer import ModelTrainer

BANDS = ('B2', 'B3', 'B4', 'B5', 'B6', 'B7', 'B8')

def _float_feature(values):
    return tf.train.Feature(float_list=tf.train.FloatList(value=values))

def _write_dummy_shard(path, num_records=40, side=129):
    options = tf.io.TFRecordOptions(compression_type='GZIP')
    rng = np.random.default_rng(0)
    with tf.io.TFRecordWriter(path, options=options) as w:
        for _ in range(num_records):
            features = {}
            for b in BANDS:
                values = rng.uniform(500.0, 3500.0, size=side * side).astype(np.float32)
                features[b] = _float_feature(values.tolist())
            features['valid'] = _float_feature([1.0] * side * side)
            example = tf.train.Example(features=tf.train.Features(feature=features))
            w.write(example.SerializeToString())

def main():
    tmpdir = tempfile.mkdtemp(prefix='nir_smoke_')
    shard = os.path.join(tmpdir, 'dummy.tfrecord.gz')
    ckpt_dir = os.path.join(tmpdir, 'checkpoints')

    try:
        print("[1/4] writing dummy TFRecord shard...")
        _write_dummy_shard(shard, num_records=40)

        print("[2/4] building dataset...")
        ds = build_dataset(shard, batch_size=4, patch_size=128,
                           is_training=True, shuffle_buffer=8)

        print("[3/4] building MicroNIR + trainer...")
        model = build_micronir(MicroNIRConfig())
        trainer = ModelTrainer(
            model,
            learning_rate=1e-3,
            epochs=1,
            warmup_epochs=0,
            steps_per_epoch=2,
            checkpoint_dir=ckpt_dir,
            early_stopping_patience=None,
        )

        print("[4/4] running 1 train_step + 1 val_step...")
        rgb, target, mask = next(iter(ds))
        train_loss = trainer.train_step(rgb, target, mask).numpy()
        trainer.val_step(rgb, target, mask)
        val_l1 = trainer.val_l1.result().numpy()

        assert np.isfinite(train_loss), f"train loss is not finite: {train_loss}"
        assert np.isfinite(val_l1), f"val L1 is not finite: {val_l1}"
        assert train_loss > 0, f"train loss is non-positive: {train_loss}"

        print(f"\n  train_step loss = {train_loss:.6f}")
        print(f"  val_step  L1   = {val_l1:.6f}")
        print("\nsmoke test OK")
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)

if __name__ == '__main__':
    main()
