import argparse
import sys

import tensorflow as tf

from src.data.data_loader import build_dataset, build_train_val_split
from src.models.cnn_msgpp import build_cnn_msgpp_decoder
from src.models.cnn_msgpp import Config as CNNMSGPPConfig
from src.models.micronir import build_micronir
from src.models.micronir import Config as MicroNIRConfig
from src.models.mstpp import build_mstpp
from src.models.mstpp import Config as MSTPPConfig
from src.models.waveletmst import build_wavelet_mst
from src.models.waveletmst import Config as WaveletMSTConfig
from src.training.trainer import ModelTrainer

MODEL_REGISTRY = {
    'MicroNIR':   (build_micronir,          MicroNIRConfig),
    'WaveletMST': (build_wavelet_mst,       WaveletMSTConfig),
    'CNN_MSGPP':  (build_cnn_msgpp_decoder, CNNMSGPPConfig),
    'MSTPP':      (build_mstpp,             MSTPPConfig),
}

TRAINING_DEFAULTS = {
    'MicroNIR':   dict(lr=1e-3, batch_size=32, epochs=30, lambda_sam=0.5,
                       band_weights=[1.5, 1.0, 1.0, 2.0]),
    'WaveletMST': dict(lr=8e-4, batch_size=10, epochs=25, lambda_sam=0.5,
                       band_weights=[1.5, 1.0, 1.0, 2.0]),
    'CNN_MSGPP':  dict(lr=8e-4, batch_size=32, epochs=25, lambda_sam=0.5,
                       band_weights=[1.5, 1.0, 1.0, 2.0]),
    'MSTPP':      dict(lr=1e-4, batch_size=8,  epochs=300, lambda_sam=0.0,
                       band_weights=None),
}

def parse_args():
    p = argparse.ArgumentParser(
        description='Train an RGB to NIR/RedEdge model on Sentinel-2 TFRecords.',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument('--model_name', required=True, choices=sorted(MODEL_REGISTRY),
                   help='Architecture to train.')
    p.add_argument('--train_data', required=True,
                   help='Glob pattern for training TFRecords (local path or gs://).')
    p.add_argument('--val_data', default=None,
                   help='Glob pattern for validation TFRecords. If omitted, '
                        'a 10%% slice of --train_data is used.')
    p.add_argument('--patch_size', type=int, default=128)
    p.add_argument('--batch_size', type=int, default=None,
                   help='Override the per-model default batch size.')
    p.add_argument('--epochs', type=int, default=None,
                   help='Override the per-model default epoch count.')
    p.add_argument('--learning_rate', type=float, default=None,
                   help='Override the per-model default peak learning rate.')
    p.add_argument('--warmup_epochs', type=int, default=2)
    p.add_argument('--clip_norm', type=float, default=1.0)
    p.add_argument('--lambda_l1', type=float, default=1.0)
    p.add_argument('--lambda_sam', type=float, default=None,
                   help='Override the per-model default SAM weight.')
    p.add_argument('--lambda_freq', type=float, default=0.0)
    p.add_argument('--lambda_wavelet', type=float, default=0.0)
    p.add_argument('--sam_mode', choices=['cosine', 'acos'], default='cosine')
    p.add_argument('--checkpoint_dir', default='checkpoints')
    p.add_argument('--drive_path', default=None,
                   help='Optional secondary save location (e.g. /content/drive/MyDrive/...).')
    p.add_argument('--resume_from', default=None,
                   help='Path to a .weights.h5 file to resume from.')
    p.add_argument('--mixed_precision', action='store_true',
                   help='Enable mixed_float16 (recommended on Ampere/L4 GPUs).')
    p.add_argument('--steps_per_epoch', type=int, default=2000,
                   help='Estimate used to schedule LR warmup + cosine.')
    p.add_argument('--early_stopping_patience', type=int, default=None)
    return p.parse_args()

def main():
    args = parse_args()
    defaults = TRAINING_DEFAULTS[args.model_name]

    batch_size = args.batch_size or defaults['batch_size']
    epochs = args.epochs or defaults['epochs']
    lr = args.learning_rate or defaults['lr']
    lambda_sam = args.lambda_sam if args.lambda_sam is not None else defaults['lambda_sam']

    if args.mixed_precision:
        tf.keras.mixed_precision.set_global_policy('mixed_float16')
        print("Enabled mixed_float16 global policy.")
    else:
        tf.keras.mixed_precision.set_global_policy('float32')

    if args.val_data:
        train_ds = build_dataset(args.train_data, batch_size=batch_size,
                                 patch_size=args.patch_size, is_training=True)
        val_ds = build_dataset(args.val_data, batch_size=batch_size,
                               patch_size=args.patch_size, is_training=False)
    else:
        train_ds, val_ds = build_train_val_split(
            args.train_data, batch_size=batch_size, patch_size=args.patch_size,
        )

    builder, config_cls = MODEL_REGISTRY[args.model_name]
    config = config_cls()
    if hasattr(config, 'PATCH_SIZE'):
        config.PATCH_SIZE = args.patch_size
    model = builder(config)
    print(f"{args.model_name} built with {model.count_params():,} parameters.")
    model.summary(print_fn=lambda s: print('  ' + s))

    trainer = ModelTrainer(
        model=model,
        learning_rate=lr,
        weight_decay=1e-4,
        epochs=epochs,
        warmup_epochs=args.warmup_epochs,
        steps_per_epoch=args.steps_per_epoch,
        clip_norm=args.clip_norm,
        lambda_l1=args.lambda_l1,
        lambda_sam=lambda_sam,
        lambda_freq=args.lambda_freq,
        lambda_wavelet=args.lambda_wavelet,
        band_weights=defaults['band_weights'],
        sam_mode=args.sam_mode,
        checkpoint_dir=args.checkpoint_dir,
        drive_path=args.drive_path,
        early_stopping_patience=args.early_stopping_patience,
    )

    if args.resume_from:
        trainer.restore(weights_path=args.resume_from)

    best = trainer.fit(train_ds, val_ds)
    print(f"Best validation L1: {best:.4f}")
    return 0

if __name__ == '__main__':
    sys.exit(main())
