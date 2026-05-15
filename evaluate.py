import argparse
import sys

import numpy as np
import tensorflow as tf

from src.data.data_loader import build_dataset
from src.models.cnn_msgpp import build_cnn_msgpp_decoder
from src.models.cnn_msgpp import Config as CNNMSGPPConfig
from src.models.micronir import build_micronir
from src.models.micronir import Config as MicroNIRConfig
from src.models.mstpp import build_mstpp
from src.models.mstpp import Config as MSTPPConfig
from src.models.waveletmst import build_wavelet_mst
from src.models.waveletmst import Config as WaveletMSTConfig

MODEL_REGISTRY = {
    'MicroNIR':   (build_micronir,          MicroNIRConfig),
    'WaveletMST': (build_wavelet_mst,       WaveletMSTConfig),
    'CNN_MSGPP':  (build_cnn_msgpp_decoder, CNNMSGPPConfig),
    'MSTPP':      (build_mstpp,             MSTPPConfig),
}

EPS = 1e-7
TARGET_BAND_NAMES = ('B5', 'B6', 'B7', 'B8')

def _masked_mean(values, mask):
    return float(np.sum(values * mask) / (np.sum(mask) + EPS))

def _accumulate(stats, target, pred, mask, rgb):
    n_channels = target.shape[-1]
    mask4 = mask[..., None]

    diff = np.abs(target - pred)
    sq_diff = (target - pred) ** 2

    for c in range(n_channels):
        stats['per_band_sum'][c] += float(np.sum(diff[..., c] * mask))
    stats['valid_pixels'] += float(np.sum(mask))

    stats['mae_sum'] += float(np.sum(diff * mask4))
    stats['mse_sum'] += float(np.sum(sq_diff * mask4))
    stats['valid_elements'] += float(np.sum(mask) * n_channels)

    norm_t = np.sqrt(np.sum(target ** 2, axis=-1) + EPS)
    norm_p = np.sqrt(np.sum(pred ** 2, axis=-1) + EPS)
    cos = np.sum(target * pred, axis=-1) / (norm_t * norm_p + EPS)
    cos = np.clip(cos, -1.0 + EPS, 1.0 - EPS)
    angle = np.arccos(cos)
    stats['sam_sum'] += float(np.sum(angle * mask))

    t_safe = np.clip(target, EPS, None)
    p_safe = np.clip(pred, EPS, None)
    t_norm = t_safe / np.sum(t_safe, axis=-1, keepdims=True)
    p_norm = p_safe / np.sum(p_safe, axis=-1, keepdims=True)
    sid = np.sum(t_norm * np.log(t_norm / p_norm) +
                 p_norm * np.log(p_norm / t_norm), axis=-1)
    stats['sid_sum'] += float(np.sum(sid * mask))

    red = rgb[..., 0]
    b8_true = target[..., -1]
    b8_pred = pred[..., -1]
    ndvi_true = (b8_true - red) / (b8_true + red + EPS)
    ndvi_pred = (b8_pred - red) / (b8_pred + red + EPS)
    stats['ndvi_sum'] += float(np.sum(np.abs(ndvi_true - ndvi_pred) * mask))

    b5_true = target[..., 0]
    b5_pred = pred[..., 0]
    ndre_true = (b8_true - b5_true) / (b8_true + b5_true + EPS)
    ndre_pred = (b8_pred - b5_pred) / (b8_pred + b5_pred + EPS)
    stats['ndre_sum'] += float(np.sum(np.abs(ndre_true - ndre_pred) * mask))

def parse_args():
    p = argparse.ArgumentParser(
        description='Evaluate a trained model on a TFRecord shard.',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument('--model_name', required=True, choices=sorted(MODEL_REGISTRY))
    p.add_argument('--weights', required=True,
                   help='Path to the .weights.h5 checkpoint.')
    p.add_argument('--test_data', required=True,
                   help='Glob pattern for held-out TFRecords.')
    p.add_argument('--patch_size', type=int, default=128)
    p.add_argument('--batch_size', type=int, default=16)
    p.add_argument('--max_batches', type=int, default=None,
                   help='If set, evaluate at most this many batches.')
    return p.parse_args()

def main():
    args = parse_args()
    tf.keras.mixed_precision.set_global_policy('float32')

    builder, config_cls = MODEL_REGISTRY[args.model_name]
    config = config_cls()
    if hasattr(config, 'PATCH_SIZE'):
        config.PATCH_SIZE = args.patch_size
    model = builder(config)
    model.load_weights(args.weights)
    print(f"Loaded {args.model_name} ({model.count_params():,} params) from {args.weights}")

    ds = build_dataset(args.test_data, batch_size=args.batch_size,
                       patch_size=args.patch_size, is_training=False)

    n_channels = 4
    stats = {
        'mae_sum': 0.0,
        'mse_sum': 0.0,
        'sam_sum': 0.0,
        'sid_sum': 0.0,
        'ndvi_sum': 0.0,
        'ndre_sum': 0.0,
        'valid_pixels': 0.0,
        'valid_elements': 0.0,
        'per_band_sum': [0.0] * n_channels,
    }

    for i, (rgb, target, mask) in enumerate(ds):
        pred = model(rgb, training=False)
        _accumulate(stats,
                    target.numpy(),
                    pred.numpy(),
                    mask.numpy(),
                    rgb.numpy())
        if args.max_batches is not None and i + 1 >= args.max_batches:
            break

    valid_px = stats['valid_pixels']
    valid_el = stats['valid_elements']
    if valid_px == 0:
        print("No valid pixels in evaluation set.")
        return 1

    mae = stats['mae_sum'] / valid_el
    rmse = np.sqrt(stats['mse_sum'] / valid_el)
    sam = stats['sam_sum'] / valid_px
    sid = stats['sid_sum'] / valid_px
    ndvi_mae = stats['ndvi_sum'] / valid_px
    ndre_mae = stats['ndre_sum'] / valid_px

    print(f"\nResults for {args.model_name}:")
    print(f"  MAE      {mae:.4f}")
    print(f"  RMSE     {rmse:.4f}")
    print(f"  SAM (rad){sam:>10.4f}")
    print(f"  SID      {sid:.4f}")
    print(f"  NDVI MAE {ndvi_mae:.4f}")
    print(f"  NDRE MAE {ndre_mae:.4f}")

    print("\nPer-band MAE:")
    for name, total in zip(TARGET_BAND_NAMES, stats['per_band_sum']):
        print(f"  {name}: {total / valid_px:.4f}")

    return 0

if __name__ == '__main__':
    sys.exit(main())
