import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import tensorflow as tf

from src.models.cnn_msgpp import build_cnn_msgpp_decoder
from src.models.cnn_msgpp import Config as CNNMSGPPConfig
from src.models.micronir import build_micronir
from src.models.micronir import Config as MicroNIRConfig
from src.models.mstpp import build_mstpp
from src.models.mstpp import Config as MSTPPConfig
from src.models.waveletmst import build_wavelet_mst
from src.models.waveletmst import Config as WaveletMSTConfig

TRAINED_TARGETS = {
    'MicroNIR':   130_852,
    'WaveletMST': 1_049_398,
    'CNN_MSGPP':  982_500,
    'MSTPP':      1_721_110,
}

def main():
    tf.keras.mixed_precision.set_global_policy('float32')

    builders = [
        ('MicroNIR',   build_micronir,           MicroNIRConfig()),
        ('WaveletMST', build_wavelet_mst,        WaveletMSTConfig()),
        ('CNN_MSGPP',  build_cnn_msgpp_decoder,  CNNMSGPPConfig()),
        ('MSTPP',      build_mstpp,              MSTPPConfig()),
    ]

    print(f"{'Model':<12} {'Count':>14}  {'Expected':>14}  Status")
    print('-' * 60)

    failures = 0
    for name, builder, cfg in builders:
        model = builder(cfg)
        n = model.count_params()
        expected = TRAINED_TARGETS[name]
        ok = n == expected
        status = 'OK' if ok else f'MISMATCH (delta={n - expected:+d})'
        print(f"{name:<12} {n:>14,}  {expected:>14,}  {status}")
        if not ok:
            failures += 1

    print('-' * 60)
    if failures == 0:
        print("All parameter counts match.")
        return 0
    print(f"{failures} mismatch(es). Re-check extraction or update the expected count.")
    return 1

if __name__ == '__main__':
    sys.exit(main())
