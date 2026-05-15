import numpy as np
import tensorflow as tf

EPS = 1e-7

def _valid_count(mask, n_channels):
    return tf.reduce_sum(mask) * tf.cast(n_channels, tf.float32) + 1e-8

def l1_loss_masked(y_true, y_pred, mask, band_weights=None):
    diff = tf.abs(y_true - y_pred) * mask[..., tf.newaxis]
    n_channels = tf.shape(y_true)[-1]
    if band_weights is None:
        return tf.reduce_sum(diff) / _valid_count(mask, n_channels)
    bw = tf.constant(band_weights, dtype=diff.dtype)
    bw = bw / tf.reduce_mean(bw)
    return tf.reduce_sum(diff * bw) / _valid_count(mask, n_channels)

def sam_loss_masked(y_true, y_pred, mask):
    dot = tf.reduce_sum(y_true * y_pred, axis=-1)
    norm_t = tf.sqrt(tf.reduce_sum(tf.square(y_true), axis=-1) + EPS)
    norm_p = tf.sqrt(tf.reduce_sum(tf.square(y_pred), axis=-1) + EPS)
    cos = tf.clip_by_value(dot / (norm_t * norm_p + EPS), -1.0 + EPS, 1.0 - EPS)
    loss = (1.0 - cos) * mask
    return tf.reduce_sum(loss) / (tf.reduce_sum(mask) + EPS)

def sam_acos_loss_masked(y_true, y_pred, mask):
    dot = tf.reduce_sum(y_true * y_pred, axis=-1)
    norm_t = tf.sqrt(tf.reduce_sum(tf.square(y_true), axis=-1) + EPS)
    norm_p = tf.sqrt(tf.reduce_sum(tf.square(y_pred), axis=-1) + EPS)
    cos = tf.clip_by_value(dot / (norm_t * norm_p + EPS), -1.0 + EPS, 1.0 - EPS)
    angle = tf.acos(cos) * mask
    return tf.reduce_sum(angle) / (tf.reduce_sum(mask) + EPS)

def frequency_loss_masked(y_true, y_pred, mask):
    mask4 = mask[..., tf.newaxis]
    yt = tf.transpose(y_true * mask4, [0, 3, 1, 2])
    yp = tf.transpose(y_pred * mask4, [0, 3, 1, 2])
    yt_fft = tf.abs(tf.signal.fft2d(tf.cast(yt, tf.complex64)))
    yp_fft = tf.abs(tf.signal.fft2d(tf.cast(yp, tf.complex64)))
    return tf.reduce_mean(tf.abs(yt_fft - yp_fft))

def _haar_dwt(x):
    sqrt2 = np.sqrt(2.0)
    L = np.array([1.0, 1.0]) / sqrt2
    H = np.array([1.0, -1.0]) / sqrt2

    def make_filter(a, b):
        return tf.constant(np.outer(a, b).astype(np.float32).reshape(2, 2, 1, 1))

    f_LL, f_LH, f_HL, f_HH = make_filter(L, L), make_filter(L, H), make_filter(H, L), make_filter(H, H)
    x_f32 = tf.cast(x, tf.float32)
    C = tf.shape(x_f32)[-1]

    def conv(f):
        return tf.nn.depthwise_conv2d(x_f32, tf.tile(f, [1, 1, C, 1]),
                                      strides=[1, 2, 2, 1], padding='SAME')

    return conv(f_LL), conv(f_LH), conv(f_HL), conv(f_HH)

def haar_wavelet_loss_masked(y_true, y_pred, mask):
    mask4 = mask[..., tf.newaxis]
    yt = tf.cast(y_true * mask4, tf.float32)
    yp = tf.cast(y_pred * mask4, tf.float32)
    loss = 0.0
    for ct, cp in zip(_haar_dwt(yt), _haar_dwt(yp)):
        loss = loss + tf.reduce_mean(tf.abs(ct - cp))
    return loss / 4.0

def combined_loss(y_true, y_pred, mask, lambda_l1=1.0, lambda_sam=0.5,
                  lambda_freq=0.0, lambda_wavelet=0.0,
                  band_weights=None, sam_mode='cosine'):
    loss = lambda_l1 * l1_loss_masked(y_true, y_pred, mask, band_weights)
    if lambda_sam > 0:
        if sam_mode == 'cosine':
            loss = loss + lambda_sam * sam_loss_masked(y_true, y_pred, mask)
        elif sam_mode == 'acos':
            loss = loss + lambda_sam * sam_acos_loss_masked(y_true, y_pred, mask)
        else:
            raise ValueError(f"Unknown sam_mode: {sam_mode}")
    if lambda_freq > 0:
        loss = loss + lambda_freq * frequency_loss_masked(y_true, y_pred, mask)
    if lambda_wavelet > 0:
        loss = loss + lambda_wavelet * haar_wavelet_loss_masked(y_true, y_pred, mask)
    return loss

def mae_metric_masked(y_true, y_pred, mask):
    return l1_loss_masked(y_true, y_pred, mask)

def rmse_metric_masked(y_true, y_pred, mask):
    diff = (y_true - y_pred) * mask[..., tf.newaxis]
    n_channels = tf.shape(y_true)[-1]
    return tf.sqrt(tf.reduce_sum(tf.square(diff)) / _valid_count(mask, n_channels))
