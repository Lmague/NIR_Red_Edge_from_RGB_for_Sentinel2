import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

def _activation_name(name):
    table = {'relu': 'relu', 'gelu': 'gelu', 'swish': 'swish', 'silu': 'swish', 'mish': 'mish'}
    if name.lower() not in table:
        raise ValueError(f"Unknown activation: {name}. Options: {list(table)}")
    return table[name.lower()]

def _initializer_for(activation):
    if activation in ('relu', 'leaky_relu', 'elu'):
        return tf.keras.initializers.HeNormal()
    return tf.keras.initializers.GlorotUniform()

class DropPath(layers.Layer):
    def __init__(self, drop_prob=0.1, **kwargs):
        super().__init__(**kwargs)
        self.drop_prob = drop_prob

    def call(self, x, training=None):
        if not training or self.drop_prob == 0:
            return x
        keep_prob = 1.0 - self.drop_prob
        shape = (tf.shape(x)[0],) + (1,) * (len(x.shape) - 1)
        random_tensor = keep_prob + tf.random.uniform(shape, dtype=x.dtype)
        binary_mask = tf.floor(random_tensor)
        return x / keep_prob * binary_mask

    def get_config(self):
        cfg = super().get_config()
        cfg.update({'drop_prob': self.drop_prob})
        return cfg

class RMSNorm(layers.Layer):
    def __init__(self, eps=1e-6, **kwargs):
        super().__init__(**kwargs)
        self.eps = eps

    def build(self, input_shape):
        self.gamma = self.add_weight(
            name='gamma',
            shape=(input_shape[-1],),
            initializer='ones',
            trainable=True,
        )

    def call(self, x):
        x_float = tf.cast(x, tf.float32)
        rms = tf.sqrt(tf.reduce_mean(tf.square(x_float), axis=-1, keepdims=True) + self.eps)
        x_normed = x_float / rms
        gamma = tf.cast(self.gamma, x_float.dtype)
        return tf.cast(x_normed * gamma, x.dtype)

    def get_config(self):
        cfg = super().get_config()
        cfg.update({'eps': self.eps})
        return cfg

class HaarWavelet2D(layers.Layer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def build(self, input_shape):
        sqrt2 = np.sqrt(2)
        L = np.array([1, 1]) / sqrt2
        H = np.array([1, -1]) / sqrt2

        self.filter_LL = self.add_weight(
            name='filter_LL', shape=(2, 2, 1, 1),
            initializer=tf.constant_initializer(np.outer(L, L).reshape(2, 2, 1, 1)),
            trainable=False, dtype='float32',
        )
        self.filter_LH = self.add_weight(
            name='filter_LH', shape=(2, 2, 1, 1),
            initializer=tf.constant_initializer(np.outer(L, H).reshape(2, 2, 1, 1)),
            trainable=False, dtype='float32',
        )
        self.filter_HL = self.add_weight(
            name='filter_HL', shape=(2, 2, 1, 1),
            initializer=tf.constant_initializer(np.outer(H, L).reshape(2, 2, 1, 1)),
            trainable=False, dtype='float32',
        )
        self.filter_HH = self.add_weight(
            name='filter_HH', shape=(2, 2, 1, 1),
            initializer=tf.constant_initializer(np.outer(H, H).reshape(2, 2, 1, 1)),
            trainable=False, dtype='float32',
        )
        super().build(input_shape)

    def call(self, x):
        x_f32 = tf.cast(x, tf.float32)
        C = tf.shape(x_f32)[-1]

        results = {}
        for name, filt in [
            ('LL', self.filter_LL), ('LH', self.filter_LH),
            ('HL', self.filter_HL), ('HH', self.filter_HH),
        ]:
            filt_expanded = tf.tile(filt, [1, 1, C, 1])
            out = tf.nn.depthwise_conv2d(
                x_f32, filt_expanded,
                strides=[1, 2, 2, 1],
                padding='SAME',
            )
            results[name] = tf.cast(out, x.dtype)
        return results

class MultiLevelDWT(layers.Layer):
    def __init__(self, num_levels=2, **kwargs):
        super().__init__(**kwargs)
        self.num_levels = num_levels

    def build(self, input_shape):
        self.dwt_layers = [HaarWavelet2D(name=f'dwt_level_{i}') for i in range(self.num_levels)]
        super().build(input_shape)

    def call(self, x):
        all_details = []
        current = x
        for level, dwt in enumerate(self.dwt_layers):
            coeffs = dwt(current)
            all_details.append({
                'level': level,
                'LH': coeffs['LH'], 'HL': coeffs['HL'], 'HH': coeffs['HH'],
                'LL': coeffs['LL'],
            })
            current = coeffs['LL']
        return all_details, current

class DWTFeatureExtractor(layers.Layer):
    def __init__(self, num_levels=2, output_dim=64, activation='swish', **kwargs):
        super().__init__(**kwargs)
        self.num_levels = num_levels
        self.output_dim = output_dim
        self.activation_name = _activation_name(activation)
        self.initializer = _initializer_for(activation)

    def build(self, input_shape):
        self.dwt = MultiLevelDWT(self.num_levels)

        detail_dim = max(self.output_dim // (self.num_levels + 1), 16)
        approx_dim = self.output_dim - (detail_dim * self.num_levels)

        self.detail_projs = []
        self.detail_activations = []
        self.detail_upsamplers = []
        for level in range(self.num_levels):
            self.detail_projs.append(
                layers.Conv2D(detail_dim, 1, kernel_initializer=self.initializer,
                              name=f'detail_proj_level_{level}')
            )
            self.detail_activations.append(
                layers.Activation(self.activation_name, name=f'detail_act_level_{level}')
            )
            scale_factor = 2 ** (level + 1)
            self.detail_upsamplers.append(
                layers.UpSampling2D(size=scale_factor, interpolation='nearest')
            )

        self.approx_proj = layers.Conv2D(max(approx_dim, 16), 1,
                                         kernel_initializer=self.initializer, name='approx_proj')
        self.approx_activation = layers.Activation(self.activation_name, name='approx_act')
        self.approx_upsampler = layers.UpSampling2D(size=2 ** self.num_levels, interpolation='bilinear')

        self.fusion = layers.Conv2D(self.output_dim, 1,
                                    kernel_initializer=self.initializer, name='dwt_fusion')
        super().build(input_shape)

    def call(self, x):
        input_shape = tf.shape(x)
        H, W = input_shape[1], input_shape[2]

        details_by_level, final_approx = self.dwt(x)

        features = []
        for level, (detail_proj, detail_act, upsampler) in enumerate(
            zip(self.detail_projs, self.detail_activations, self.detail_upsamplers)
        ):
            level_details = details_by_level[level]
            detail_concat = tf.concat([
                level_details['LH'], level_details['HL'], level_details['HH'],
            ], axis=-1)
            projected = detail_proj(detail_concat)
            projected = detail_act(projected)
            features.append(upsampler(projected))

        approx_proj = self.approx_proj(final_approx)
        approx_proj = self.approx_activation(approx_proj)
        features.append(self.approx_upsampler(approx_proj))

        features_aligned = [tf.image.resize(f, [H, W], method='nearest') for f in features]
        all_features = tf.concat(features_aligned, axis=-1)
        return self.fusion(all_features)

class TrueSpectralAttention(layers.Layer):
    def __init__(self, dim, num_heads, qk_norm=True, **kwargs):
        super().__init__(**kwargs)
        if dim % num_heads != 0:
            raise ValueError(f"dim ({dim}) must be divisible by num_heads ({num_heads})")
        self.num_heads = num_heads
        self.dim = dim
        self.head_dim = dim // num_heads
        self.qk_norm = qk_norm

    def build(self, input_shape):
        init = tf.keras.initializers.TruncatedNormal(stddev=0.02)
        self.qkv = layers.Dense(self.dim * 3, use_bias=False,
                                kernel_initializer=init, name='qkv_proj')
        self.proj = layers.Dense(self.dim, kernel_initializer=init, name='out_proj')
        if self.qk_norm:
            self.q_norm = RMSNorm(name='q_norm')
            self.k_norm = RMSNorm(name='k_norm')
        super().build(input_shape)

    def call(self, x, training=False):
        input_dtype = x.dtype
        B = tf.shape(x)[0]
        H = tf.shape(x)[1]
        W = tf.shape(x)[2]
        C = self.dim
        N = tf.cast(H, tf.int64) * tf.cast(W, tf.int64)
        N_i32 = tf.cast(N, tf.int32)

        x_flat = tf.reshape(x, (B, N_i32, C))
        qkv = self.qkv(x_flat)
        qkv = tf.reshape(qkv, (B, N_i32, 3, C))
        q, k, v = qkv[:, :, 0], qkv[:, :, 1], qkv[:, :, 2]

        if self.qk_norm:
            q = self.q_norm(q)
            k = self.k_norm(k)

        q = tf.reshape(q, (B, N_i32, self.num_heads, self.head_dim))
        k = tf.reshape(k, (B, N_i32, self.num_heads, self.head_dim))
        v = tf.reshape(v, (B, N_i32, self.num_heads, self.head_dim))
        q = tf.transpose(q, (0, 2, 1, 3))
        k = tf.transpose(k, (0, 2, 1, 3))
        v = tf.transpose(v, (0, 2, 1, 3))

        q_f32 = tf.cast(q, tf.float32)
        k_f32 = tf.cast(k, tf.float32)
        v_f32 = tf.cast(v, tf.float32)

        q_t = tf.transpose(q_f32, (0, 1, 3, 2))
        scale_n = tf.math.rsqrt(tf.cast(N, tf.float32))
        attn = tf.matmul(q_t, k_f32) * scale_n
        attn = attn - tf.reduce_max(attn, axis=-1, keepdims=True)
        attn = tf.nn.softmax(attn, axis=-1)

        v_t = tf.transpose(v_f32, (0, 1, 3, 2))
        out = tf.matmul(attn, v_t)
        out = tf.transpose(out, (0, 1, 3, 2))
        out = tf.reshape(out, (B, N_i32, C))
        out = tf.cast(out, input_dtype)
        out = self.proj(out)
        out = tf.reshape(out, (B, H, W, C))
        return out

class ChannelAttention(layers.Layer):
    def __init__(self, channels, reduction=8, activation='swish', **kwargs):
        super().__init__(**kwargs)
        self.channels = channels
        self.reduction = reduction
        self.activation_name = _activation_name(activation)

    def build(self, input_shape):
        init = _initializer_for('swish')
        reduced_channels = max(self.channels // self.reduction, 4)
        self.fc1 = layers.Dense(reduced_channels, kernel_initializer=init)
        self.activation = layers.Activation(self.activation_name)
        self.fc2 = layers.Dense(self.channels,
                                kernel_initializer=tf.keras.initializers.GlorotNormal())
        super().build(input_shape)

    def call(self, x):
        gap = tf.reduce_mean(x, axis=[1, 2], keepdims=True)
        squeeze = tf.squeeze(gap, axis=[1, 2])
        excite = self.fc1(squeeze)
        excite = self.activation(excite)
        excite = self.fc2(excite)
        excite = tf.nn.sigmoid(excite)
        excite = tf.reshape(excite, (-1, 1, 1, self.channels))
        return x * excite

class SpatialMixerDWConv(layers.Layer):
    def __init__(self, channels, kernel_size=7, expansion=2, drop_rate=0.0, **kwargs):
        super().__init__(**kwargs)
        self.channels = channels
        self.kernel_size = kernel_size
        self.expansion = expansion
        self.drop_rate = drop_rate

    def build(self, input_shape):
        self.norm = RMSNorm(name='norm')
        self.dwconv = layers.DepthwiseConv2D(
            self.kernel_size, padding='same',
            depthwise_initializer='he_normal', name='dwconv',
        )
        self.pwconv_expand = layers.Conv2D(
            self.channels * self.expansion, 1, padding='same',
            kernel_initializer='he_normal', name='pwconv_expand',
        )
        self.act = layers.Activation('gelu', name='gelu')
        self.pwconv_project = layers.Conv2D(
            self.channels, 1, padding='same',
            kernel_initializer='he_normal', name='pwconv_project',
        )
        self.drop = layers.Dropout(self.drop_rate) if self.drop_rate > 0 else None
        super().build(input_shape)

    def call(self, x, training=False):
        y = self.norm(x)
        y = self.dwconv(y)
        y = self.pwconv_expand(y)
        y = self.act(y)
        y = self.pwconv_project(y)
        if self.drop is not None:
            y = self.drop(y, training=training)
        return y

class SpectralTransformerBlock(layers.Layer):
    def __init__(self, embed_dim, num_heads, mlp_ratio=4, dropout=0.1,
                 drop_path=0.0, activation='gelu',
                 use_spatial_mixing=False, spatial_kernel=7, spatial_expansion=2,
                 use_channel_attention=False, **kwargs):
        super().__init__(**kwargs)
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.mlp_ratio = mlp_ratio
        self.dropout_rate = dropout
        self.drop_path_rate = drop_path
        self.activation_name = activation
        self.use_spatial_mixing = use_spatial_mixing
        self.spatial_kernel = spatial_kernel
        self.spatial_expansion = spatial_expansion
        self.use_channel_attention = use_channel_attention

    def build(self, input_shape):
        self.norm1 = RMSNorm(name='norm1')
        self.norm2 = RMSNorm(name='norm2')

        if self.use_spatial_mixing:
            self.spatial_mix = SpatialMixerDWConv(
                self.embed_dim,
                kernel_size=self.spatial_kernel,
                expansion=self.spatial_expansion,
                drop_rate=self.dropout_rate,
                name='spatial_mix',
            )

        self.spectral_attn = TrueSpectralAttention(self.embed_dim, self.num_heads, qk_norm=True)
        if self.use_channel_attention:
            self.channel_attn = ChannelAttention(self.embed_dim, activation='swish')

        self.drop_path = (DropPath(self.drop_path_rate)
                          if self.drop_path_rate > 0 else lambda x, training: x)

        init = _initializer_for(self.activation_name)
        mlp_hidden = self.embed_dim * self.mlp_ratio

        self.mlp = keras.Sequential([
            layers.Dense(mlp_hidden, kernel_initializer=init),
            layers.Activation(_activation_name(self.activation_name)),
            layers.Dropout(self.dropout_rate),
            layers.Dense(self.embed_dim, kernel_initializer=init),
            layers.Dropout(self.dropout_rate),
        ], name='mlp')

        super().build(input_shape)

    def call(self, x, training=False):
        if self.use_spatial_mixing:
            x = x + self.drop_path(self.spatial_mix(x, training=training), training=training)

        normed = self.norm1(x)
        attn_out = self.spectral_attn(normed, training=training)
        if self.use_channel_attention:
            attn_out = self.channel_attn(attn_out)
        x = x + self.drop_path(attn_out, training=training)
        x = x + self.drop_path(self.mlp(self.norm2(x), training=training), training=training)
        return x

class SpatialAttention(layers.Layer):
    def __init__(self, kernel_size=7, **kwargs):
        super().__init__(**kwargs)
        self.kernel_size = kernel_size

    def build(self, input_shape):
        self.conv = layers.Conv2D(
            1, self.kernel_size, padding='same',
            kernel_initializer='glorot_uniform', use_bias=False,
        )
        super().build(input_shape)

    def call(self, x):
        avg_pool = tf.reduce_mean(x, axis=-1, keepdims=True)
        max_pool = tf.reduce_max(x, axis=-1, keepdims=True)
        concat = tf.concat([avg_pool, max_pool], axis=-1)
        attention = tf.nn.sigmoid(self.conv(concat))
        return x * attention

class CBAM(layers.Layer):
    def __init__(self, channels, reduction=8, **kwargs):
        super().__init__(**kwargs)
        self.channels = channels
        self.reduction = reduction

    def build(self, input_shape):
        self.channel_att = ChannelAttention(self.channels, self.reduction)
        self.spatial_att = SpatialAttention()
        super().build(input_shape)

    def call(self, x):
        x = self.channel_att(x)
        x = self.spatial_att(x)
        return x

class Config:
    PATCH_SIZE = 128
    INPUT_CHANNELS = 3
    OUTPUT_CHANNELS = 4

    OUTPUT_CONSTRAINT = 'tanh'
    OUTPUT_MIN = 0.0
    OUTPUT_MAX = 1.0

    DWT_LEVELS = 2
    WAVELET_TYPE = 'haar'

    NUM_HEADS = 8
    NUM_TRANSFORMER_BLOCKS = 5
    EMBED_DIM = 112
    MLP_RATIO = 3
    DROPOUT = 0.1
    DROP_PATH_RATE = 0.15

    USE_SPATIAL_MIXING = True
    USE_CHANNEL_ATTENTION = False
    SPATIAL_KERNEL = 7
    SPATIAL_EXPANSION = 2

    USE_DECODER_ATTENTION = True

    ACTIVATION = 'swish'

def build_wavelet_mst(config=None):
    if config is None:
        config = Config()

    inputs = keras.Input(shape=(config.PATCH_SIZE, config.PATCH_SIZE, config.INPUT_CHANNELS))

    init = _initializer_for(config.ACTIVATION)
    activation_name = _activation_name(config.ACTIVATION)

    rgb_to_nir = layers.Conv2D(
        config.OUTPUT_CHANNELS, 1,
        padding='same',
        use_bias=True,
        kernel_initializer='glorot_uniform',
        bias_initializer=tf.keras.initializers.Constant(0.1),
        dtype='float32',
        name='rgb_to_nir_proj',
    )(inputs)

    dwt_features = DWTFeatureExtractor(
        num_levels=config.DWT_LEVELS,
        output_dim=config.EMBED_DIM // 2,
        activation=config.ACTIVATION,
        name='dwt_encoder',
    )(inputs)

    shallow = layers.Conv2D(64, 3, padding='same', kernel_initializer=init, name='shallow_1')(inputs)
    shallow = layers.Activation(activation_name, name='shallow_act_1')(shallow)
    shallow = layers.Conv2D(config.EMBED_DIM // 2, 3, padding='same',
                            kernel_initializer=init, name='shallow_2')(shallow)
    shallow = layers.Activation(activation_name, name='shallow_act_2')(shallow)

    fused = layers.Concatenate(name='dwt_cnn_fusion')([dwt_features, shallow])
    fused = layers.Conv2D(config.EMBED_DIM, 1, kernel_initializer=init, name='fusion_proj')(fused)
    fused = layers.Activation(activation_name, name='fusion_act')(fused)

    x = fused
    num_blocks = config.NUM_TRANSFORMER_BLOCKS
    dpr = [i * config.DROP_PATH_RATE / max(num_blocks - 1, 1) for i in range(num_blocks)]
    for i in range(num_blocks):
        x = SpectralTransformerBlock(
            embed_dim=config.EMBED_DIM,
            num_heads=config.NUM_HEADS,
            mlp_ratio=config.MLP_RATIO,
            dropout=config.DROPOUT,
            drop_path=dpr[i],
            activation='gelu',
            use_spatial_mixing=getattr(config, 'USE_SPATIAL_MIXING', False),
            spatial_kernel=getattr(config, 'SPATIAL_KERNEL', 7),
            spatial_expansion=getattr(config, 'SPATIAL_EXPANSION', 2),
            use_channel_attention=getattr(config, 'USE_CHANNEL_ATTENTION', False),
            name=f'spectral_block_{i}',
        )(x)

    x = RMSNorm(name='final_norm')(x)
    x = layers.Add(name='local_skip')([x, fused])

    x = layers.Conv2D(64, 3, padding='same', kernel_initializer=init, name='refine_1')(x)
    x = RMSNorm(name='refine_norm_1')(x)
    x = layers.Activation(activation_name, name='refine_act_1')(x)

    if getattr(config, 'USE_DECODER_ATTENTION', False):
        x = CBAM(64, reduction=8, name='decoder_cbam')(x)

    x = layers.Conv2D(32, 3, padding='same', kernel_initializer=init, name='refine_2')(x)
    x = RMSNorm(name='refine_norm_2')(x)
    x = layers.Activation(activation_name, name='refine_act_2')(x)

    residual = layers.Conv2D(
        config.OUTPUT_CHANNELS, 3,
        padding='same',
        kernel_initializer=tf.keras.initializers.TruncatedNormal(stddev=0.01),
        bias_initializer='zeros',
        dtype='float32',
        name='residual_output',
    )(x)

    outputs = layers.Add(name='final_output')([rgb_to_nir, residual])

    constraint = getattr(config, 'OUTPUT_CONSTRAINT', None)
    if constraint == 'clip':
        outputs = layers.Lambda(
            lambda t: tf.clip_by_value(t, config.OUTPUT_MIN, config.OUTPUT_MAX),
            name='output_clip',
        )(outputs)
    elif constraint == 'tanh':
        outputs = layers.Lambda(
            lambda t: (0.5 * (tf.tanh(t) + 1.0)) * (config.OUTPUT_MAX - config.OUTPUT_MIN) + config.OUTPUT_MIN,
            name='output_tanh',
        )(outputs)

    return keras.Model(inputs=inputs, outputs=outputs, name='WaveletMST')
