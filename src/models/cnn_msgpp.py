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

class MSGPPMixBlock(layers.Layer):
    def __init__(self, channels, num_heads, mlp_ratio=4, drop_rate=0.0, drop_path=0.0,
                 spatial_kernel=7, spatial_expansion=2, **kwargs):
        super().__init__(**kwargs)
        self.channels = channels
        self.num_heads = num_heads
        self.mlp_ratio = mlp_ratio
        self.drop_rate = drop_rate
        self.drop_path_rate = drop_path
        self.spatial_kernel = spatial_kernel
        self.spatial_expansion = spatial_expansion

    def build(self, input_shape):
        init = _initializer_for('gelu')

        self.norm_spatial = RMSNorm(name='norm_spatial')
        self.norm_attn = RMSNorm(name='norm_attn')
        self.norm_mlp = RMSNorm(name='norm_mlp')

        self.dwconv = layers.DepthwiseConv2D(
            self.spatial_kernel, padding='same',
            depthwise_initializer='he_normal', name='dwconv',
        )
        self.pw_expand = layers.Conv2D(
            self.channels * self.spatial_expansion, 1, padding='same',
            kernel_initializer='he_normal', name='pw_expand',
        )
        self.spatial_act = layers.Activation('gelu', name='spatial_gelu')
        self.pw_project = layers.Conv2D(
            self.channels, 1, padding='same',
            kernel_initializer='he_normal', name='pw_project',
        )

        self.spectral_attn = TrueSpectralAttention(
            self.channels, self.num_heads, qk_norm=True, name='spectral_attn',
        )

        mlp_hidden = self.channels * self.mlp_ratio
        self.mlp_expand = layers.Conv2D(mlp_hidden, 1, padding='same',
                                        kernel_initializer=init, name='mlp_expand')
        self.mlp_act = layers.Activation('gelu', name='mlp_gelu')
        self.mlp_project = layers.Conv2D(self.channels, 1, padding='same',
                                         kernel_initializer=init, name='mlp_project')
        self.mlp_drop = layers.Dropout(self.drop_rate) if self.drop_rate > 0 else None

        self.drop_path = DropPath(self.drop_path_rate) if self.drop_path_rate > 0 else None

        super().build(input_shape)

    def _maybe_drop_path(self, x, training):
        if self.drop_path is None:
            return x
        return self.drop_path(x, training=training)

    def call(self, x, training=False):
        y = self.norm_spatial(x)
        y = self.dwconv(y)
        y = self.pw_expand(y)
        y = self.spatial_act(y)
        y = self.pw_project(y)
        x = x + self._maybe_drop_path(y, training=training)

        y = self.norm_attn(x)
        y = self.spectral_attn(y, training=training)
        x = x + self._maybe_drop_path(y, training=training)

        y = self.norm_mlp(x)
        y = self.mlp_expand(y)
        y = self.mlp_act(y)
        if self.mlp_drop is not None:
            y = self.mlp_drop(y, training=training)
        y = self.mlp_project(y)
        if self.mlp_drop is not None:
            y = self.mlp_drop(y, training=training)
        x = x + self._maybe_drop_path(y, training=training)

        return x

class Config:
    PATCH_SIZE = 128
    INPUT_CHANNELS = 3
    OUTPUT_CHANNELS = 4

    OUTPUT_CONSTRAINT = 'tanh'
    OUTPUT_MIN = 0.0
    OUTPUT_MAX = 1.0

    BASE_DIM = 112
    NUM_HEADS = 8
    NUM_MSGPP_BLOCKS = 4
    MLP_RATIO = 4
    DROPOUT = 0.1
    DROP_PATH_RATE = 0.15

    SPATIAL_KERNEL = 7
    SPATIAL_EXPANSION = 2
    STEM_RMSNORM = True

    ACTIVATION = 'gelu'

def build_cnn_msgpp_decoder(config=None):
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

    x = layers.Conv2D(64, 3, padding='same', kernel_initializer=init, name='stem_conv_1')(inputs)
    if getattr(config, 'STEM_RMSNORM', True):
        x = RMSNorm(name='stem_norm_1')(x)
    x = layers.Activation(activation_name, name='stem_act_1')(x)

    x = layers.Conv2D(config.BASE_DIM, 3, padding='same', kernel_initializer=init,
                      name='stem_conv_2')(x)
    if getattr(config, 'STEM_RMSNORM', True):
        x = RMSNorm(name='stem_norm_2')(x)
    x = layers.Activation(activation_name, name='stem_act_2')(x)

    num_blocks = config.NUM_MSGPP_BLOCKS
    dpr = [i * config.DROP_PATH_RATE / max(num_blocks - 1, 1) for i in range(num_blocks)]
    for i in range(num_blocks):
        x = MSGPPMixBlock(
            channels=config.BASE_DIM,
            num_heads=config.NUM_HEADS,
            mlp_ratio=config.MLP_RATIO,
            drop_rate=config.DROPOUT,
            drop_path=dpr[i],
            spatial_kernel=config.SPATIAL_KERNEL,
            spatial_expansion=config.SPATIAL_EXPANSION,
            name=f'msgpp_block_{i}',
        )(x)

    x = RMSNorm(name='final_norm')(x)

    x = layers.Conv2D(64, 3, padding='same', kernel_initializer=init, name='refine_1')(x)
    x = RMSNorm(name='refine_norm_1')(x)
    x = layers.Activation(activation_name, name='refine_act_1')(x)

    x = layers.Conv2D(32, 3, padding='same', kernel_initializer=init, name='refine_2')(x)
    x = RMSNorm(name='refine_norm_2')(x)
    x = layers.Activation(activation_name, name='refine_act_2')(x)

    residual = layers.Conv2D(
        config.OUTPUT_CHANNELS, 3,
        padding='same',
        kernel_initializer=tf.keras.initializers.TruncatedNormal(stddev=0.01),
        bias_initializer=tf.keras.initializers.Constant(-0.6689549),
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

    return keras.Model(inputs=inputs, outputs=outputs, name='CNN_MSGPP_Decoder')
