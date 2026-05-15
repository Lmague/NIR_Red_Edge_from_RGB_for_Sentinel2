import math

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

def _activation_name(name):
    table = {
        'relu':  'relu',
        'gelu':  'gelu',
        'swish': 'swish',
        'silu':  'swish',
        'mish':  'mish',
    }
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

class SqueezeExcitation(layers.Layer):
    def __init__(self, channels, ratio=4, **kwargs):
        super().__init__(**kwargs)
        self.channels = channels
        self.ratio = ratio

    def build(self, input_shape):
        reduced = max(self.channels // self.ratio, 4)
        self.fc1 = layers.Dense(reduced, kernel_initializer='he_normal')
        self.fc2 = layers.Dense(self.channels, kernel_initializer='glorot_normal')
        super().build(input_shape)

    def call(self, x):
        gap = tf.reduce_mean(x, axis=[1, 2], keepdims=False)
        s = tf.nn.swish(self.fc1(gap))
        s = tf.nn.sigmoid(self.fc2(s))
        s = tf.reshape(s, (-1, 1, 1, self.channels))
        return x * s

    def get_config(self):
        cfg = super().get_config()
        cfg.update({'channels': self.channels, 'ratio': self.ratio})
        return cfg

class InvertedResidualBlock(layers.Layer):
    def __init__(self, channels, expansion=2, drop_path_rate=0.0, **kwargs):
        super().__init__(**kwargs)
        self.channels = channels
        self.expansion = expansion
        self.drop_path_rate = drop_path_rate

    def build(self, input_shape):
        expanded = self.channels * self.expansion

        self.norm = RMSNorm(name='pre_norm')

        self.expand_conv = layers.Conv2D(
            expanded, 1, padding='same',
            kernel_initializer='he_normal',
            use_bias=False, name='expand',
        )
        self.expand_norm = RMSNorm(name='expand_norm')
        self.expand_act = layers.Activation('gelu', name='expand_act')

        self.dwconv = layers.DepthwiseConv2D(
            3, padding='same',
            depthwise_initializer='he_normal',
            use_bias=False, name='dwconv',
        )
        self.dw_norm = RMSNorm(name='dw_norm')
        self.dw_act = layers.Activation('gelu', name='dw_act')

        self.project_conv = layers.Conv2D(
            self.channels, 1, padding='same',
            kernel_initializer='glorot_uniform',
            use_bias=False, name='project',
        )

        self.drop_path = DropPath(self.drop_path_rate) if self.drop_path_rate > 0 else None

        super().build(input_shape)

    def call(self, x, training=False):
        residual = x
        y = self.norm(x)
        y = self.expand_conv(y)
        y = self.expand_norm(y)
        y = self.expand_act(y)

        y = self.dwconv(y)
        y = self.dw_norm(y)
        y = self.dw_act(y)

        y = self.project_conv(y)

        if self.drop_path is not None:
            y = self.drop_path(y, training=training)

        return residual + y

    def get_config(self):
        cfg = super().get_config()
        cfg.update({
            'channels': self.channels,
            'expansion': self.expansion,
            'drop_path_rate': self.drop_path_rate,
        })
        return cfg

class Config:
    PATCH_SIZE = 128
    INPUT_CHANNELS = 3
    OUTPUT_CHANNELS = 4

    OUTPUT_CONSTRAINT = 'tanh'  # 'tanh' | 'clip' | None
    OUTPUT_MIN = 0.0
    OUTPUT_MAX = 1.0

    BASE_DIM = 64
    NUM_BLOCKS = 6
    EXPANSION_RATIO = 2
    SE_RATIO = 4
    DROP_PATH_RATE = 0.1

    ACTIVATION = 'gelu'

def build_micronir(config=None):
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

    x = layers.Conv2D(
        config.BASE_DIM, 3, padding='same',
        kernel_initializer=init, name='stem_conv',
    )(inputs)
    x = RMSNorm(name='stem_norm')(x)
    x = layers.Activation(activation_name, name='stem_act')(x)

    num_blocks = config.NUM_BLOCKS
    dpr = [i * config.DROP_PATH_RATE / max(num_blocks - 1, 1) for i in range(num_blocks)]
    for i in range(num_blocks):
        x = InvertedResidualBlock(
            channels=config.BASE_DIM,
            expansion=config.EXPANSION_RATIO,
            drop_path_rate=dpr[i],
            name=f'invres_block_{i}',
        )(x)

    x = RMSNorm(name='final_norm')(x)
    x = SqueezeExcitation(
        config.BASE_DIM,
        ratio=config.SE_RATIO,
        name='se_block',
    )(x)

    x = layers.Conv2D(
        32, 3, padding='same',
        kernel_initializer=init, name='refine_conv',
    )(x)
    x = RMSNorm(name='refine_norm')(x)
    x = layers.Activation(activation_name, name='refine_act')(x)

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

    return keras.Model(inputs=inputs, outputs=outputs, name='MicroNIR')
