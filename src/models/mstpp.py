import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

class SMSA(layers.Layer):
    def __init__(self, dim, num_heads=1, bias=False, **kwargs):
        super().__init__(**kwargs)
        self.dim = dim
        self.num_heads = num_heads
        self.bias = bias
        self.head_dim = dim // num_heads

    def build(self, input_shape):
        self.qkv = layers.Conv2D(
            self.dim * 3, 1, use_bias=self.bias,
            kernel_initializer='glorot_uniform',
            name='qkv',
        )
        self.proj = layers.Conv2D(
            self.dim, 1, use_bias=self.bias,
            kernel_initializer='glorot_uniform',
            name='proj',
        )
        super().build(input_shape)

    def call(self, x, training=None):
        shape = tf.shape(x)
        B, H, W, C = shape[0], shape[1], shape[2], shape[3]

        qkv = self.qkv(x)
        qkv = tf.reshape(qkv, [B, H * W, 3, self.num_heads, self.head_dim])
        qkv = tf.transpose(qkv, [2, 0, 3, 4, 1])  # (3, B, heads, head_dim, H*W)
        q, k, v = qkv[0], qkv[1], qkv[2]

        seq_len = tf.cast(H * W, tf.float32)
        # Deviation from MST++: the paper learns a per-head temperature here.
        # A fixed 1/seq_len scale was more stable at this tiny embedding dim.
        scale = 1.0 / seq_len

        attn = tf.matmul(q, k, transpose_b=True) * scale
        attn = tf.clip_by_value(attn, -30.0, 30.0)
        attn = tf.nn.softmax(attn, axis=-1)

        out = tf.matmul(attn, v)
        out = tf.transpose(out, [0, 3, 1, 2])
        out = tf.reshape(out, [B, H, W, C])

        return self.proj(out)

class GDFN(layers.Layer):
    def __init__(self, dim, ffn_expansion=2.66, bias=False, **kwargs):
        super().__init__(**kwargs)
        self.dim = dim
        self.hidden_dim = int(dim * ffn_expansion)
        self.bias = bias

    def build(self, input_shape):
        self.project_in = layers.Conv2D(
            self.hidden_dim * 2, 1, use_bias=self.bias,
            kernel_initializer='glorot_uniform',
            name='project_in',
        )
        self.dwconv = layers.DepthwiseConv2D(
            3, padding='same', use_bias=self.bias,
            depthwise_initializer='glorot_uniform',
            name='dwconv',
        )
        self.project_out = layers.Conv2D(
            self.dim, 1, use_bias=self.bias,
            kernel_initializer='glorot_uniform',
            name='project_out',
        )
        super().build(input_shape)

    def call(self, x, training=None):
        x = self.project_in(x)
        x1, x2 = tf.split(x, 2, axis=-1)
        x1 = tf.clip_by_value(x1, -10.0, 10.0)
        x2 = tf.clip_by_value(x2, -10.0, 10.0)
        x = self.dwconv(x1) * tf.nn.gelu(x2)
        x = tf.clip_by_value(x, -10.0, 10.0)
        return self.project_out(x)

class SAB(layers.Layer):
    def __init__(self, dim, num_heads=1, ffn_expansion=2.66, bias=False, **kwargs):
        super().__init__(**kwargs)
        self.dim = dim
        self.num_heads = num_heads
        self.ffn_expansion = ffn_expansion
        self.bias = bias

    def build(self, input_shape):
        self.norm1 = layers.LayerNormalization(epsilon=1e-5, name='norm1')
        self.attn = SMSA(self.dim, self.num_heads, self.bias, name='attn')
        self.norm2 = layers.LayerNormalization(epsilon=1e-5, name='norm2')
        self.ffn = GDFN(self.dim, self.ffn_expansion, self.bias, name='ffn')
        super().build(input_shape)

    def call(self, x, training=None):
        # Deviation from MST++: 0.1 residual scale on both branches, carried
        # over from the original notebook — helped stability at DIM=28.
        x = x + 0.1 * self.attn(self.norm1(x), training=training)
        x = x + 0.1 * self.ffn(self.norm2(x), training=training)
        return x

class Downsample(layers.Layer):
    def __init__(self, dim, **kwargs):
        super().__init__(**kwargs)
        self.dim = dim

    def build(self, input_shape):
        self.conv = layers.Conv2D(
            self.dim * 2, 1, use_bias=False,
            kernel_initializer='glorot_uniform',
            name='conv',
        )
        super().build(input_shape)

    def call(self, x, training=None):
        shape = tf.shape(x)
        B, H, W, C = shape[0], shape[1], shape[2], shape[3]
        x = tf.reshape(x, [B, H // 2, 2, W // 2, 2, C])
        x = tf.transpose(x, [0, 1, 3, 2, 4, 5])
        x = tf.reshape(x, [B, H // 2, W // 2, C * 4])
        return self.conv(x)

class Upsample(layers.Layer):
    def __init__(self, dim, **kwargs):
        super().__init__(**kwargs)
        self.dim = dim

    def build(self, input_shape):
        self.conv = layers.Conv2D(
            self.dim * 2, 1, use_bias=False,
            kernel_initializer='glorot_uniform',
            name='conv',
        )
        super().build(input_shape)

    def call(self, x, training=None):
        x = self.conv(x)
        return tf.nn.depth_to_space(x, 2)

class SST(layers.Layer):
    def __init__(self, dim, num_blocks, num_heads, ffn_expansion=2.66, **kwargs):
        super().__init__(**kwargs)
        self.dim = dim
        self.num_blocks = num_blocks
        self.num_heads = num_heads
        self.ffn_expansion = ffn_expansion

    def build(self, input_shape):
        self.enc0 = [
            SAB(self.dim, self.num_heads[0], self.ffn_expansion, name=f'enc0_sab{i}')
            for i in range(self.num_blocks[0])
        ]
        self.down0 = Downsample(self.dim, name='down0')

        self.enc1 = [
            SAB(self.dim * 2, self.num_heads[1], self.ffn_expansion, name=f'enc1_sab{i}')
            for i in range(self.num_blocks[1])
        ]
        self.down1 = Downsample(self.dim * 2, name='down1')

        self.bottleneck = [
            SAB(self.dim * 4, self.num_heads[2], self.ffn_expansion, name=f'bottleneck_sab{i}')
            for i in range(self.num_blocks[2])
        ]

        self.up1 = Upsample(self.dim * 4, name='up1')
        self.reduce1 = layers.Conv2D(self.dim * 2, 1, use_bias=False, name='reduce1')
        self.dec1 = [
            SAB(self.dim * 2, self.num_heads[1], self.ffn_expansion, name=f'dec1_sab{i}')
            for i in range(self.num_blocks[1])
        ]

        self.up0 = Upsample(self.dim * 2, name='up0')
        self.reduce0 = layers.Conv2D(self.dim, 1, use_bias=False, name='reduce0')
        self.dec0 = [
            SAB(self.dim, self.num_heads[0], self.ffn_expansion, name=f'dec0_sab{i}')
            for i in range(self.num_blocks[0])
        ]

        super().build(input_shape)

    def call(self, x, training=None):
        enc0_out = x
        for blk in self.enc0:
            enc0_out = blk(enc0_out, training=training)

        enc1_in = self.down0(enc0_out, training=training)
        enc1_out = enc1_in
        for blk in self.enc1:
            enc1_out = blk(enc1_out, training=training)

        bn_in = self.down1(enc1_out, training=training)
        bn_out = bn_in
        for blk in self.bottleneck:
            bn_out = blk(bn_out, training=training)

        dec1_in = self.up1(bn_out, training=training)
        dec1_in = self.reduce1(tf.concat([dec1_in, enc1_out], axis=-1))
        dec1_out = dec1_in
        for blk in self.dec1:
            dec1_out = blk(dec1_out, training=training)

        dec0_in = self.up0(dec1_out, training=training)
        dec0_in = self.reduce0(tf.concat([dec0_in, enc0_out], axis=-1))
        dec0_out = dec0_in
        for blk in self.dec0:
            dec0_out = blk(dec0_out, training=training)

        return dec0_out

class _MSTPPCore(layers.Layer):
    def __init__(self, config, **kwargs):
        super().__init__(**kwargs)
        self.config = config
        self.dim = config.DIM
        self.num_stages = config.NUM_STAGES
        self.num_blocks = config.NUM_BLOCKS
        self.num_heads = config.NUM_HEADS
        self.ffn_expansion = config.FFN_EXPANSION

    def build(self, input_shape):
        self.input_proj = layers.Conv2D(
            self.dim, 3, padding='same', use_bias=False,
            kernel_initializer='glorot_uniform',
            name='input_proj',
        )
        self.stages = [
            SST(self.dim, self.num_blocks, self.num_heads, self.ffn_expansion,
                name=f'sst_stage{i}')
            for i in range(self.num_stages)
        ]
        self.output_proj = layers.Conv2D(
            self.config.OUTPUT_CHANNELS, 3, padding='same', use_bias=True,
            kernel_initializer=tf.initializers.TruncatedNormal(stddev=0.02),
            bias_initializer='zeros',
            name='output_proj',
        )
        super().build(input_shape)

    def call(self, x, training=None):
        x = self.input_proj(x)
        for stage in self.stages:
            x = stage(x, training=training) + x
        x = self.output_proj(x)
        return tf.nn.sigmoid(x)

class Config:
    PATCH_SIZE = 128
    INPUT_CHANNELS = 3
    OUTPUT_CHANNELS = 4

    DIM = 28
    NUM_STAGES = 3
    NUM_BLOCKS = [2, 2, 2]
    NUM_HEADS = [1, 2, 4]
    FFN_EXPANSION = 2.66

def build_mstpp(config=None):
    if config is None:
        config = Config()
    inputs = keras.Input(shape=(config.PATCH_SIZE, config.PATCH_SIZE, config.INPUT_CHANNELS))
    outputs = _MSTPPCore(config, name='mstpp')(inputs)
    return keras.Model(inputs, outputs, name='MSTPP')
