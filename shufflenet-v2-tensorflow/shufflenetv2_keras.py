import keras
# DOĞRU SIRALAMA: Önce Layer importu, sonra custom layer tanımları
from tensorflow.keras.layers import Layer

# Custom ChannelSplit Layer
@keras.saving.register_keras_serializable()
class ChannelSplitLayer(Layer):
    def __init__(self, index, **kwargs):
        super(ChannelSplitLayer, self).__init__(**kwargs)
        self.index = index

    def call(self, x):
        c = tf.shape(x)[-1] // 2
        if self.index == 0:
            return x[..., :c]
        else:
            return x[..., c:]

    def get_config(self):
        config = super().get_config()
        config.update({'index': self.index})
        return config

# Custom ChannelShuffle Layer
@keras.saving.register_keras_serializable()
class ChannelShuffleLayer(Layer):
    def __init__(self, groups=2, **kwargs):
        super(ChannelShuffleLayer, self).__init__(**kwargs)
        self.groups = groups

    def call(self, x):
        batch_size = tf.shape(x)[0]
        height = tf.shape(x)[1]
        width = tf.shape(x)[2]
        channels = tf.shape(x)[3]
        channels_per_group = channels // self.groups
        x = tf.reshape(x, [batch_size, height, width, self.groups, channels_per_group])
        x = tf.transpose(x, [0, 1, 2, 4, 3])
        x = tf.reshape(x, [batch_size, height, width, channels])
        return x

    def get_config(self):
        config = super().get_config()
        config.update({'groups': self.groups})
        return config
# Keras ile ShuffleNetV2 (1.0x, 1.5x, 2.0x destekli, weights=None ile transfer learning)
# Kaynak: https://github.com/eriklindernoren/Keras-ShuffleNetV2 (MIT License)
import tensorflow as tf
from tensorflow.keras.layers import Layer

# Diğer layer ve model importları
from tensorflow.keras.layers import Conv2D, BatchNormalization, Activation, Add, Concatenate, GlobalAveragePooling2D, Dense, Input, DepthwiseConv2D, Lambda
from tensorflow.keras.models import Model

def channel_shuffle(x, groups):
    height, width, in_channels = x.shape[1:]
    channels_per_group = in_channels // groups
    x = tf.reshape(x, [-1, height, width, groups, channels_per_group])
    x = tf.transpose(x, [0, 1, 2, 4, 3])
    x = tf.reshape(x, [-1, height, width, in_channels])
    return x

def shuffle_unit(x, in_channels, out_channels, stride, groups):
    half_channels = out_channels // 2
    if stride == 1:
        x1 = ChannelSplitLayer(0)(x)
        x2 = ChannelSplitLayer(1)(x)
        out = Concatenate()([
            x1,
            _shuffle_branch(x2, half_channels, stride, groups)
        ])
    else:
        out = Concatenate()([
            _shuffle_branch(x, half_channels, stride, groups),
            _shuffle_branch(x, half_channels, stride, groups)
        ])
    out = ChannelShuffleLayer(groups=groups)(out)
    return out

def _shuffle_branch(x, out_channels, stride, groups):
    x = Conv2D(out_channels, 1, padding='same', use_bias=False)(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = DepthwiseConv2D(3, strides=stride, padding='same', use_bias=False)(x)
    x = BatchNormalization()(x)
    x = Conv2D(out_channels, 1, padding='same', use_bias=False)(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    return x

def ShuffleNetV2(input_shape=(224,224,3), n_classes=10, scale_factor=1.5):
    # scale_factor: 0.5, 1.0, 1.5, 2.0
    out_channels_dict = {
        0.5: [24, 48, 96, 192, 1024],
        1.0: [24, 116, 232, 464, 1024],
        1.5: [24, 176, 352, 704, 1024],
        2.0: [24, 244, 488, 976, 2048],
    }
    out_channels = out_channels_dict[scale_factor]
    groups = 2
    inp = Input(shape=input_shape)
    x = Conv2D(out_channels[0], 3, strides=2, padding='same', use_bias=False)(inp)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = tf.keras.layers.MaxPooling2D(pool_size=3, strides=2, padding='same')(x)
    # Stage 2
    for i in range(4):
        stride = 2 if i == 0 else 1
        x = shuffle_unit(x, out_channels[0], out_channels[1], stride, groups)
    # Stage 3
    for i in range(8):
        stride = 2 if i == 0 else 1
        x = shuffle_unit(x, out_channels[1], out_channels[2], stride, groups)
    # Stage 4
    for i in range(4):
        stride = 2 if i == 0 else 1
        x = shuffle_unit(x, out_channels[2], out_channels[3], stride, groups)
    x = Conv2D(out_channels[4], 1, padding='same', use_bias=False)(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = GlobalAveragePooling2D()(x)
    out = Dense(n_classes, activation='softmax')(x)
    model = Model(inputs=inp, outputs=out)
    return model
