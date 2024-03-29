from functools import wraps

import tensorflow as tf
from tensorflow.keras import backend as K
from tensorflow.keras.initializers import RandomNormal
from tensorflow.keras.layers import (Add, BatchNormalization, Concatenate,
                                     Conv2D, Layer, MaxPooling2D,
                                     ZeroPadding2D)
from tensorflow.keras.regularizers import l2
from utils.utils import compose


class SiLU(Layer):
    def __init__(self, **kwargs):
        super(SiLU, self).__init__(**kwargs)
        self.supports_masking = True

    def call(self, inputs):
        return inputs * K.sigmoid(inputs)

    def get_config(self):
        config = super(SiLU, self).get_config()
        return config

    def compute_output_shape(self, input_shape):
        return input_shape

class Focus(Layer):
    def __init__(self):
        super(Focus, self).__init__()

    def compute_output_shape(self, input_shape):
        return (input_shape[0], input_shape[1] // 2 if input_shape[1] != None else input_shape[1], input_shape[2] // 2 if input_shape[2] != None else input_shape[2], input_shape[3] * 4)

    def call(self, x):
        return tf.concat(
            [x[...,  ::2,  ::2, :],
             x[..., 1::2,  ::2, :],
             x[...,  ::2, 1::2, :],
             x[..., 1::2, 1::2, :]],
             axis=-1
        )
#------------------------------------------------------#
#   Single Convolution DarknetConv2D
#   If the step size is 2, set the padding method yourself.
#------------------------------------------------------#
@wraps(Conv2D)
def DarknetConv2D(*args, **kwargs):
    darknet_conv_kwargs = {'kernel_initializer' : RandomNormal(stddev=0.02), 'kernel_regularizer' : l2(5e-4)}
    darknet_conv_kwargs['padding'] = 'valid' if kwargs.get('strides')==(2, 2) else 'same'
    darknet_conv_kwargs.update(kwargs)
    return Conv2D(*args, **darknet_conv_kwargs)

#---------------------------------------------------#
#   Convolution block -> convolution + normalization + activation function
#   DarknetConv2D + BatchNormalization + SiLU
#---------------------------------------------------#
def DarknetConv2D_BN_SiLU(*args, **kwargs):
    no_bias_kwargs = {'use_bias': False}
    no_bias_kwargs.update(kwargs)
    if "name" in kwargs.keys():
        no_bias_kwargs['name'] = kwargs['name'] + '.conv'
    return compose(
        DarknetConv2D(*args, **no_bias_kwargs),
        BatchNormalization(momentum = 0.97, epsilon = 0.001,name = kwargs['name'] + '.bn'),
        SiLU())

def SPPBottleneck(x, out_channels, name = ""):
    #---------------------------------------------------#
    #   The SPP structure is used, i.e. stacking after max-pooling of different scales.
    #---------------------------------------------------#
    x = DarknetConv2D_BN_SiLU(out_channels // 2, (1, 1), name = name + '.conv1')(x)
    maxpool1 = MaxPooling2D(pool_size=(5, 5), strides=(1, 1), padding='same')(x)
    maxpool2 = MaxPooling2D(pool_size=(9, 9), strides=(1, 1), padding='same')(x)
    maxpool3 = MaxPooling2D(pool_size=(13, 13), strides=(1, 1), padding='same')(x)
    x = Concatenate()([x, maxpool1, maxpool2, maxpool3])
    x = DarknetConv2D_BN_SiLU(out_channels, (1, 1), name = name + '.conv2')(x)
    return x

def Bottleneck(x, out_channels, shortcut=True, name = ""):
    y = compose(
            DarknetConv2D_BN_SiLU(out_channels, (1, 1), name = name + '.conv1'),
            DarknetConv2D_BN_SiLU(out_channels, (3, 3), name = name + '.conv2'))(x)
    if shortcut:
        y = Add()([x, y])
    return y

def CSPLayer(x, num_filters, num_blocks, shortcut=True, expansion=0.5, name=""):
    hidden_channels = int(num_filters * expansion)  # hidden channels
    #----------------------------------------------------------------#
    #   The main part will cycle num_blocks, and the inner part of the cycle is the residual structure.
    #----------------------------------------------------------------#
    x_1 = DarknetConv2D_BN_SiLU(hidden_channels, (1, 1), name = name + '.conv1')(x)
    #--------------------------------------------------------------------#
    #   Then create a large residual edge shortconv, which bypasses many residual structures
    #--------------------------------------------------------------------#
    x_2 = DarknetConv2D_BN_SiLU(hidden_channels, (1, 1), name = name + '.conv2')(x)
    for i in range(num_blocks):
        x_1 = Bottleneck(x_1, hidden_channels, shortcut=shortcut, name = name + '.m.' + str(i))
    #----------------------------------------------------------------#
    #   Stack the large residual edges back again
    #----------------------------------------------------------------#
    route = Concatenate()([x_1, x_2])

    #----------------------------------------------------------------#
    #   Finally, the number of channels is integrated
    #----------------------------------------------------------------#
    return DarknetConv2D_BN_SiLU(num_filters, (1, 1), name = name + '.conv3')(route)

def resblock_body(x, num_filters, num_blocks, expansion=0.5, shortcut=True, last=False, name = ""):
    #----------------------------------------------------------------#
    #   Height and width compression using ZeroPadding2D and a stride 2x2 convolutional block
    #----------------------------------------------------------------#

    # 320, 320, 64 => 160, 160, 128
    x = ZeroPadding2D(((1, 0),(1, 0)))(x)
    x = DarknetConv2D_BN_SiLU(num_filters, (3, 3), strides = (2, 2), name = name + '.0')(x)
    if last:
        x = SPPBottleneck(x, num_filters, name = name + '.1')
    return CSPLayer(x, num_filters, num_blocks, shortcut=shortcut, expansion=expansion, name = name + '.1' if not last else name + '.2')

#---------------------------------------------------#
#   The main part of CSPdarknet
#   Input is a 640x640x3 image
#   The output is three valid feature layers
#---------------------------------------------------#
def darknet_body(x, dep_mul, wid_mul):
    base_channels   = int(wid_mul * 64)  # 64
    base_depth      = max(round(dep_mul * 3), 1)  # 3
    # 640, 640, 3 => 320, 320, 12
    x = Focus()(x)
    # 320, 320, 12 => 320, 320, 64
    x = DarknetConv2D_BN_SiLU(base_channels, (3, 3), name = 'backbone.backbone.stem.conv')(x)
    # 320, 320, 64 => 160, 160, 128
    x = resblock_body(x, base_channels * 2, base_depth, name = 'backbone.backbone.dark2')
    # 160, 160, 128 => 80, 80, 256
    x = resblock_body(x, base_channels * 4, base_depth * 3, name = 'backbone.backbone.dark3')
    feat1 = x
    # 80, 80, 256 => 40, 40, 512
    x = resblock_body(x, base_channels * 8, base_depth * 3, name = 'backbone.backbone.dark4')
    feat2 = x
    # 40, 40, 512 => 20, 20, 1024
    x = resblock_body(x, base_channels * 16, base_depth, shortcut=False, last=True, name = 'backbone.backbone.dark5')
    feat3 = x
    return feat1,feat2,feat3

