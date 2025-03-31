# self-implementation of CBAM
# For more information, please refer to the paper: https://arxiv.org/abs/1807.06521

import tensorflow as tf
from tensorflow import keras
import load

# Channel Attention, global pooling to make channel discriptor, return a 1*1*channel tensor
def channal_attention(input_feature, ratio=8):
    channel = input_feature.shape[-1]
    # Middle MLP
    shared_layer_one = keras.layers.Dense(channel//ratio,
                                          activation='relu',
                                          kernel_initializer='he_normal',
                                          use_bias=True,
                                          bias_initializer='zeros')
    shared_layer_two = keras.layers.Dense(channel,
                                          kernel_initializer='he_normal',
                                          use_bias=True,
                                          bias_initializer='zeros')
    # Pooling
    avg_pool = keras.layers.GlobalAveragePooling2D(keepdims=True)(input_feature)
    avg_pool = shared_layer_one(avg_pool)
    avg_pool = shared_layer_two(avg_pool)
    max_pool = keras.layers.GlobalMaxPooling2D(keepdims=True)(input_feature)
    max_pool = shared_layer_one(max_pool)
    max_pool = shared_layer_two(max_pool)
    cbam_feature = keras.layers.Add()([avg_pool, max_pool])
    cbam_feature = keras.layers.Activation('sigmoid')(cbam_feature)
    cbam_feature = keras.layers.Reshape((1, 1, channel))(cbam_feature)
    return keras.layers.Multiply()([input_feature, cbam_feature])

# Spacial Attention, global pooling to make spacial discriptor, return a h*w*1 tensor
def spacial_attention(input_feature):
    kernel_size = 7
    # Pooling
    avg_pool = keras.layers.Lambda(lambda x: keras.backend.mean(x, axis=3, keepdims=True))(input_feature)
    max_pool = keras.layers.Lambda(lambda x: keras.backend.max(x, axis=3, keepdims=True))(input_feature)
    concat = keras.layers.Concatenate(axis=3)([avg_pool, max_pool])
    # Middle Conv
    cbam_feature = keras.layers.Conv2D(filters=1,
                                       kernel_size=kernel_size,
                                       strides=1,
                                       padding='same',
                                       activation='sigmoid',
                                       kernel_initializer='he_normal',
                                       use_bias=False)(concat)
    return keras.layers.Multiply()([input_feature, cbam_feature])

# CBAM block
def cbam_block(cbam_feature):
    return spacial_attention(channal_attention(cbam_feature))

# Residual bottleneck block with CBAM (For ResNet50+)
class CBAM_BottleneckBlock(keras.Model):

    def __init__(self, channels, dowsampling=False):
        super(CBAM_BottleneckBlock, self).__init__()
        self.conv1 = keras.layers.Conv2D(filters=channels, kernel_size=1, strides=1, padding='same')
        self.bn1 = keras.layers.BatchNormalization()
        self.conv2 = keras.layers.Conv2D(filters=channels, kernel_size=3, strides=1, padding='same')
        self.bn2 = keras.layers.BatchNormalization()
        self.conv3 = keras.layers.Conv2D(filters=channels*4, kernel_size=1, strides=1, padding='same')
        self.bn3 = keras.layers.BatchNormalization()
        # CBAM
        self.cbam = cbam_block
        
        if dowsampling:
            self.shortcut = keras.layers.Conv2D(filters=channels*4, kernel_size=1, strides=2, padding='same')
        else:
            self.shortcut = keras.layers.Lambda(lambda x: x)
    
    def call(self, inputs):
        Residual = self.shortcut(inputs)
        inputs = self.conv1(inputs)
        inputs = self.bn1(inputs)
        inputs = tf.nn.relu(inputs)
        inputs = self.conv2(inputs)
        inputs = self.bn2(inputs)
        inputs = tf.nn.relu(inputs)
        inputs = self.conv3(inputs)
        inputs = self.bn3(inputs)
        inputs = self.cbam(inputs)
        inputs += Residual
        return tf.nn.relu(inputs)
    
# CBAM ResNet builder
class CBAM_ResNets(keras.Model):
    def __init__(self, dims):
        super(CBAM_ResNets, self).__init__()
        self.conv1 = keras.layers.Conv2D(filters=64, kernel_size=7, strides=2, padding='same')
        self.bn = keras.layers.BatchNormalization()
        self.pool = keras.layers.MaxPool2D(pool_size=3, strides=2, padding='same')
        self.conv2 = self.build_resBlock(64,dims[0],True)
        self.conv3 = self.build_resBlock(128,dims[1])
        self.conv4 = self.build_resBlock(256,dims[2])
        self.conv5 = self.build_resBlock(512,dims[3])


    def build_resBlock(self,channels, nums, is_first = False):
        resBlock = keras.Sequential()
        for i in nums:
            if i == 1 and is_first:
                downsampling = True
            else:
                downsampling = False
            resBlock.add(CBAM_BottleneckBlock(channels,downsampling))
        return resBlock

    def call(self, input):
        input = self.bn(self.conv1(input))
        input = self.pool(input)
        input = self.conv2(input)
        input = self.conv3(input)
        input = self.conv4(input)
        input = self.conv5(input)
        return input

def CBAM_ResNet101():
    return CBAM_ResNets([3,4,23,3])

def CBAM_ResNet50():
    return CBAM_ResNets([3,4,6,3])
