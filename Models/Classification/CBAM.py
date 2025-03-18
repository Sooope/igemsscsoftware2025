#self-implementation of CBAM
import tensorflow as tf
from tensorflow import keras
import load

def channal_attention(input_feature, ratio=8):
    channel = input_feature.shape[-1]
    shared_layer_one = keras.layers.Dense(channel//ratio,
                                          activation='relu',
                                          kernel_initializer='he_normal',
                                          use_bias=True,
                                          bias_initializer='zeros')
    shared_layer_two = keras.layers.Dense(channel,
                                          kernel_initializer='he_normal',
                                          use_bias=True,
                                          bias_initializer='zeros')
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

def spacial_attention(input_feature):
    kernel_size = 7
    avg_pool = keras.layers.Lambda(lambda x: keras.backend.mean(x, axis=3, keepdims=True))(input_feature)
    max_pool = keras.layers.Lambda(lambda x: keras.backend.max(x, axis=3, keepdims=True))(input_feature)
    concat = keras.layers.Concatenate(axis=3)([avg_pool, max_pool])
    cbam_feature = keras.layers.Conv2D(filters=1,
                                       kernel_size=kernel_size,
                                       strides=1,
                                       padding='same',
                                       activation='sigmoid',
                                       kernel_initializer='he_normal',
                                       use_bias=False)(concat)
    return keras.layers.Multiply()([input_feature, cbam_feature])

def cbam_block(cbam_feature):
    return spacial_attention(channal_attention(cbam_feature))