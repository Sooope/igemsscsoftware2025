import tensorflow as tf
from tensorflow import keras
import load
import datetime

name = "EfficientNetV2B3"
Epoches = 10
Batch_size = 32
# Choose base model by changing keras.applicatoin._____

# EfficientNetV2B3
baseModel = keras.applications.EfficientNetV2B3(
    include_top=False, weights=None
)
#ResNet50
# baseModel = keras.applications.ResNet50(
#     include_top=False, weights=None)
# MobileNetV2
# baseModel = keras.applications.MobileNetV2(
#     include_top=False, weights=None)
# InceptionV3
# baseModel = keras.applications.InceptionV3(
#     include_top=False, weights=None)
# VGG16
# baseModel = keras.applications.VGG16(
#     include_top=False, weights=None)
# Xception
# baseModel = keras.applications.Xception(
#     include_top=False, weights=None)
# DenseNet121
# baseModel = keras.applications.DenseNet121(
#     include_top=False, weights=None)

train_data, val_data, class_names = load.processImage(batch_size=Batch_size)
class_num = len(class_names)


model = keras.Sequential(
    [
        load.data_augmentation(),
        baseModel,
        keras.layers.Flatten(),
        keras.layers.Dense(1000, activation="relu"),
        keras.layers.Dense(class_num, activation="softmax"),
    ]
)

model.compile(
    optimizer="adam",
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"],
    
)
log_dir = "logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)

model.fit(train_data, validation_data=val_data, epochs=Epoches, callbacks=[tensorboard_callback])
model.save(name)



