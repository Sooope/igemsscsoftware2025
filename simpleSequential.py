import tensorflow as tf
import os
from tensorflow import keras
import numpy as np
import Models.load as load
import pathlib
import matplotlib.pyplot as plt

# os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

trail='E50_Conv33_64_Conv22_128_256_512_FC256_128_64_BN'

data_dir = pathlib.Path("./Set/")

img_height = 256
img_width = 256

epochs=50
units=64

l2=kernel_regularizer=keras.regularizers.l2(1e-3)

(train_ds, val_ds, class_names) = load.processImage(
    data_dir,
    batch_size=16,
    img_height=img_height,
    img_width=img_width,
)

num_classes = len(class_names)

data_augmentation = keras.Sequential(
    [
        keras.layers.RandomFlip(
            "horizontal", input_shape=(img_height, img_width, 3)
        ),
        keras.layers.RandomRotation(0.1),
        keras.layers.RandomZoom(0.1),
    ]
)


def convBlk(output_channel,kernel_size):
    convBlk =keras.Sequential([
            keras.layers.Conv2D(output_channel, (kernel_size,kernel_size), activation="relu", padding ="same"),
            keras.layers.Conv2D(output_channel, (kernel_size,kernel_size), activation="relu", padding ="same"),
            keras.layers.MaxPooling2D(),])
    return convBlk

model = keras.Sequential(
    [
        data_augmentation,
        keras.layers.BatchNormalization(),
        convBlk(64,3),
        convBlk(128,2),
        convBlk(256,2),
        # convBlk(512,2),
    ]
)
model.add(keras.layers.Flatten())

# model.add(keras.layers.Dense(256,activation="relu"))

model.add(keras.layers.Dropout(0.9))
model.add(keras.layers.BatchNormalization())

    
model.add(keras.layers.Dense(128, activation="relu"))
# model.add(keras.layers.Dense(64, activation="relu"))


model.add(keras.layers.Dropout(0.9))

model.add(keras.layers.Dense(num_classes, activation="softmax"))


model.compile(
    optimizer="adam",
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"],
)

log_dir = './logs/'
tb_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1) 

history=model.fit(train_ds, validation_data=val_ds, epochs=epochs,callbacks=[tb_callback])


# acc = history.history['accuracy']
# val_acc = history.history['val_accuracy']

# loss = history.history['loss']
# val_loss = history.history['val_loss']

# epochs_range = range(epochs)

# plt.figure(figsize=(8, 8))
# plt.subplot(1, 2, 1)
# plt.plot(epochs_range, acc, label='Training Accuracy')
# plt.plot(epochs_range, val_acc, label='Validation Accuracy')
# plt.legend(loc='lower right')
# plt.title('Training and Validation Accuracy')

# plt.subplot(1, 2, 2)
# plt.plot(epochs_range, loss, label='Training Loss')
# plt.plot(epochs_range, val_loss, label='Validation Loss')
# plt.legend(loc='upper right')
# plt.title('Training and Validation Loss')
# plt.savefig(trail+'.png')


model.summary()

load.savetf(model,trail)
