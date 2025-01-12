import tensorflow as tf
from tensorflow import keras
import numpy as np
import load
import pathlib
import matplotlib.pyplot as plt

trail=input("Enter the trail name")

data_dir = pathlib.Path("./eczemads/")

img_height = 180
img_width = 180

epochs=100
layers=1
units=64
dropout=True

(train_ds, val_ds, class_names) = load.processImage(
    data_dir,
    batch_size=32,
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



model = keras.Sequential(
    [
        data_augmentation,
        keras.layers.Conv2D(32, (3,3), activation="relu"),
        keras.layers.MaxPooling2D(),
        # keras.layers.Conv2D(64, (3,3), activation="relu"),
        # keras.layers.MaxPooling2D(),
        # keras.layers.Conv2D(64, (3,3), activation="relu"),
        # keras.layers.MaxPooling2D(),
    ]
)
model.add(keras.layers.Flatten())

# for i in range(layers):
#     model.add(            
#             keras.layers.Dense(
#             units,kernel_regularizer=keras.regularizers.l2(0.001),
#             activation="relu",
#         ),model.add(keras.layers.Dropout(0.5))
# )
    
# model.add(keras.layers.Dense(128, activation="relu"))

# model.add(keras.layers.Dropout(0.5))

# model.add(keras.layers.Dense(256, activation="relu"))

# model.add(keras.layers.Dropout(0.5))


model.add(keras.layers.Dense(num_classes, activation="softmax"))


model.compile(
    optimizer="adam",
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"],
)


history=model.fit(train_ds, validation_data=val_ds, epochs=epochs)

acc = history.history['accuracy']
val_acc = history.history['val_accuracy']

loss = history.history['loss']
val_loss = history.history['val_loss']

epochs_range = range(epochs)

plt.figure(figsize=(8, 8))
plt.subplot(1, 2, 1)
plt.plot(epochs_range, acc, label='Training Accuracy')
plt.plot(epochs_range, val_acc, label='Validation Accuracy')
plt.legend(loc='lower right')
plt.title('Training and Validation Accuracy')

plt.subplot(1, 2, 2)
plt.plot(epochs_range, loss, label='Training Loss')
plt.plot(epochs_range, val_loss, label='Validation Loss')
plt.legend(loc='upper right')
plt.title('Training and Validation Loss')
plt.savefig(trail+'.png')


model.summary()

load.savetf(model)
