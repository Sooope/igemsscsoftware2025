import os
import tensorflow as tf
import tensorflow_datasets as tfds
import matplotlib.pyplot as plt
import numpy as np
import load
import pathlib

epochs = 50
data_dir = pathlib.Path("./flower_photos")
train_ds, val_ds, class_names = load.processImage(data_dir)


while True:
    labeltxtGen = input("Generate labels.txt?(y/n):")
    if labeltxtGen == "y":
        o = open("labels.txt", "w")
        for item in class_names:
            o.write(item)
            o.write("\n")
        o.close()
        break
    elif labeltxtGen == "n":
        break


# while True:
#     i = input(f"Epochs? (default={epochs}):")
#     if i == "":
#         break
#     elif i.isdigit():
#         epochs = int(i)
#         break

num_classes = len(class_names)

# HERE DEFINES THE MODEL
model = tf.keras.Sequential(
    [
        tf.keras.layers.Conv2D(32, 3, activation="relu"),
        tf.keras.layers.MaxPooling2D(),
        tf.keras.layers.Conv2D(32, 3, activation="relu"),
        tf.keras.layers.MaxPooling2D(),
        tf.keras.layers.Conv2D(32, 3, activation="relu"),
        tf.keras.layers.MaxPooling2D(),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(128, activation="relu"),
        tf.keras.layers.Dense(num_classes),
    ]
)

# HERE COMPILES
model.compile(
    optimizer="adam",
    loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    metrics=["accuracy"],
)

model.fit(train_ds, validation_data=val_ds, epochs=epochs)

load.savetf(model)
