import tensorflow as tf
from tensorflow import keras

import load
import pathlib
import matplotlib.pyplot as plt

# import numpy as np
# import keras_tuner

data_dir = pathlib.Path("./eczemads/")

img_height = 180
img_width = 180

(train_ds, val_ds, class_names) = load.processImage(
    data_dir,
    batch_size=256,
    img_height=img_height,
    img_width=img_width,
)

num_classes = len(class_names)


data_augmentation = keras.Sequential(
    [
        keras.layers.RandomFlip(
            "horizontal", input_shape=(img_height, img_width)
        ),
        keras.layers.RandomRotation(0.1),
        keras.layers.RandomZoom(0.1),
    ]
)


# def build_model(hp):
#     model = keras.Sequential(
#         [
#             data_augmentation,
#             keras.layers.Conv2D(32, 3, activation="relu"),
#             keras.layers.MaxPooling2D(),
#             keras.layers.Conv2D(32, 3, activation="relu"),
#             keras.layers.MaxPooling2D(),
#             keras.layers.Conv2D(32, 3, activation="relu"),
#             keras.layers.MaxPooling2D(),
#             keras.layers.Dropout(0.2),
#             keras.layers.Flatten(),
#             keras.layers.Dense(
#                 hp.Int("units", min_value=32, max_value=256, step=32),
#                 activation="relu",
#             ),
#             keras.layers.Dense(num_classes, activation="relu"),
#         ]
#     )
#     model.compile(
#         optimizer="adam",
#         loss="sparse_categorical_crossentropy",
#         metrics=["accuracy"],
#     )
#     return model


model = keras.Sequential(
    [
        keras.layers.Conv2D(32, 3, activation="relu"),
        keras.layers.MaxPooling2D(),
        keras.layers.Conv2D(64, 3, activation="relu"),
        keras.layers.MaxPooling2D(),
        keras.layers.Conv2D(128, 3, activation="relu"),
        keras.layers.Flatten(),
        keras.layers.Dense(64, activation="relu"),
        keras.layers.Dense(num_classes, activation="relu"),
    ]
)
# tuner = keras_tuner.RandomSearch(
#     build_model, objective="val_loss", directory="tuner"
# )
# tuner.search(train_ds, epochs=7, validation_data=val_ds)
# best_model = tuner.get_best_models()[0]
# best_model.save("./best_model.keras")

model.compile(
    optimizer="adam",
    loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    metrics=["accuracy"],
)

history = model.fit(train_ds, validation_data=val_ds, epochs=2)
plt.plot(history.history["accuracy"])
plt.plot(history.history["val_accuracy"])
plt.title("model accuracy")
plt.ylabel("accuracy")
plt.xlabel("epoch")
plt.legend(["train", "val"], loc="upper left")
plt.show()

plt.plot(history.history["loss"])
plt.plot(history.history["val_loss"])
plt.title("model loss")
plt.ylabel("loss")
plt.xlabel("epoch")
plt.legend(["train", "val"], loc="upper left")
plt.show()
