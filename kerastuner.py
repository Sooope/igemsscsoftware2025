import tensorflow as tf
from tensorflow import keras
import numpy as np
import keras_tuner
import load
import pathlib

data_dir = pathlib.Path("./eczemads/")

img_height = 180
img_width = 180

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


def build_model(hp):
    model = keras.Sequential(
        [
            data_augmentation,
            keras.layers.Conv2D(32, 3, activation="relu"),
            keras.layers.MaxPooling2D(),
            keras.layers.Conv2D(32, 3, activation="relu"),
            keras.layers.MaxPooling2D(),
            keras.layers.Conv2D(32, 3, activation="relu"),
            keras.layers.MaxPooling2D(),
            keras.layers.Dropout(0.2),
            keras.layers.Flatten(),
            keras.layers.Dense(
                hp.Int("units", min_value=32, max_value=256, step=32),
                activation="relu",
            ),
            keras.layers.Dense(
                hp.Int("units", min_value=32, max_value=256, step=32),
                activation="relu",
            ),
            keras.layers.Dense(num_classes, activation="softmax"),
        ]
    )
    model.compile(
        optimizer="adam",
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


tuner = keras_tuner.RandomSearch(
    build_model, objective="val_loss", directory="tuner"
)

tuner.search(train_ds, epochs=10, validation_data=val_ds)
best_model = tuner.get_best_models()[0]
best_model.save("./best_model.keras")
