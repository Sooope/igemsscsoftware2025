import os
import tensorflow as tf
from tensorflow import keras
import tensorflow_datasets as tfds
import matplotlib.pyplot as plt
import numpy as np
import keras_tuner
import load
import pathlib

data_dir = pathlib.Path("./eczemads/")

(train_ds, val_ds, class_names) = load.processImage(
    data_dir,
)


def build_model(hp):
    model = keras.Sequential()
    model.add(
        keras.layers.Dense(hp.Choice("units", [8, 16, 32]), activation="relu")
    )
    model.add(keras.layers.Dense(1, activation="relu"))
    model.compile(loss="mse")
    return model


tuner = keras_tuner.RandomSearch(
    build_model, objective="val_loss", max_trials=5
)

tuner.search(x_train, y_train, epochs=5, validation_data=(x_val, y_val))
best_model = tuner.get_best_models()[0]
