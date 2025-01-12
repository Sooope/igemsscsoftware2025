import tensorflow as tf
from tensorflow import keras
import numpy as np
import keras_tuner
import load
import pathlib
import matplotlib.pylab as plt

epochs = 30

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
        keras.layers.RandomFlip("horizontal", input_shape=(img_height, img_width, 3)),
        keras.layers.RandomRotation(0.1),
        keras.layers.RandomZoom(0.1),
    ]
)


class HyperModel(keras_tuner.HyperModel):
    def build(self, hp):

        Kernal_size = hp.Int("kernel_size", min_value=3, max_value=5, step=1)
        units1=hp.Int("units_1", min_value=32, max_value=256, step=32)
        units2=hp.Int("units_2", min_value=32, max_value=256, step=32)
        unitsC1=hp.Int("unit_C1", min_value=16, max_value=128, step=16)
        unitsC2=hp.Int("unit_C2", min_value=16, max_value=128, step=16)

        model = keras.Sequential([data_augmentation])

        model.add(
            keras.layers.Conv2D(
                hp.Int("unit_C0", min_value=16, max_value=128, step=16),
                kernel_size=(Kernal_size, Kernal_size),
                activation="relu",
            ),
            keras.layers.MaxPooling2D(),
        )

        if hp.Boolean("CNN_1"):
            model.add(
                keras.layers.Conv2D(
                    unitsC1,
                    kernel_size=(Kernal_size, Kernal_size),
                    activation="relu",
                ),
                keras.layers.MaxPooling2D(),
            )

        if hp.Boolean("CNN_2"):
            model.add(
                keras.layers.Conv2D(
                    unitsC2,
                    kernel_size=(Kernal_size, Kernal_size),
                    activation="relu",
                ),
                keras.layers.MaxPooling2D(),
            )

        model.add(keras.layers.Flatten())

        model.add(
            keras.layers.Dense(
                hp.Int("units_0", min_value=32, max_value=256, step=32),
                activation="relu",
            ),
            model.add(keras.layers.Dropout(0.5)),
        )
        if hp.Boolean("layer_1"):
            model.add(
                keras.layers.Dense(
                    units1,
                    activation="relu",
                ),
                model.add(keras.layers.Dropout(0.5)),
            )
        if hp.Boolean("layer_2"):
            model.add(
                keras.layers.Dense(
                    units2,
                    activation="relu",
                ),
                model.add(keras.layers.Dropout(0.5)),
            )

        model.add(keras.layers.Dense(num_classes, activation="softmax"))

        model.compile(
            optimizer="adam",
            loss="sparse_categorical_crossentropy",
            metrics=["accuracy"],
        )
        return model

hm=HyperModel()

tuner = keras_tuner.Hyperband(
    hypermodel=hm,
    objective="val_loss",
    max_epochs=50,
    overwrite=True,
    directory="tuner",
    project_name="CNN_D_U",
)

tuner.search_space_summary()
tuner.search(train_ds, epochs=epochs, validation_data=val_ds)
best_model = tuner.get_best_models()[0]

acc = best_model.history["accuracy"]
val_acc = best_model.history["val_accuracy"]

loss = best_model.history["loss"]
val_loss = best_model.history["val_loss"]

epochs_range = range(epochs)

plt.figure(figsize=(8, 8))
plt.subplot(1, 2, 1)
plt.plot(epochs_range, acc, label="Training Accuracy")
plt.plot(epochs_range, val_acc, label="Validation Accuracy")
plt.legend(loc="lower right")
plt.title("Training and Validation Accuracy")

plt.subplot(1, 2, 2)
plt.plot(epochs_range, loss, label="Training Loss")
plt.plot(epochs_range, val_loss, label="Validation Loss")
plt.legend(loc="upper right")
plt.title("Training and Validation Loss")
plt.savefig(best_model)

best_model.summary()
best_model.save("./best_model.keras")
