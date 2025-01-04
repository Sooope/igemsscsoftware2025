import numpy as np
import os
import PIL
import PIL.Image
import tensorflow as tf
import tensorflow_datasets as tfds
import pathlib
import matplotlib.pyplot as plt


# def get_label(file_path):
#   # Convert the path to a list of path components
#   parts = tf.strings.split(file_path, os.path.sep)
#   # The second to last is the class-directory
#   one_hot = parts[-2] == class_names
#   # Integer encode the label
#   return tf.argmax(one_hot)
#
# def decode_img(img):
#   # Convert the compressed string to a 3D uint8 tensor
#   img = tf.io.decode_jpeg(img, channels=3)
#   # Resize the image to the desired size
#   return tf.image.resize(img, [img_height, img_width])
#
#
# def process_path(file_path):
#   label = get_label(file_path)
#   # Load the raw data from the file as a string
#   img = tf.io.read_file(file_path)
#   img = decode_img(img)
#   return img, label


def processImage(
    data_dir,
    batch_size=32,
    img_height=180,
    img_width=180,
    valSplit=0.2,
    doSplit=True,
    seed=123,
):
    if doSplit:
        train_ds = tf.keras.utils.image_dataset_from_directory(
            data_dir,
            validation_split=valSplit,
            subset="training",
            seed=seed,
            image_size=(img_height, img_width),
            batch_size=batch_size,
        )
        val_ds = tf.keras.utils.image_dataset_from_directory(
            data_dir,
            validation_split=valSplit,
            subset="validation",
            seed=seed,
            image_size=(img_height, img_width),
            batch_size=batch_size,
        )
    else:
        train_ds = tf.keras.utils.image_dataset_from_directory(
            data_dir,
            seed=seed,
            image_size=(img_height, img_width),
            batch_size=batch_size,
        )
    class_names = train_ds.class_names

    normalization_layer = tf.keras.layers.Rescaling(1.0 / 255)

    train_ds = train_ds.map(lambda x, y: (normalization_layer(x), y))
    AUTOTUNE = tf.data.AUTOTUNE

    train_ds = train_ds.cache().prefetch(buffer_size=AUTOTUNE)
    val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)

    return (train_ds, val_ds, class_names)


def savetf(model):
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    tflite_model = converter.convert()
    with open("model.tflite", "wb") as f:
        f.write(tflite_model)

    # converter.optimizations = [tf.lite.Optimize.DEFAULT]
    # tflite_model_quantized = converter.convert()
    # with open('model_quatified.tflite', 'wb') as f:
    #     f.write(tflite_model_quantized)


if __name__ == "__main__":
    data_dir = pathlib.Path("./flower_photos")
    print(processImage(data_dir))
