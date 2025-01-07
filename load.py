import tensorflow as tf
import pathlib


def processImage(
    data_dir,
    batch_size=32,
    img_height=180,
    img_width=180,
    valSplit=0.2,
    seed=123,
):
    train_ds = tf.keras.utils.image_dataset_from_directory(
        data_dir,
        validation_split=valSplit,
        subset="training",
        seed=seed,
        image_size=(img_height, img_width),
        batch_size=batch_size,
        shuffle=True,
    )
    val_ds = tf.keras.utils.image_dataset_from_directory(
        data_dir,
        validation_split=valSplit,
        subset="validation",
        seed=seed,
        image_size=(img_height, img_width),
        batch_size=batch_size,
    )
    class_names = train_ds.class_names

    normalization_layer = tf.keras.layers.Rescaling(1.0 / 255)

    train_ds = train_ds.map(lambda x, y: (normalization_layer(x), y))
    val_ds = val_ds.map(lambda x, y: (normalization_layer(x), y))

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
