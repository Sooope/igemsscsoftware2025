import tensorflow as tf
import pathlib

dir = "./Data/Set"

def processImage(
    data_dir=dir,
    batch_size=32,
    img_height=224,
    img_width=224,
    valSplit=0.2,
    seed=10001,
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


def saveModel(model,name):
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    tflite_model = converter.convert()
    with open(name+".tflite", "wb") as f:
        f.write(tflite_model)

    # converter.optimizations = [tf.lite.Optimize.DEFAULT]
    # tflite_model_quantized = converter.convert()
    # with open('model_quatified.tflite', 'wb') as f:
    #     f.write(tflite_model_quantized)


if __name__ == "__main__":
    data_dir = pathlib.Path("./flower_photos")
    print(processImage(data_dir))

def data_augmentation():
    data_augmentation = tf.keras.Sequential(
        [
            tf.keras.layers.RandomFlip("horizontal"),
            tf.keras.layers.RandomRotation(0.1),
            tf.keras.layers.RandomZoom(0.1),
            tf.keras.layers.RandomContrast(0.1),
            tf.keras.layers.RandomBrightness(0.1),
        ]
    )
    return data_augmentation