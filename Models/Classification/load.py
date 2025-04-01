import tensorflow as tf
import pathlib
from sklearn.utils import class_weight
import numpy as np

dir = "./Data/"
weights_dir = "./Models/Classification/weights/"
tflite_dir = "./Models/Classification/tflite/"
model_dir = "./Models/Classification/models/"


def processImage(
    data_set,
    batch_size=32,
    img_height=224,
    img_width=224,
    valSplit=0.2,
    seed=10001,
):
    data_dir=dir+data_set
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

# Class weight for imbalance dataset
def getClassWeight(train_ds):
    labels = []
    for _,label in train_ds:
        labels.append(label.numpy())
    classes = np.unique(labels)
    class_weights = class_weight.compute_class_weight(
        class_weight="balanced",
        classes=classes,
        y=labels,
    )
    return {cls: weight for cls, weight in zip(classes, class_weights)}
def saveModel(model,name):
    model.save_weights(weights_dir+name)
    model.save(model_dir+name+".h5")
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    tflite_model = converter.convert()
    with open(tflite_dir+name+".tflite", "wb") as f:
        f.write(tflite_model)

    # converter.optimizations = [tf.lite.Optimize.DEFAULT]
    # tflite_model_quantized = converter.convert()
    # with open('model_quatified.tflite', 'wb') as f:
    #     f.write(tflite_model_quantized)

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
