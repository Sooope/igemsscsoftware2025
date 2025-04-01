import tensorflow as tf
from tensorflow import keras
import load
import datetime
import CBAM


def getModel(num_neurons,class_num,weights=None,base="mnv2"):
    # Choose base model by changing keras.applicatoin._____

    if base == "env2b3":
        # EfficientNetV2B3
        baseModel = keras.applications.EfficientNetV2B3(include_top=False, weights=weights)
    elif base == "rn50":
        # ResNet50
        baseModel = keras.applications.ResNet50(include_top=False, weights=weights)
    elif base == "rn101":
        baseModel = keras.applications.ResNet101(include_top=False, weights=weights)
    elif base == "mnv2":
        # MobileNetV2
        baseModel = keras.applications.MobileNetV2(include_top=False, weights=weights)
    elif base == "iv3":
        # InceptionV3
        baseModel = keras.applications.InceptionV3(include_top=False, weights=weights)
    elif base == "vgg16":
        # VGG16
        baseModel = keras.applications.VGG16(include_top=False, weights=weights)
    elif base == "xcep":
        # Xception
        baseModel = keras.applications.Xception(include_top=False, weights=weights)
    elif base == "dn121":
        # DenseNet121
        baseModel = keras.applications.DenseNet121(include_top=False, weights=weights)
    elif base == "CBAM_RN50":
        # CBAM ResNet50
        baseModel = CBAM.CBAM_ResNet50()
    elif base == "CBAM_RN101":
        # CBAM ResNet101
        baseModel = CBAM.CBAM_ResNet101()
    else:
        print("Wrong model id, fallback to EfficientNetV2B3")
        baseModel = keras.applications.EfficientNetV2B3(include_top=False, weights=weights)

    model = keras.Sequential([load.data_augmentation(), baseModel, keras.layers.Flatten()])
    for i in range(len(num_neurons)):
        model.add(keras.layers.Dense(num_neurons[i], activation="relu"))
    model.add(keras.layers.Dense(class_num, activation="softmax"))

    return model

if __name__=="__main__":
    ds = "Eczema"
    name = "EfficientNetV2B3"
    base = "env2b3"
    Epoches = 10
    Batch_size = 32

    # top parameters
    num_neurons = [1024, 512]

    # weights of basemodel
    weights = None
    train_data, val_data, class_names = load.processImage(
        data_set=ds, batch_size=Batch_size
    )
    model = getModel(
        num_neurons=num_neurons,
        class_num=len(class_names), 
        weights=weights,
        base=base
    )
    model.compile(
        optimizer="adam",
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    log_dir = "logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)

    model.fit(
        train_data,
        validation_data=val_data,
        epochs=Epoches,
        callbacks=[tensorboard_callback],
    )
    model.save(model, name)
