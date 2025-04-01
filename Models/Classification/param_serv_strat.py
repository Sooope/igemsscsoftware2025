# https://www.tensorflow.org/api_docs/python/tf/distribute/Server
# https://www.tensorflow.org/tutorials/distribute/parameter_server_training#clusters_in_the_real_world

import tensorflow as tf
import os
import json
from CNNs import getModel
import load
import datetime

tf.config.optimizer.set_jit(False)

config = 'Models/Classification/TF_CONFIG.json'

weights = None

with open(config, 'r') as src:
    confjson = json.load(src)
os.environ["TF_CONFIG"] = json.dumps(confjson)

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "0"  # Enable full logging
os.environ["GRPC_VERBOSITY"] = "DEBUG"    # Enable gRPC debug logs

cluster_resolver = tf.distribute.cluster_resolver.TFConfigClusterResolver()
if cluster_resolver.task_type in ("worker", "ps"):
    # Start a TensorFlow server and wait.

    # Set the environment variable to allow reporting worker and ps failure to the
    # coordinator. This is a workaround and won't be necessary in the future.
    os.environ["GRPC_FAIL_FAST"] = "use_caller"

    # NOTE: This part might need extra config
    server = tf.distribute.Server(
        cluster_resolver.cluster_spec(),
        job_name=cluster_resolver.task_type,
        task_index=cluster_resolver.task_id,
        protocol=cluster_resolver.rpc_layer or "grpc",
        start=True)
    server.join()
elif cluster_resolver.task_type == "evaluator":
    # Run sidecar evaluation
    # Not written yet
    pass
else:
    # Run the coordinator.

    ds = "Eczema"
    name = "ResNet50"
    base = "rn50"
    Epoches = 10
    Batch_size = 32

    # Number of PS
    NUM_PS=1

    # top parameters
    num_neurons = [1000]

    # weights of basemodel
    weights = None

    variable_partitioner = (
        tf.distribute.experimental.partitioners.MinSizePartitioner(
            min_shard_bytes=(256 << 10),
            max_shards=NUM_PS))

    strategy = tf.distribute.ParameterServerStrategy(
        cluster_resolver,
        variable_partitioner=variable_partitioner
    )

    train_data, val_data, class_names = load.processImage(
        data_set=ds, batch_size=Batch_size
    )

    with strategy.scope():
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

    working_dir = "tmp/fit/"
    log_dir = os.path.join(working_dir,"logs/",datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))
    ckpt_filepath = os.path.join(working_dir, "ckpt.keras")
    backup_dir = os.path.join(working_dir, "backup")

    # tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)
    callbacks = [
        tf.keras.callbacks.TensorBoard(log_dir=log_dir),
        tf.keras.callbacks.ModelCheckpoint(filepath=ckpt_filepath),
        tf.keras.callbacks.BackupAndRestore(backup_dir=backup_dir),
    ]

    model.fit(
        train_data,
        validation_data=val_data,
        epochs=Epoches,
        callbacks=callbacks,
        class_weight=load.getClassWeight(train_data),
    )
    model.save(model, name)

