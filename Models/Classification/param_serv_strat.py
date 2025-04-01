# https://www.tensorflow.org/api_docs/python/tf/distribute/Server
# https://www.tensorflow.org/tutorials/distribute/parameter_server_training#clusters_in_the_real_world

import tensorflow as tf
import os
import json
from CNNs import getModel
import load
import datetime
from functools import partial

config = './Models/Classification/TF_CONFIG.json'

weights = None

with open(config, 'r') as src:
    confjson = json.load(src)
os.environ["TF_CONFIG"] = json.dumps(confjson)

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "0"  # Enable full logging
os.environ["GRPC_VERBOSITY"] = "DEBUG"    # Enable gRPC debug logs

# Class weight
# WARNING: THIS WILL CONSUME A LOT OF MEMORY
# is_class_weight = False

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
    per_worker_batch_size = Batch_size // strategy.num_replicas_in_sync
    
    coodinator = tf.distribute.experimental.coordinator.ClusterCoordinator(strategy)
    
    
    def dataset_fn(input_context=None, is_training=True):
    # input_context is sometimes provided by TF for sharding hints, may not be needed here
    # Calculate batch size again inside the function in case it's needed.
        effective_batch_size = Batch_size
        if input_context:
                effective_batch_size = input_context.get_per_replica_batch_size(Batch_size)

        # This function runs ON EACH WORKER
        # We only need one of the datasets (train or val) per function.
        train_ds, val_ds, _ = load.processImage(
            data_set=ds, batch_size=effective_batch_size 
        )
        if is_training:
            # Ensure dataset repeats indefinitely for training
            return train_ds.repeat()
        else:
            return val_ds
    
    per_worker_val_dataset_fn=partial(dataset_fn,is_training=False)
    per_worker_train_dataset_fn=partial(dataset_fn,is_training=True)
    
    per_worker_val_dataset=coodinator.create_per_worker_dataset(per_worker_val_dataset_fn)
    per_worker_train_dataset=coodinator.create_per_worker_dataset(per_worker_train_dataset_fn)
    
    _,_, class_names = load.processImage(
        data_set=ds, batch_size=Batch_size
    )
        
    # # calculate class weight
    # if is_class_weight:
    #     class_weight = load.getClassWeight(train_data)
    # else:
    #     class_weight = None

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
        print("Model build in scope done")

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
        per_worker_train_dataset,
        validation_data=per_worker_val_dataset,
        epochs=Epoches,
        callbacks=callbacks,
        # class_weight=class_weight,
    )
    load.saveModel(model, name)

