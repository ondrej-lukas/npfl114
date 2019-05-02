#!/usr/bin/env python3
#bfc95faa-444e-11e9-b0fd-00505601122b
#3da961ed-4364-11e9-b0fd-00505601122b

import numpy as np
import tensorflow as tf

from modelnet import ModelNet

# The neural network model
class Network:
    def __init__(self, modelnet, args):
        # TODO: Define a suitable model, and either `.compile` it, or prepare
        # optimizer and loss manually.
        # inputs = tf.keras.layers.Input(shape=[args.modelnet, args.modelnet, args.modelnet,1])
        # hidden = tf.keras.layers.Conv3D(filters=32,kernel_size=(5,5,5), strides=(1,1,1), padding="same" ,activation=None)(inputs)
        # hidden = tf.keras.layers.BatchNormalization()(hidden)
        # hidden = tf.keras.activations.relu(hidden)
        # hidden = tf.keras.layers.MaxPool3D(pool_size=(2, 2, 2), strides=(1,1,1),padding="same")(hidden)
        # hidden = tf.keras.layers.Conv3D(filters=32,kernel_size=(3,3,3), strides=(1,1,1), padding="same" ,activation=None)(hidden)
        # hidden = tf.keras.layers.BatchNormalization()(hidden)
        # hidden = tf.keras.activations.relu(hidden)
        # hidden = tf.keras.layers.MaxPool3D(pool_size=(2, 2, 2), strides=(1,1,1),padding="same")(hidden)
        # hidden = tf.keras.layers.Conv3D(filters=64,kernel_size=(3,3,3), strides=(1,1,1), padding="same" ,activation=None)(hidden)
        # hidden = tf.keras.layers.BatchNormalization()(hidden)
        # hidden = tf.keras.activations.relu(hidden)
        # hidden = tf.keras.layers.Flatten()(hidden)
        # outputs = tf.keras.layers.Dense(len(modelnet.LABELS), activation="softmax")(hidden)

        # Architecture 2
        inputs = tf.keras.layers.Input(shape=[args.modelnet, args.modelnet, args.modelnet,1])
        hidden = tf.keras.layers.AveragePooling3D((2,2,2))(inputs)
        hidden = tf.keras.layers.Conv3D(128, kernel_size=(3,3,3))(hidden)
        hidden = tf.keras.layers.Conv3D(32, kernel_size=(3,3,3))(hidden)
        hidden = tf.keras.layers.MaxPooling3D((2,2,2))(hidden)
        hidden = tf.keras.layers.Flatten()(hidden)
        hidden = tf.keras.layers.ELU()(hidden)
        outputs = tf.keras.layers.Dense(len(modelnet.LABELS), activation="softmax")(hidden)


        self.model = tf.keras.Model(inputs=inputs, outputs=outputs)
        adam = tf.keras.optimizers.Adam()
        self.model.compile(
            optimizer=adam,
            loss=tf.keras.losses.SparseCategoricalCrossentropy(),
            metrics=[tf.keras.metrics.SparseCategoricalAccuracy(name="accuracy")],
        )

        self.tb_callback=tf.keras.callbacks.TensorBoard(args.logdir, update_freq=1000, profile_batch=1)
        self.tb_callback.on_train_end = lambda *_: None

    def train(self, modelnet, args):
        # TODO: Train the network on a given dataset.
        self.model.fit(modelnet.train.data["voxels"], modelnet.train.data["labels"],
        batch_size=args.batch_size, epochs=args.epochs,
        validation_data=(modelnet.dev.data["voxels"], modelnet.dev.data["labels"]),
        callbacks=[self.tb_callback],)
        #raise NotImplementedError()

    def predict(self, dataset, args):
        # TODO: Predict method should return a list/np.ndarray of
        # label probabilities from the test set
        #raise NotImplementedError()
        return self.model.predict(dataset.data["voxels"])

if __name__ == "__main__":
    import argparse
    import datetime
    import os
    import re

    # Parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size", default=64, type=int, help="Batch size.")
    parser.add_argument("--modelnet", default=32, type=int, help="ModelNet dimension.")
    parser.add_argument("--epochs", default=50, type=int, help="Number of epochs.")
    parser.add_argument("--threads", default=0, type=int, help="Maximum number of threads to use.")
    args = parser.parse_args()

    # Fix random seeds and number of threads
    np.random.seed(42)
    tf.random.set_seed(42)
    tf.config.threading.set_inter_op_parallelism_threads(args.threads)
    tf.config.threading.set_intra_op_parallelism_threads(args.threads)

    # Create logdir name
    args.logdir = os.path.join("logs", "{}-{}-{}".format(
        os.path.basename(__file__),
        datetime.datetime.now().strftime("%Y-%m-%d_%H%M%S"),
        ",".join(("{}={}".format(re.sub("(.)[^_]*_?", r"\1", key), value) for key, value in sorted(vars(args).items())))
    ))

    # Load the data
    modelnet = ModelNet(args.modelnet)
    # Create the network and train
    network = Network(modelnet, args)
    network.train(modelnet, args)

    # Generate test set annotations, but in args.logdir to allow parallel execution.
    out_path = "3d_recognition_test.txt"
    if os.path.isdir(args.logdir): out_path = os.path.join(args.logdir, out_path)
    with open(out_path, "w", encoding="utf-8") as out_file:
        for probs in network.predict(modelnet.test, args):
            print(np.argmax(probs), file=out_file)
