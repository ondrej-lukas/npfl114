#!/usr/bin/env python3
import numpy as np
import tensorflow as tf
#bfc95faa-444e-11e9-b0fd-00505601122b
#3da961ed-4364-11e9-b0fd-00505601122b

from cifar10 import CIFAR10
# The neural network model

class Network(tf.keras.Model):
    def __init__(self, args,input_shape=None,n_clases=None):
        inputs = tf.keras.layers.Input(shape=input_shape)

        # TODO: Add CNN layers specified by `args.cnn`, which contains
        # comma-separated list of the following layers:
        # - `C-filters-kernel_size-stride-padding`: Add a convolutional layer with ReLU
        #   activation and specified number of filters, kernel size, stride and padding.
        # - `CB-filters-kernel_size-stride-padding`: Same as `C`, but use batch normalization.
        #   In detail, start with a convolutional layer without bias and activation,
        #   then add batch normalization layer, and finally ReLU activation.
        # - `M-kernel_size-stride`: Add max pooling with specified size and stride.
        # - `R-[layers]`: Add a residual connection. The `layers` contain a specification
        #   of at least one convolutional layer (but not a recursive residual connection `R`).
        #   The input to the specified layers is then added to their output.
        # - `F`: Flatten inputs. Must appear exactly once in the architecture.
        # - `D-hidden_layer_size`: Add a dense layer with ReLU activation and specified size.
        # Produce the results in variable `hidden`.

        hidden = inputs
        for item in re.split(r',(?![^\[]*\])', args.cnn):
            tmp = item.split("-")
            if tmp[0] == "C":
                filters = int(tmp[1])
                kernel_size = int(tmp[2])
                stride = int(tmp[3])
                padding = tmp[4].lower()
    
                hidden = tf.keras.layers.Conv2D(filters=filters, kernel_size=(kernel_size,kernel_size), strides=(stride,stride), padding=padding, activation="relu")(hidden)
            elif tmp[0] == "CB":
                filters = int(tmp[1])
                kernel_size = int(tmp[2])
                stride = int(tmp[3])
                padding = tmp[4].lower()

                hidden = tf.keras.layers.Conv2D(filters=filters,kernel_size=(kernel_size,kernel_size), strides=(stride,stride), padding=padding, activation=None, use_bias=False)(hidden)
                hidden = tf.keras.layers.BatchNormalization()(hidden)
                hidden = tf.keras.activations.relu(hidden)
            elif tmp[0] == "M":
                pool_size = int(tmp[1])
                stride = int(tmp[2])
                hidden = tf.keras.layers.MaxPool2D(pool_size=(pool_size, pool_size), strides=(stride, stride))(hidden)
            elif tmp[0] == "R":
                #get the layers R-[l1,l2,...]
                layers = item.split('-', 1)[1]
                layers = layers[1:-1]
                layers = layers.split(",")
                out = hidden
                for l in layers:
                    setup = l.split("-")
                    filters = int(setup[1])
                    kernel_size = int(setup[2])
                    stride = int(setup[3])
                    padding = setup[4]
                
                    out = tf.keras.layers.Conv2D(filters=filters, kernel_size=(kernel_size,kernel_size), strides=(stride,stride), padding=padding, activation="relu")(out)
                hidden = tf.keras.layers.add([out, hidden])
            elif tmp[0] == "F":
                hidden = tf.keras.layers.Flatten()(hidden)
            elif tmp[0] == "D":
                hidden = tf.keras.layers.Dense(int(tmp[1]), activation="relu")(hidden)
            elif tmp[0] == "DR":
                hidden = tf.keras.layers.Dropout(float(tmp[1]))(hidden)
            print("Adding:",hidden)

        # Add the final output layer
        outputs = tf.keras.layers.Dense(n_clases, activation=tf.nn.softmax)(hidden)

        super().__init__(inputs=inputs, outputs=outputs)

        self.compile(
            optimizer=tf.keras.optimizers.Adam(),
            loss=tf.keras.losses.SparseCategoricalCrossentropy(),
            metrics=[tf.keras.metrics.SparseCategoricalAccuracy(name="accuracy")],
        )

        self.tb_callback=tf.keras.callbacks.TensorBoard(args.logdir, update_freq=1000, profile_batch=1)
        self.tb_callback.on_train_end = lambda *_: None

    def train(self, dataset, args):
        self.fit(
            dataset.train.data["images"], dataset.train.data["labels"],
            batch_size=args.batch_size, epochs=args.epochs,
            validation_data=(dataset.dev.data["images"], dataset.dev.data["labels"]),
            callbacks=[self.tb_callback],
        )

    def test(self, mnist, args):
        test_logs = self.evaluate(dataset.test.data["images"], dataset.test.data["labels"], batch_size=args.batch_size)
        self.tb_callback.on_epoch_end(1, dict(("val_test_" + metric, value) for metric, value in zip(self.metrics_names, test_logs)))
        return test_logs[self.metrics_names.index("accuracy")]

if __name__ == "__main__":
    import argparse
    import datetime
    import os
    import re

    # Parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size", default=50, type=int, help="Batch size.")
    parser.add_argument("--epochs", default=30, type=int, help="Number of epochs.")
    parser.add_argument("--threads", default=1, type=int, help="Maximum number of threads to use.")
    args = parser.parse_args()

    # Fix random seeds
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

    # Load data
    cifar = CIFAR10()
    args.cnn = "CB-32-3-1-same,CB-32-3-1-same,M-2-2,DR-0.2,CB-64-3-1-same,CB-64-3-1-same,M-2-2,DR-0.3,CB-128-3-1-same,CB-128-3-1-same,M-2-2,DR-0.4,F"
    # Create the network and train
    network = Network(args,input_shape=[32,32,3],n_clases=10)
    network.train(cifar, args)

    # Generate test set annotations, but in args.logdir to allow parallel execution.
    with open(os.path.join(args.logdir, "cifar_competition_test.txt"), "w", encoding="utf-8") as out_file:
        for probs in network.predict(cifar.test.data["images"], batch_size=args.batch_size):
            print(np.argmax(probs), file=out_file)
