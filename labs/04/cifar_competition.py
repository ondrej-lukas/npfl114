#!/usr/bin/env python3
import numpy as np
import tensorflow as tf
#bfc95faa-444e-11e9-b0fd-00505601122b
#3da961ed-4364-11e9-b0fd-00505601122b

from cifar10 import CIFAR10
# The neural network model
import warnings

import requests


class SlackCallback(tf.keras.callbacks.Callback):
    def __init__(self, channel='#hulk'):
        super(SlackCallback, self).__init__()
        
    
    def on_train_end(self, logs=None):
        #Report best results and plot losses
        #self.send_message(f'Job started by{self.user} finished at: {time.time()}')
        r = requests.post("https://hooks.slack.com/services/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", data={"payload":"{\"channel\": \"#hulk\", \"username\": \"TensorBot\", \"text\": \"Job finished.\", \"icon_emoji\": \":hulk:\"}"})




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
        weight_decay = 1e-4
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

                hidden = tf.keras.layers.Conv2D(filters=filters,kernel_size=(kernel_size,kernel_size), strides=(stride,stride), padding=padding, activation=None, use_bias=False,kernel_regularizer=tf.keras.regularizers.l2(weight_decay))(hidden)
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
                    print("Adding (inside residual):",out)
                hidden = tf.keras.layers.add([out, hidden])
            elif tmp[0] == "F":
                hidden = tf.keras.layers.Flatten()(hidden)
            elif tmp[0] == "D":
                hidden = tf.keras.layers.Dense(int(tmp[1]), activation="tanh")(hidden)
            elif tmp[0] == "DR":
                hidden = tf.keras.layers.Dropout(float(tmp[1]))(hidden)
            elif tmp[0] == "A":
                pool_size = int(tmp[1])
                stride = int(tmp[2])
                hidden = tf.keras.layers.AveragePooling2D(pool_size=(pool_size, pool_size), strides=(stride, stride))(hidden)
            print("Adding:",hidden)
        # Add the final output layer
        outputs = tf.keras.layers.Dense(n_clases, activation=tf.nn.softmax)(hidden)

        super().__init__(inputs=inputs, outputs=outputs)
        opt_rms = tf.keras.optimizers.RMSprop(lr=0.001,decay=1e-6)
        adam = tf.keras.optimizers.Adam()
        self.compile(
            optimizer=opt_rms,
            loss=tf.keras.losses.SparseCategoricalCrossentropy(),
            metrics=[tf.keras.metrics.SparseCategoricalAccuracy(name="accuracy")],
        )

        self.tb_callback=tf.keras.callbacks.TensorBoard(args.logdir, update_freq=1000, profile_batch=1)
        self.tb_callback.on_train_end = lambda *_: None
        self.slack_cb = SlackCallback()

    def train(self, dataset, args):
        datagen = tf.keras.preprocessing.image.ImageDataGenerator(
            featurewise_center=False,
            samplewise_center=False,
            featurewise_std_normalization=False,
            samplewise_std_normalization=False,
            zca_whitening=False,
            rotation_range=15,
            width_shift_range=0.1,
            height_shift_range=0.1,
            horizontal_flip=True,
            vertical_flip=True
            )

        datagen.fit(dataset.train.data["images"])
        self.fit(
        datagen.flow(dataset.train.data["images"], dataset.train.data["labels"], batch_size=args.batch_size),steps_per_epoch=dataset.train.data["images"].shape[0],
        batch_size=args.batch_size, epochs=args.epochs,
        validation_data=(dataset.dev.data["images"], dataset.dev.data["labels"]),
        callbacks=[self.tb_callback,self.slack_cb],
        )

        """
         self.fit(
            dataset.train.data["images"], dataset.train.data["labels"],
            batch_size=args.batch_size, epochs=args.epochs,
            validation_data=(dataset.dev.data["images"], dataset.dev.data["labels"]),
            callbacks=[self.tb_callback,self.slack_cb],
        )
        """

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
    parser.add_argument("--batch_size", default=64, type=int, help="Batch size.")
    parser.add_argument("--epochs", default=60, type=int, help="Number of epochs.")
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
    #args.cnn = "R-[CB-32-3-1-same,CB-128-3-1-same],M-2-2,DR-0.5,R-[CB-64-3-1-same,CB-128-3-3-same],M-2-2,DR-0.5,CB-128-3-1-same,CB-256-3-1-same,M-2-2,DR-0.5,F"
    #args.cnn = "CB-32-3-1-same,CB-32-3-1-same,A-2-2,CB-64-3-1-same,CB-64-3-1-same,A-2-2,CB-128-3-1-same,CB-128-3-1-same,A-2-2,F,D-128,DR-0.3,D-10"
    #args.cnn = "CB-32-4-4-same,R-[CB-64-3-3-same,CB-32-4-4-same],A-2-2,R-[CB-128-8-8-same,CB-32-2-2-same],A-3-3,R-[CB-128-8-8-same,CB-32-4-4-same],M-2-2,F"
    """
    args.cnn = "CB-64-3-2-same,M-2-2,C-64-1-1-same,C-64-3-3-same,C-256-1-1-same,R-[C-64-1-1-same,C-64-3-3-same,C-256-1-1-same],R-[C-64-1-1-same,C-64-3-3-same,C-256-1-1-same],C-128-1-2-valid,C-128-3-3-same,C-512-1-1-same,R-[C-128-1-1-same,C-128-3-3-same,C-512-1-1-same],R-[C-128-1-1-same,C-128-3-3-same,C-512-1-1-same],C-256-1-2-same,C-256-3-3-same,C-1024-1-1-same,R-[C-256-1-1-same,C-256-3-3-same,C-1024-1-1-same],R-[C-256-1-1-same,C-256-3-3-same,C-1024-1-1-same],C-512-1-2-same,C-512-3-3-same,C-2048-1-1-same,R-[C-512-1-1-same,C-512-3-3-same,C-2048-1-1-same],R-[C-512-1-1-same,C-512-3-3-same,C-2048-1-1-same],A-2-2,F"
    
    args.cnn = "CB-64-3-2-same,M-2-2,C-64-1-1-same,C-64-3-3-same,C-512-1-1-same,R-[C-64-1-1-same,C-64-3-3-same,C-512-1-1-same],R-[C-64-1-1-same,C-64-3-3-same,C-512-1-1-same],F,D-1024,DR-0.5"
    """
    #OK
    """
    args.cnn= "CB-64-3-1-same,CB-128-3-1-same,CB-128-3-1-same,CB-128-3-1-same,M-2-2,CB-128-3-1-same,CB-128-3-1-same,CB-128-3-1-same,M-2-1,CB-128-3-1-same,CB-128-3-1-same,M-2-2,CB-128-3-1-same,CB-128-1-1-same,CB-128-3-1-same,M-2-2,CB-128-3-1-same,M-2-2,F"
    """
    """
    args.cnn= "CB-32-3-1-same,CB-64-3-1-same,CB-64-3-1-same,CB-64-3-1-same,M-2-2,CB-64-3-1-same,CB-64-3-1-same,CB-64-3-1-same,M-2-1,CB-64-3-1-same,CB-64-3-1-same,M-2-2,CB-64-3-1-same,CB-64-1-1-same,CB-64-3-1-same,M-2-2,CB-64-3-1-same,M-2-2,F"
    """
    args.cnn= "CB-32-3-1-same,CB-32-3-1-same,M-2-2,DR-0.2,CB-64-3-1-same,CB-64-3-1-same,M-2-2,DR-0.3,CB-128-3-1-same,CB-128-3-1-same,M-2-2,DR-0.4,F"

    # Create the network and train
    network = Network(args,input_shape=[32,32,3],n_clases=10)
    network.train(cifar, args)

    # Generate test set annotations, but in args.logdir to allow parallel execution.
    with open(os.path.join(args.logdir, "cifar_competition_test_adam.txt"), "w", encoding="utf-8") as out_file:
        for probs in network.predict(cifar.test.data["images"], batch_size=args.batch_size):
            print(np.argmax(probs), file=out_file)
