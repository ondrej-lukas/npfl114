#!/usr/bin/env python3
#bfc95faa-444e-11e9-b0fd-00505601122b
#3da961ed-4364-11e9-b0fd-00505601122b
import argparse
import datetime
import os
import re

import numpy as np
import tensorflow as tf

# Parse arguments
# TODO: Set reasonable defaults and possibly add more arguments.
parser = argparse.ArgumentParser()
parser.add_argument("--batch_size", default=10, type=int, help="Batch size.")
parser.add_argument("--epochs", default=550, type=int, help="Number of epochs.")
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
# Load the data
observations, labels = [], []
with open("gym_cartpole-data.txt", "r") as data:
    for line in data:
        columns = line.rstrip("\n").split()
        observations.append([float(column) for column in columns[0:-1]])
        labels.append(float(columns[-1]))
observations, labels = np.array(observations), np.array(labels)

# TODO: Create the model in the `model` variable.
# # However, beware that there is currently a bug in Keras which does
# # not correctly serialize InputLayer. Instead of using an InputLayer,
# # pass explicitly `input_shape` to the first real model layer.

# model = tf.keras.Sequential([
#     tf.keras.layers.Dense(4, activation=tf.nn.relu, input_shape=(4,)),
#     tf.keras.layers.Dense(20, activation=tf.nn.relu),
#     tf.keras.layers.Dense(80, activation=tf.nn.relu),
#     tf.keras.layers.Dense(40, activation=tf.nn.relu),
#     tf.keras.layers.Dense(2, activation=tf.nn.softmax)]
# )

# THE BETTER
# model = tf.keras.Sequential([
#     tf.keras.layers.Dense(32, activation=tf.nn.sigmoid, input_shape=(4,)),
#     tf.keras.layers.Dense(128, activation=tf.nn.relu),
#     tf.keras.layers.Dense(2, activation=tf.nn.softmax)]
# )

# THE BESTEST
model = tf.keras.Sequential([
    tf.keras.layers.Dense(24, activation=tf.nn.tanh, input_shape=(4,)),
    tf.keras.layers.Dense(2, activation=tf.nn.softmax)]
)

model.compile(
    optimizer=tf.keras.optimizers.Adam(),
    loss=tf.keras.losses.SparseCategoricalCrossentropy(),
    metrics=[tf.keras.metrics.SparseCategoricalAccuracy()],
)

tb_callback=tf.keras.callbacks.TensorBoard(args.logdir)
model.fit(observations, labels, batch_size=args.batch_size, epochs=args.epochs, callbacks=[tb_callback])

model.save("gym_cartpole_model.h5", include_optimizer=False)
