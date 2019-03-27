#!/usr/bin/env python3
#bfc95faa-444e-11e9-b0fd-00505601122b
#3da961ed-4364-11e9-b0fd-00505601122b

#TODO

import argparse
import datetime
import os
import re

import numpy as np
import tensorflow as tf

from uppercase_data import UppercaseData

# Parse arguments
parser = argparse.ArgumentParser()
parser.add_argument("--alphabet_size", default=50, type=int, help="If nonzero, limit alphabet to this many most frequent chars.")
parser.add_argument("--batch_size", default=1000, type=int, help="Batch size.")
parser.add_argument("--epochs", default=25, type=int, help="Number of epochs.")
parser.add_argument("--hidden_layers", default="500", type=str, help="Hidden layer configuration.")
parser.add_argument("--threads", default=0, type=int, help="Maximum number of threads to use.")
parser.add_argument("--window", default=10, type=int, help="Window size to use.")
args = parser.parse_args()
args.hidden_layers = [int(hidden_layer) for hidden_layer in args.hidden_layers.split(",") if hidden_layer]

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
uppercase_data = UppercaseData(args.window, args.alphabet_size)
print("Data loaded")

# TODO: Implement a suitable model, optionally including regularization, select
# 40od hyperparameters and train the model.
#
# The inputs are _windows_ of fixed size (`args.window` characters on left,
# the character in question, and `args.window` characters on right), where
# each character is representedy by a `tf.int32` index. To suitably represent
# the characters, you can:
# - Convert the character indices into _one-hot encoding_. There is no
#   explicit Keras layer, so you can
#   - use a Lambda layer which can encompass any function:
#       Sequential([
#         tf.layers.InputLayer(input_shape=[2 * args.window + 1], dtype=tf.int32),
#         tf.layers.Lambda(lambda x: tf.one_hot(x, len(uppercase_data.train.alphabet))),
#   - or use Functional API and a code looking like
#       inputs = tf.keras.layers.Input(shape=[2 * args.window + 1], dtype=tf.int32)
#       encoded = tf.one_hot(inputs, len(uppercase_data.train.alphabet))
#   You can then flatten the one-hot encoded windows and follow with a dense layer.
# - Alternatively, you can use `tf.keras.layers.Embedding`, which is an efficient
#   implementation of one-hot encoding followed by a Dense layer, and flatten afterwards.

model = tf.keras.Sequential([
    tf.keras.layers.InputLayer(input_shape=[2 * args.window + 1], dtype=tf.int32),
    tf.keras.layers.Lambda(lambda x: tf.one_hot(x, len(uppercase_data.train.alphabet))),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(units=40,activation=tf.nn.relu),
    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.Dense(units=500,activation=tf.nn.relu),
    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.Dense(units=100,activation=tf.nn.relu),
    tf.keras.layers.Dropout(0.3),
    tf.keras.layers.Dense(units=2,activation=tf.nn.softmax),
])

model.compile(
    optimizer=tf.keras.optimizers.Adam(),
    loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    metrics=[tf.keras.metrics.SparseCategoricalAccuracy(name="accuracy")],
)

model.fit(
    uppercase_data.train.data["windows"],uppercase_data.train.data["labels"],
    batch_size=args.batch_size,
    epochs=args.epochs,
    validation_data=(uppercase_data.dev.data["windows"], uppercase_data.dev.data["labels"]),
)

print("MODEL FITTED")

test_logs = model.evaluate(uppercase_data.test.data["windows"],
                           uppercase_data.test.data["labels"],
                           batch_size=args.batch_size,
                           )
accuracy = test_logs[1]
print("Accuracy on test data: ", accuracy)

model.save("uppercase_model.h5", include_optimizer=False)
with open("uppercase_test.txt", "w", encoding="utf-8") as out_file:
    # TODO: Generate correctly capitalized test set.
    # Use `uppercase_data.test.text` as input, capitalize suitable characters,
    # and write the result to `uppercase_test.txt` file.
    output = model.predict(uppercase_data.test.data["windows"])
    for i in range(0,len(output)):
        if output[i][1] > output[i][0]:
            out_file.write(uppercase_data.test.text[i].upper())
        else:
            out_file.write(uppercase_data.test.text[i])