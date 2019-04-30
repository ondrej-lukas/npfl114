#!/usr/bin/env python3
#bfc95faa-444e-11e9-b0fd-00505601122b
#3da961ed-4364-11e9-b0fd-00505601122b
import numpy as np
import tensorflow as tf

from morpho_analyzer import MorphoAnalyzer
from morpho_dataset import MorphoDataset

class Network:
    def __init__(self, args, num_words, num_tags, num_chars):
        # TODO: Define a suitable model.
        word_ids = tf.keras.layers.Input(shape=(None,), dtype=tf.int32)
        charseq_ids = tf.keras.layers.Input(shape=(None,), dtype=tf.int32)
        charseqs = tf.keras.layers.Input(shape=(None,), dtype=tf.int32)

        embedded_chars = tf.keras.layers.Embedding(input_dim=num_chars, output_dim=args.cle_dim, mask_zero=True)(
            charseqs)
        gru_chars = tf.keras.layers.Bidirectional(tf.keras.layers.GRU(args.cle_dim, return_sequences=False),
                                                  merge_mode="concat")(embedded_chars)
        replace = tf.keras.layers.Lambda(lambda args: tf.gather(*args))([gru_chars, charseq_ids])
        embedded_words = tf.keras.layers.Embedding(input_dim=num_words, output_dim=args.we_dim, mask_zero=True)(
            word_ids)
        concat = tf.keras.layers.Concatenate()([embedded_words, replace])
        hidden = tf.keras.layers.Bidirectional(tf.keras.layers.GRU(args.rnn_dim, return_sequences=True),
                                                  merge_mode="concat")(concat)
        hidden = tf.keras.layers.Dense(80,"relu")(hidden)
        predictions = tf.keras.layers.Dense(num_tags, activation="softmax")(hidden)

        self.model = tf.keras.Model(inputs=[word_ids, charseq_ids, charseqs], outputs=predictions)

        opt = tf.optimizers.Adam()
        loss = tf.losses.SparseCategoricalCrossentropy()
        metrics = {'loss': tf.metrics.Mean(), 'accuracy': tf.metrics.SparseCategoricalAccuracy()}
        self.model.compile(opt,loss,metrics)
        self._writer = tf.summary.create_file_writer(args.logdir, flush_millis=10 * 1000)

    def train_batch(self, inputs, tags):
        tags_ex = np.expand_dims(tags, axis=2)
        mask = tf.not_equal(tags_ex, 0)

        with tf.GradientTape() as tape:
            probabilities = self.model(inputs, training=True)
            loss = self._loss(tags_ex, probabilities, mask)
        gradients = tape.gradient(loss, self.model.variables)
        self._optimizer.apply_gradients(zip(gradients, self.model.variables))

        tf.summary.experimental.set_step(self._optimizer.iterations)
        with self._writer.as_default():
            for name, metric in self._metrics.items():
                metric.reset_states()
                if name == "loss":
                    metric(loss)
                else:
                    metric(tags_ex, probabilities, mask)
                # tf.summary.scalar("train/{}".format(name), metric.result())

    def train(self, pdt, args):
        self.model.fit(x=[pdt.data[pdt.FORMS].word_ids,
                        pdt.data[pdt.FORMS].charseq_ids,
                        pdt.data[pdt.FORMS].charseqs],
                       y=np.expand_dims(pdt.data[pdt.TAGS].word_ids, axis=2), batch_size=args.batch_size, epochs=args.epochs)


    def predict(self, dataset, args):
        # TODO: Predict method should return a list, each element corresponding
        # to one sentence. Each sentence should be a list/np.ndarray
        # containing _indices_ of chosen tags (not the logits/probabilities).
        pass


if __name__ == "__main__":
    import argparse
    import datetime
    import os
    import re

    # Parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size", default=50, type=int, help="Batch size.")
    parser.add_argument("--epochs", default=10, type=int, help="Number of epochs.")
    parser.add_argument("--threads", default=1, type=int, help="Maximum number of threads to use.")
    parser.add_argument("--cle_dim", default=16, type=int, help="Character lvl embedding dimension.")
    parser.add_argument("--we_dim", default=64, type=int, help="Word lvl embedding dimension.")
    parser.add_argument("--rnn_dim", default=32, type=int, help="RNN dimension.")
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

    # Load the data. Using analyses is only optional.
    morpho = MorphoDataset("czech_pdt")
    analyses = MorphoAnalyzer("czech_pdt_analyses")

    # Create the network and train
    network = Network(args, num_words=len(morpho.train.data[morpho.train.FORMS].words),
                            num_tags=len(morpho.train.data[morpho.train.TAGS].words),
                            num_chars=len(morpho.train.data[morpho.train.FORMS].alphabet))
    network.train(morpho.train, args)

    # Generate test set annotations, but in args.logdir to allow parallel execution.
    out_path = "tagger_competition_test.txt"
    if os.path.isdir(args.logdir): out_path = os.path.join(args.logdir, out_path)
    with open(out_path, "w", encoding="utf-8") as out_file:
        for i, sentence in enumerate(network.predict(morpho.test, args)):
            for j in range(len(morpho.test.data[morpho.test.FORMS].word_strings[i])):
                print(morpho.test.data[morpho.test.FORMS].word_strings[i][j],
                      morpho.test.data[morpho.test.LEMMAS].word_strings[i][j],
                      morpho.test.data[morpho.test.TAGS].words[sentence[j]],
                      sep="\t", file=out_file)
            print(file=out_file)
