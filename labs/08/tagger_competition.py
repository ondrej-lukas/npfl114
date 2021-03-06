#!/usr/bin/env python3
#bfc95faa-444e-11e9-b0fd-00505601122b
#3da961ed-4364-11e9-b0fd-00505601122b
import numpy as np
import tensorflow as tf

from morpho_analyzer import MorphoAnalyzer
from morpho_dataset import MorphoDataset

class Network:
    def __init__(self, args, num_words, num_tags, num_chars):
        word_ids = tf.keras.layers.Input(shape=(None,), dtype=tf.int32)
        charseq_ids = tf.keras.layers.Input(shape=(None,), dtype=tf.int32)
        charseqs = tf.keras.layers.Input(shape=(None,), dtype=tf.int32)

        embedded_chars = tf.keras.layers.Embedding(input_dim=num_chars, output_dim=args.cle_dim, mask_zero=True)(
            charseqs)
        gru_chars = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(args.cle_dim, return_sequences=False),
                                                  merge_mode="concat")(embedded_chars)

        replace = tf.keras.layers.Lambda(lambda args: tf.gather(*args))([gru_chars, charseq_ids])
        embedded_words = tf.keras.layers.Embedding(input_dim=num_words, output_dim=args.we_dim, mask_zero=True)(
            word_ids)

        embedded_charseqs = tf.keras.layers.Embedding(num_chars, args.cle_dim)(charseqs)
        conv_layer = []
        for width in range(2, args.cnn_max_width + 1):
            conv1d = tf.keras.layers.Conv1D(args.cnn_filters, width, 1, 'valid', activation='relu')(
                embedded_charseqs)
            conv_layer.append(tf.keras.layers.GlobalMaxPool1D()(conv1d))
        concat = tf.keras.layers.Concatenate()(conv_layer)
        charseqs_word_embeddings = tf.keras.layers.Dense(args.we_dim, activation='relu')(concat)
        formatted_charseqs_word_embeddings = tf.keras.layers.Lambda(lambda largs: tf.gather(*largs))([charseqs_word_embeddings, charseq_ids])



        concat = tf.keras.layers.Concatenate()([embedded_words, replace,formatted_charseqs_word_embeddings])
        hidden = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(args.rnn_dim, return_sequences=True),merge_mode="sum")(concat)
        hidden = tf.keras.layers.Dense(500,"tanh")(hidden)

        predictions = tf.keras.layers.Dense(num_tags, activation="softmax")(hidden)

        self.model = tf.keras.Model(inputs=[word_ids, charseq_ids, charseqs], outputs=predictions)

        self._optimizer = tf.optimizers.Adam()
        self._loss = tf.losses.SparseCategoricalCrossentropy()
        self._metrics = {'loss': tf.metrics.Mean(), 'accuracy': tf.metrics.SparseCategoricalAccuracy()}
        self._writer = tf.summary.create_file_writer(args.logdir, flush_millis=10 * 1000)

    def evaluate(self, dataset, args):
        for metric in self._metrics.values():
            metric.reset_states()
        for batch in dataset.batches(args.batch_size):
            self.evaluate_batch(
                [batch[dataset.FORMS].word_ids, batch[dataset.FORMS].charseq_ids, batch[dataset.FORMS].charseqs],
                batch[dataset.TAGS].word_ids)

        metrics = {name: metric.result() for name, metric in self._metrics.items()}
        return metrics

    def evaluate_batch(self, inputs, tags):
        new_tags = self.fix_tags(tags)
        tags_ex = np.expand_dims(new_tags, axis=2)
        mask = tf.not_equal(tags_ex, 0)
        probabilities = self.model(inputs, training=False)
        loss = self._loss(tags_ex, probabilities, mask)
        for name, metric in self._metrics.items():
            if name == "loss":
                metric(loss)
            else:
                metric(tags_ex, probabilities, mask)

    def fix_tags(self, tags):
        max_length = max(map(len, tags))
        arr = None
        for t in tags:
            t = np.append(t, np.zeros((max_length - len(t),1)))
            if arr is None:
                arr = t
            else:
                arr = np.vstack((arr,t))
        return arr.astype('int32')

    def fix_data(self, dataset):
        tag_lens = [max(map(len, dataset.dev.data[2].word_ids)),max(map(len, dataset.train.data[2].word_ids)),
                                       max(map(len, dataset.test.data[2].word_ids))]
        max_len = max(tag_lens)
        tag_dict = {}
        for tag in dataset.dev.data[2].word_ids:
            tag_dict[tuple(tag)] = np.append(tag, np.zeros((max_len - len(tag),1)))
        for tag in dataset.train.data[2].word_ids:
            tag_dict[tuple(tag)] = np.append(tag, np.zeros((max_len - len(tag), 1)))
        for tag in dataset.test.data[2].word_ids:
            tag_dict[tuple(tag)] = np.append(tag, np.zeros((max_len - len(tag),1)))
        self.tag_dict = tag_dict

    def gen_tags(self,tags):
        longer_tags = [self.tag_dict[tuple(tag)] for tag in tags]
        stacked = np.vstack(longer_tags)
        return stacked

    def train_batch(self, inputs, tags):
        # new_tags = self.fix_tags(tags)
        # new_tags = self.gen_tags(tags)
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


    def train(self, train_data, dev_data, args):
        for epoch in range(0,args.epochs):
            # batch_count = 0
            for batch in train_data.batches(args.batch_size):
                self.train_batch([batch[train_data.FORMS].word_ids,
                                  batch[train_data.FORMS].charseq_ids,
                                  batch[train_data.FORMS].charseqs],
                                 batch[train_data.TAGS].word_ids)
                # batch_count += 1
            # Evaluate on dev data
            metrics = network.evaluate(dev_data, args)
            print("Dev accuracy: ", metrics['accuracy'])

    def predict(self, dataset, args):
        # TODO: Predict method should return a list, each element corresponding
        # to one sentence. Each sentence should be a list/np.ndarray
        # containing _indices_ of chosen tags (not the logits/probabilities).
        predictions = self.model([self.fix_tags(dataset.data[dataset.FORMS].word_ids),
                                  self.fix_tags(dataset.data[dataset.FORMS].charseq_ids),
                                  self.fix_tags(dataset.data[dataset.FORMS].charseqs)],training=False)
        edited = tf.argmax(predictions, axis=2)
        return edited


if __name__ == "__main__":
    import argparse
    import datetime
    import os
    import re

    # Parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size", default=50, type=int, help="Batch size.")
    parser.add_argument("--epochs", default=30, type=int, help="Number of epochs.")
    parser.add_argument("--threads", default=0, type=int, help="Maximum number of threads to use.")
    parser.add_argument("--cle_dim", default=32, type=int, help="Character lvl embedding dimension.")
    parser.add_argument("--we_dim", default=64, type=int, help="Word lvl embedding dimension.")
    parser.add_argument("--rnn_dim", default=64, type=int, help="RNN dimension.")
    parser.add_argument("--cnn_filters", default=24, type=int, help="CNN embedding filters per length.")
    parser.add_argument("--cnn_max_width", default=4, type=int, help="Maximum CNN filter width.")
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

    morpho = MorphoDataset("czech_pdt", max_sentences=5000)
    analyses = MorphoAnalyzer("czech_pdt_analyses")

    # Create the network and train
    network = Network(args, num_words=len(morpho.train.data[morpho.train.FORMS].words),
                            num_tags=len(morpho.train.data[morpho.train.TAGS].words),
                            num_chars=len(morpho.train.data[morpho.train.FORMS].alphabet))
    # network.fix_data(morpho)
    network.train(morpho.train, morpho.dev, args)
    #p = network.predict(morpho.test, args)

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
