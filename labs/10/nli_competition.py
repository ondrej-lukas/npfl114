#!/usr/bin/env python3
import numpy as np
import tensorflow as tf

from nli_dataset import NLIDataset

class Network:
    def __init__(self, args):
        # TODO: Define a suitable model.

        self._writer = tf.summary.create_file_writer(args.logdir, flush_millis=10 * 1000)

        #TEST - ARCHITECTURE 1
        # #inputs
        # word_ids = tf.keras.layers.Input(shape=(None,), dtype=tf.int32)
        # charseq_ids = tf.keras.layers.Input(shape=(None,), dtype=tf.int32)
        # charseqs = tf.keras.layers.Input(shape=(None,), dtype=tf.int32)
        # lvl = tf.keras.layers.Input(shape=(1,), dtype=tf.int32)
        #
        # #character lvl embedding
        # embedded_chars = tf.keras.layers.Embedding(input_dim=args.num_chars, output_dim=args.cle_dim, mask_zero=True)(charseqs)
        # gru_chars = tf.keras.layers.Bidirectional(tf.keras.layers.GRU(args.cle_dim,return_sequences=False), merge_mode="sum")(embedded_chars)
        # replace = tf.keras.layers.Lambda(lambda args: tf.gather(*args))([gru_chars, charseq_ids])
        #
        # #lvl embedding
        # # em_lvl = tf.keras.layers.Embedding(input_dim = args.num_levels, output_dim=256, mask_zero=False)(lvl)
        # # em_lvl = tf.keras.layers.Flatten()(em_lvl)
        #
        # #word lvl embedding
        # embedded_words = tf.keras.layers.Embedding(input_dim=args.num_words, output_dim=args.we_dim, mask_zero=True)(word_ids)
        #
        # #concatanate
        # concat = tf.keras.layers.Concatenate()([embedded_words, replace])
        # hidden = tf.keras.layers.Bidirectional(getattr(tf.keras.layers,"GRU")(args.rnn_cell_dim,return_sequences=False), merge_mode="sum")(concat)
        # # hidden = tf.keras.layers.Dense(256, activation='relu')(hidden)
        # # hidden = tf.keras.layers.Concatenate(axis=-1)([hidden, em_lvl])
        # hidden = tf.keras.layers.Dense(512, activation="relu")(hidden)
        # # hidden = tf.keras.layers.Dropout(rate=0.5)(hidden)
        # out = tf.keras.layers.Dense(args.num_languages, activation="softmax")(hidden)
        #
        # self.model = tf.keras.Model(inputs=[word_ids, charseq_ids, charseqs, lvl], outputs=out)

        # TEST - ARCHITECTURE 2
        # word_ids = tf.keras.layers.Input(shape=(None,), dtype=tf.int32)
        # embedded_words = tf.keras.layers.Embedding(input_dim=args.num_words, output_dim=args.we_dim, mask_zero=True)(
        #     word_ids)
        # hidden = tf.keras.layers.Bidirectional(
        #     getattr(tf.keras.layers, "LSTM")(args.rnn_cell_dim, return_sequences=False), merge_mode="sum")(embedded_words)
        # hidden = tf.keras.layers.Dense(512, activation='relu')(hidden)
        # hidden = tf.keras.layers.Dropout(rate=0.5)(hidden)
        # out = tf.keras.layers.Dense(args.num_languages, activation="softmax")(hidden)
        # self.model = tf.keras.Model(inputs=word_ids, outputs=out)

        # TEST - ARCHITECTURE 3
        # charseq_ids = tf.keras.layers.Input(shape=(None,), dtype=tf.int32)
        # charseqs = tf.keras.layers.Input(shape=(None,), dtype=tf.int32)
        # embedded_chars = tf.keras.layers.Embedding(input_dim=args.num_chars, output_dim=args.cle_dim, mask_zero=True)(
        #     charseqs)
        # gru_chars = tf.keras.layers.Bidirectional(tf.keras.layers.GRU(args.cle_dim,return_sequences=False), merge_mode="sum")(embedded_chars)
        # replace = tf.keras.layers.Lambda(lambda args: tf.gather(*args))([gru_chars, charseq_ids])
        # hidden = tf.keras.layers.Bidirectional(getattr(tf.keras.layers,"LSTM")(args.rnn_cell_dim,return_sequences=False), merge_mode="sum")(replace)
        # hidden = tf.keras.layers.Dense(512,"relu")(hidden)
        # hidden = tf.keras.layers.Dropout(rate=0.5)(hidden)
        # out = tf.keras.layers.Dense(args.num_languages, "softmax")(hidden)
        # self.model = tf.keras.Model(inputs=[charseq_ids,charseqs],outputs=out)

        # TEST - ARCHITECTURE 4
        # word_ids = tf.keras.layers.Input(shape=(None,), dtype=tf.int32)
        # hidden = tf.keras.layers.Embedding(input_dim=args.num_words, output_dim=args.we_dim, mask_zero=True)(word_ids)
        # hidden = tf.keras.layers.Bidirectional(
        #     getattr(tf.keras.layers, "GRU")(args.rnn_cell_dim, return_sequences=False), merge_mode="concat")(hidden)
        # hidden = tf.keras.layers.Dense(500, kernel_initializer="glorot_uniform", activation="sigmoid")(hidden)
        # hidden = tf.keras.layers.Dropout(0.5)(hidden)
        # hidden = tf.keras.layers.Dense(300, kernel_initializer="glorot_uniform", activation="sigmoid")(hidden)
        # hidden = tf.keras.layers.Dropout(0.5)(hidden)
        # hidden = tf.keras.layers.Dense(100, kernel_initializer="glorot_uniform", activation="sigmoid")(hidden)
        # hidden = tf.keras.layers.Dropout(0.5)(hidden)
        # out = tf.keras.layers.Dense(args.num_languages, kernel_initializer="glorot_uniform", activation="softmax")(hidden)
        # self.model = tf.keras.Model(inputs=word_ids, outputs=out)

        # TEST - ARCHITECTURE 5
        charseq_ids = tf.keras.layers.Input(shape=(None,), dtype=tf.int32)
        charseqs = tf.keras.layers.Input(shape=(None,), dtype=tf.int32)
        embedded_chars = tf.keras.layers.Embedding(input_dim=args.num_chars, output_dim=args.cle_dim, mask_zero=True)(charseqs)
        gru_chars = tf.keras.layers.Bidirectional(tf.keras.layers.GRU(args.cle_dim,return_sequences=False), merge_mode="concat")(embedded_chars)
        replace = tf.keras.layers.Lambda(lambda args: tf.gather(*args))([gru_chars, charseq_ids])
        hidden = tf.keras.layers.GRU(args.cle_dim, return_sequences=False)(replace)
        hidden = tf.keras.layers.Dense(500, kernel_initializer="glorot_uniform", activation="sigmoid")(hidden)
        hidden = tf.keras.layers.Dropout(0.5)(hidden)
        hidden = tf.keras.layers.Dense(300, kernel_initializer="glorot_uniform", activation="sigmoid")(hidden)
        hidden = tf.keras.layers.Dropout(0.5)(hidden)
        hidden = tf.keras.layers.Dense(100, kernel_initializer="glorot_uniform", activation="sigmoid")(hidden)
        hidden = tf.keras.layers.Dropout(0.5)(hidden)
        out = tf.keras.layers.Dense(args.num_languages, kernel_initializer="glorot_uniform", activation="softmax")(hidden)
        self.model = tf.keras.Model(inputs=[charseq_ids,charseqs], outputs=out)

        self._optimizer = tf.optimizers.Adam()
        # self._loss = tf.losses.SparseCategoricalCrossentropy(from_logits=True)
        self._loss = tf.keras.losses.CategoricalCrossentropy()
        # self._metrics = tf.metrics.SparseCategoricalAccuracy()
        self._metrics = tf.metrics.CategoricalCrossentropy()
    
    def train_epoch(self, dataset, args, dev_dataset):
        for batch in dataset.batches(args.batch_size):
            # self.train_batch([batch.word_ids, batch.charseq_ids, batch.charseqs, batch.levels],batch.languages)
            # self.train_batch(batch.word_ids,batch.languages)
            self.train_batch([batch.charseq_ids, batch.charseqs], batch.languages)
            # for b in dev_dataset.batches(len(dev_dataset._languages)):
            #     dev_loss = self._loss(b.languages, self.model([b.charseq_ids,
            #                                            b.charseqs],training=False))
            #     print('dev data loss = ', dev_loss)

    def smooth_targets(self,targets):
        return tf.one_hot(targets,depth=args.num_languages,on_value=0.99,off_value=0.01)

    def train_batch(self, inputs, targets):
        t = self.smooth_targets(targets)
        with tf.GradientTape() as tape:
            pred = self.model(inputs, training=True)
            loss = self._loss(t, pred)
            gradients = tape.gradient(loss, self.model.variables)
            self._optimizer.apply_gradients(zip(gradients, self.model.variables))
        print(loss)

        tf.summary.experimental.set_step(self._optimizer.iterations)
        with self._writer.as_default():
           tf.summary.scalar("nli/loss", loss)


    def predict(self, dataset, args):
        # TODO: Predict method should return a list/np.ndaddar, each element
        # being the predicted language for a sencence.
        batch = dataset.batches(args.batch_size)
        ret = self.model([batch.word_ids, batch.charseq_ids, batch.charseqs, batch.levels], training=False)
        print(ret.shape)
        print(ret)


if __name__ == "__main__":
    import argparse
    import datetime
    import os
    import re

    # Parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size", default=50, type=int, help="Batch size.")
    parser.add_argument("--epochs", default=50, type=int, help="Number of epochs.")
    parser.add_argument("--threads", default=0, type=int, help="Maximum number of threads to use.")
    parser.add_argument("--cle_dim", default=128, type=int, help="CLE embedding dimension.")
    parser.add_argument("--rnn_cell_dim", default=128, type=int, help="RNN cell dimension.")
    parser.add_argument("--we_dim", default=512, type=int, help="Word embedding dimension.")
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
    nli = NLIDataset()

    args.num_levels = len(nli.train.vocabulary("levels"))
    args.num_languages = len(nli.train.vocabulary("languages"))
    args.num_chars = len(nli.train.vocabulary("chars"))
    args.num_words = len(nli.train.vocabulary("words"))
    args.lvl_dim = 32
    

    # Create the network and train
    network = Network(args)
    for epoch in range(args.epochs):
        network.train_epoch(nli.train, args, nli.dev)
    
    # Generate test set annotations, but in args.logdir to allow parallel execution.
    out_path = "nli_competition_test.txt"
    if os.path.isdir(args.logdir): out_path = os.path.join(args.logdir, out_path)
    with open(out_path, "w", encoding="utf-8") as out_file:
        languages = network.predict(nli.test, args)
        for language in languages:
            print(nli.test.vocabulary("languages")[language], file=out_file)