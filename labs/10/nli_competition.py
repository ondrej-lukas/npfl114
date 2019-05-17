#!/usr/bin/env python3
import numpy as np
import tensorflow as tf

from nli_dataset import NLIDataset

class Network:
    def __init__(self, args):
        # TODO: Define a suitable model.

        self._writer = tf.summary.create_file_writer(args.logdir, flush_millis=10 * 1000)

        #TEST - ARCHITECTURE1
        #inputs
        word_ids = tf.keras.layers.Input(shape=(None,), dtype=tf.int32)
        charseq_ids = tf.keras.layers.Input(shape=(None,), dtype=tf.int32)
        charseqs = tf.keras.layers.Input(shape=(None,), dtype=tf.int32)
        lvl = tf.keras.layers.Input(shape=(1,), dtype=tf.int32)

        #character lvl embedding
        embedded_chars = tf.keras.layers.Embedding(input_dim=args.num_chars, output_dim=args.cle_dim, mask_zero=True)(charseqs)
        gru_chars = tf.keras.layers.Bidirectional(tf.keras.layers.GRU(args.cle_dim,return_sequences=False), merge_mode="concat")(embedded_chars)
        replace = tf.keras.layers.Lambda(lambda args: tf.gather(*args))([gru_chars, charseq_ids])
        #word lvl embedding
        embedded_words = tf.keras.layers.Embedding(input_dim=args.num_words, output_dim=args.we_dim, mask_zero=True)(word_ids)
        #concatanate
        concat = tf.keras.layers.Concatenate()([embedded_words, replace])
        hidden = tf.keras.layers.Bidirectional(getattr(tf.keras.layers,"LSTM")(args.rnn_cell_dim,return_sequences=False), merge_mode="concat")(concat)
        hidden = tf.keras.layers.Dense(1024 , activation='relu')(hidden)
        out = tf.keras.layers.Dense(args.num_languages, activation="softmax")(hidden)
        self.model = tf.keras.Model(inputs=[word_ids, charseq_ids, charseqs,lvl], outputs=out)
        self._optimizer = tf.optimizers.Adam()
        self._loss = tf.losses.SparseCategoricalCrossentropy()
        self._metrics = tf.metrics.SparseCategoricalAccuracy()
    
    def train_epoch(self, dataset, args):
        for batch in dataset.batches(args.batch_size):
            self.train_batch([batch.word_ids, batch.charseq_ids, batch.charseqs, batch.levels],batch.languages)
    
    def train_batch(self, inputs, targets):
        with tf.GradientTape() as tape:
            pred = self.model(inputs, training=True)
            #print(probabilities.shape, targets.shape)
            #print(targets, pred)
            loss = self._loss(targets,pred)
            gradients = tape.gradient(loss, self.model.variables)
            self._optimizer.apply_gradients(zip(gradients, self.model.variables))
        print(loss)

        tf.summary.experimental.set_step(self._optimizer.iterations)
        with self._writer.as_default():
           tf.summary.scalar("nli/loss", loss)


    def predict(self, dataset, args):
        # TODO: Predict method should return a list/np.ndaddar, each element
        # being the predicted language for a sencence.
        batch = datast.batches(args.batch_size)
        ret = self.model([batch.word_ids, batch.charseq_ids, batch.charseqs, batch.levels], training=False)
        print(ret.shape)
        print(ret)

    def parse_sentences(self,dataset):
        dot_key = dataset._vocabulary_maps['chars']['.']
        essays = dataset._word_ids
        for e in essays:
            """
            dots = 0
            for element in e:
                if element == dot_key:
                    dots += 1
            print(essays)
            # print(dot_ixs)
            """
            #print(e)
            print("------------------------")
            for word in e:
                print(dataset._charseq_ids[word])
            break



if __name__ == "__main__":
    import argparse
    import datetime
    import os
    import re

    # Parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size", default=20, type=int, help="Batch size.")
    parser.add_argument("--epochs", default=1, type=int, help="Number of epochs.")
    parser.add_argument("--threads", default=0, type=int, help="Maximum number of threads to use.")
    parser.add_argument("--cle_dim", default=32, type=int, help="CLE embedding dimension.")
    parser.add_argument("--rnn_cell_dim", default=128, type=int, help="RNN cell dimension.")
    parser.add_argument("--we_dim", default=32, type=int, help="Word embedding dimension.")
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
    
    #print(nli.train.vocabulary("languages"))
    #print(nli.train.vocabulary("levels"))
    #print(nli.train.vocabulary("prompts"))
    #print(nli.train.vocabulary("chars"))
    #print(nli.train.vocabulary("words"))

    args.num_levels = len(nli.train.vocabulary("levels"))
    args.num_languages = len(nli.train.vocabulary("languages"))
    args.num_chars = len(nli.train.vocabulary("chars"))
    args.num_words = len(nli.train.vocabulary("words"))
    

    # Create the network and train
    network = Network(args)
    for epoch in range(args.epochs):
        network.train_epoch(nli.train, args)
    
    # Generate test set annotations, but in args.logdir to allow parallel execution.
    out_path = "nli_competition_test.txt"
    if os.path.isdir(args.logdir): out_path = os.path.join(args.logdir, out_path)
    with open(out_path, "w", encoding="utf-8") as out_file:
        languages = network.predict(nli.test, args)
        for language in languages:
            print(nli.test.vocabulary("languages")[language], file=out_file)