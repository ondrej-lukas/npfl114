#!/usr/bin/env python3
#bfc95faa-444e-11e9-b0fd-00505601122b
#3da961ed-4364-11e9-b0fd-00505601122b
import numpy as np
import tensorflow as tf

from fashion_masks_data import FashionMasks

# TODO: Define a suitable model in the Network class.
# A suitable starting model contains some number of shared
# convolutional layers, followed by two heads, one predicting
# the label and the other one the masks.
class Network:
    def __init__(self, args,fashion_masks):
        input_shape = [fashion_masks.H,fashion_masks.W,fashion_masks.C]
        inputs = tf.keras.layers.Input(shape=input_shape)
        c1 = tf.keras.layers.Conv2D(32, (3, 3), padding="same",activation="relu")(inputs)
        bn1 = tf.keras.layers.BatchNormalization(axis=-1)(c1)
        c2 = tf.keras.layers.Conv2D(32, (3, 3), padding="same",activation="relu")(bn1)
        bn2 = tf.keras.layers.BatchNormalization(axis=-1)(c2)
        mp1 = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))(bn2)
        dr1 = tf.keras.layers.Dropout(0.25)(mp1)

        cn3 = tf.keras.layers.Conv2D(64, (3, 3), padding="same", activation="relu")(dr1)
        bn3 = tf.keras.layers.BatchNormalization(axis=-1)(cn3)
        cn4 = tf.keras.layers.Conv2D(64, (3, 3), padding="same", activation="relu")(bn3)
        bn4 = tf.keras.layers.BatchNormalization(axis=-1)(cn4)
        mp2 = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))(bn4)
        dr2 = tf.keras.layers.Dropout(0.25)(mp2)

        # Mask making branch
        cn5 = tf.keras.layers.Conv2D(64, (3, 3), padding="same", activation="relu")(dr2)
        bn5 = tf.keras.layers.BatchNormalization(axis=-1)(cn5)
        tc1 = tf.keras.layers.Conv2DTranspose(64, 1, padding="same",strides=(4,4),activation="relu")(bn5)
        dr3 = tf.keras.layers.Dropout(0.3)(tc1)
        cn6 = tf.keras.layers.Conv2D(1, (1,1), padding="same",activation="sigmoid")(dr3)
        masks = tf.keras.layers.Flatten()(cn6)
        # masks = tf.keras.layers.Dense(1000,activation="relu")(masks)
        # masks = tf.keras.layers.Dense(28*28,activation="sigmoid")(masks)

        # Classification branch
        flat = tf.keras.layers.Flatten()(dr2)
        dense = tf.keras.layers.Dense(512, activation="relu")(flat)
        bn5 = tf.keras.layers.BatchNormalization(axis=-1)(dense)
        dr3 = tf.keras.layers.Dropout(0.3)(bn5)
        final = tf.keras.layers.Dense(10,activation="softmax")(dr3)
        # final = tf.keras.layers.Dense(1,activation="relu")(final)

        optimizer = tf.keras.optimizers.Adam()
        loss = [tf.keras.losses.SparseCategoricalCrossentropy(), tf.keras.losses.BinaryCrossentropy()]
        metrics = ['accuracy']
        self.model = tf.keras.Model(inputs=inputs,outputs=[final,masks])
        self.model.compile(optimizer=optimizer,
                           loss=loss,metrics=metrics)

        # for layer in self.model.layers:
        #     print(layer.output_shape)

    @staticmethod
    def _prepare_batches(batches_generator):
        new_batches = []
        for batch in batches_generator:
            model_inputs = batch["images"]
            #print([np.asarray(x).reshape(-1) for x in batch["masks"]])
            model_targets = [batch["labels"], np.array([np.asarray(x).reshape(-1) for x in batch["masks"]])]
            # TODO: yield the suitable modified inputs and targets using batches[0:2]
            yield (model_inputs, model_targets)

    def train(self, fashion_masks, args):
        for epoch in range(args.epochs):
            # TODO: Train for one epoch using `model.train_on_batch` for each batch.
            for x,y in self._prepare_batches(fashion_masks.train.batches(args.batch_size)):
                print("Training on batch")
                self.model.train_on_batch(x,y)
            # Print development evaluation
            print("Dev {}: label: {}, mask: {}, both: {}".format(epoch + 1, *self.evaluate(fashion_masks.dev, args)))
    
    def evaluate(self, dataset, args):
        # TODO: Evaluate the given dataset, returning two accuracies, the first being
        # the direct prediction of the model, and the second computed by comparing predicted
        # labels of the images.
        labels = []
        masks = []
        for inputs, targets in self._prepare_batches(dataset.batches(args.batch_size)):
            classification, mask = self.model.predict_on_batch(inputs)
            #print(mask.shape, targets[1].shape)
            label = np.argmax(classification, axis=1)
            mask =  np.array(np.round(mask), dtype=bool)
            labels.append(np.sum(label==targets[0])/len(label))
            #print(targets[1].shape)
            masks.append(np.sum(mask == targets[1],axis=1)/(28*28))
            #print(np.sum(mask == targets[1]))
            #print(masks[-1])
            print("---------------------------")
        return np.mean(labels), np.mean(masks), np.mean(labels)*np.mean(masks)

    """
    def train(self, fashion_masks, args):
        self.model.fit(fashion_masks.train.data['images'],fashion_masks.train.data['labels'], batch_size=args.batch_size, epochs=args.epochs)
    """
if __name__ == "__main__":
    import argparse
    import datetime
    import os
    import re

    # Parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size", default=10, type=int, help="Batch size.")
    parser.add_argument("--epochs", default=1, type=int, help="Number of epochs.")
    parser.add_argument("--threads", default=0, type=int, help="Maximum number of threads to use.")
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
    fashion_masks = FashionMasks()

    # Create the network and train
    network = Network(args,fashion_masks)
    network.train(fashion_masks, args)

    # Predict test data in args.logdir
    with open("fashion_masks_test.txt", "w", encoding="utf-8") as out_file:
        # TODO: Predict labels and masks on fashion_masks.test.data["images"],
        # into test_labels and test_masks (test_masks is assumed to be
        # a Numpy array with values 0/1).
        test_labels,test_masks = network.model.predict(fashion_masks.test.data['images'])
        """
        test_masks = []
        for i in range(0,test_labels.shape[0]):
            max_class = max(test_labels[i,:])
            mask = np.zeros((28,28,1))
            mask[fashion_masks.test.data['images'][i] > 0.5] = 1
        """
        for label, mask in zip(test_labels, test_masks):
            print(label, *mask.astype(np.uint8).flatten(), file=out_file)
