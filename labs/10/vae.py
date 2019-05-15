#!/usr/bin/env python3
#bfc95faa-444e-11e9-b0fd-00505601122b
#3da961ed-4364-11e9-b0fd-00505601122b
import numpy as np
import tensorflow as tf

from mnist import MNIST

# The neural network model
class Network:
    def __init__(self, args):
        self._z_dim = args.z_dim

        # TODO: Define `self.encoder` as a Model, which
        # - takes input images with shape [MNIST.H, MNIST.W, MNIST.C]
        # - flattens them
        # - applies len(args.encoder_layers) dense layers with ReLU activation,
        #   i-th layer with args.encoder_layers[i] units
        # - generate two outputs z_mean and z_log_variance, each passing the result
        #   of the above line through its own dense layer with args.z_dim units/  
        inputs = tf.keras.layers.Input(shape=[MNIST.H,MNIST.W,MNIST.C])
        hidden = tf.keras.layers.Flatten()(inputs)
        for i in range(0,len(args.encoder_layers)):
            hidden = tf.keras.layers.Dense(args.encoder_layers[i], activation="relu")(hidden)
        z_mean = tf.keras.layers.Dense(args.z_dim)(hidden)
        z_log_variance = tf.keras.layers.Dense(args.z_dim)(hidden)
        self.encoder = tf.keras.Model(inputs=inputs, outputs=[z_mean,z_log_variance])

        # TODO: Define `self.decoder` as a Model, which
        # - takes vectors of [args.z_dim] shape on input
        # - applies len(args.decoder_layers) dense layers with ReLU activation,
        #   i-th layer with args.decoder_layers[i] units
        # - applies output dense layer with MNIST.H * MNIST.W * MNIST.C units
        #   and a suitable output activation
        # - reshapes the output (tf.keras.layers.Reshape) to [MNIST.H, MNIST.W, MNISt.C]
        inputs2 = tf.keras.layers.Input(shape=[args.z_dim])
        for i in range(0,len(args.decoder_layers)):
            if i == 0:
                hidden = tf.keras.layers.Dense(args.decoder_layers[i], activation="relu")(inputs2)
            else:
                hidden = tf.keras.layers.Dense(args.decoder_layers[i], activation="relu")(hidden)
        hidden = tf.keras.layers.Dense(MNIST.H * MNIST.W * MNIST.C,activation="sigmoid")(hidden) # activation?
        output = tf.keras.layers.Reshape((MNIST.H, MNIST.W, MNIST.C))(hidden)
        self.decoder = tf.keras.Model(inputs=inputs2, outputs=output)

        self._optimizer = tf.optimizers.Adam()
        self._reconstruction_loss_fn = tf.losses.BinaryCrossentropy()
        self._writer = tf.summary.create_file_writer(args.logdir, flush_millis=10 * 1000)

    def _kl_divergence(self, a_mean, a_sd, b_mean, b_sd):
        """Method for computing KL divergence of two normal distributions."""
        a_sd_squared, b_sd_squared = a_sd ** 2, b_sd ** 2
        ratio = a_sd_squared / b_sd_squared
        return (a_mean - b_mean) ** 2 / (2 * b_sd_squared) + (ratio - tf.math.log(ratio) - 1) / 2

    @tf.function
    def train_batch(self, images):
        with tf.GradientTape() as tape:
            # TODO: Compute z_mean and z_log_variance of given images using `self.encoder`; do not forget about `training=True`.
            z_mean, z_log_variance = self.encoder(images,training=True)
            # TODO: Sample `z` from a Normal distribution with mean `z_mean` and variance `exp(z_log_variance)`.

            # tfd = tfp.distributions
            z = tf.random.normal(shape=[self._z_dim], mean=z_mean, stddev=tf.math.exp(z_log_variance/2),seed=42)
            # TODO: Decode images using `z`. # HOW?
            Xgen = self.decoder(z)
            # TODO: Define `reconstruction_loss` using self._reconstruction_loss_fn
            #binary cross entropy = Xinput*log(Xgen) + (1-Xinput)log(1-Xgen) = reconstruction loss
            reconstruction_loss = self._reconstruction_loss_fn(images, Xgen)
            #print("reconstruction loss:", reconstruction_loss)
            # TODO: Define `latent_loss` as a mean of KL divergences of suitable distributions.
            #latent loss = KL divergence of generated distribution  and p(Z) = N(0,1)
            latent_loss = tf.reduce_mean(self._kl_divergence(z_mean, tf.math.exp(z_log_variance/2), 0,1))
            #print("latent:", latent_loss)
            # TODO: Define `loss` as a weighted sum of the reconstruction_loss (weighted by the number
            #LOSS = W*H*C*bce + |z|* latent loss
            loss = reconstruction_loss*MNIST.H*MNIST.W*MNIST.C + self._z_dim*latent_loss

            # of pixels in one image) and the latent_loss (weighted by self._z_dim).
            variables = self.encoder.trainable_variables + self.decoder.trainable_variables
            gradients = tape.gradient(loss,variables)
            #print("gradients ok")
            #self._optimizer.apply_gradients(zip(gradients_decoder, self.decoder.variables))
            self._optimizer.apply_gradients(zip(gradients, variables))
            #print("optimizer ok")
            # TODO: Compute gradients with respect to trainable variables of the encoder and the decoder.
            # TODO: Apply the gradients to encoder and decoder trainable variables.x`
            tf.summary.experimental.set_step(self._optimizer.iterations)
            
            
            with self._writer.as_default():
                tf.summary.scalar("vae/reconstruction_loss", reconstruction_loss)
                tf.summary.scalar("vae/latent_loss", latent_loss)
                tf.summary.scalar("vae/loss", loss)
                #print("writing ok")
            return loss

    def generate(self):
        GRID = 20

        def sample_z(batch_size):
            return tf.random.normal(shape=[batch_size, self._z_dim])

        # Generate GRIDxGRID images
        random_images = self.decoder(sample_z(GRID * GRID))

        # Generate GRIDxGRID interpolated images
        if self._z_dim == 2:
            # Use 2D grid for sampled Z
            starts = tf.stack([-2 * tf.ones(GRID), tf.linspace(-2., 2., GRID)], -1)
            ends = tf.stack([2 * tf.ones(GRID), tf.linspace(-2., 2., GRID)], -1)
        else:
            # Generate random Z
            starts, ends = sample_z(GRID), sample_z(GRID)
        interpolated_z = tf.concat(
            [starts[i] + (ends[i] - starts[i]) * tf.expand_dims(tf.linspace(0., 1., GRID), -1) for i in range(GRID)], axis=0)
        interpolated_images = self.decoder(interpolated_z)

        # Stack the random images, then an empty row, and finally interpolated imates
        image = tf.concat(
            [tf.concat(list(images), axis=1) for images in tf.split(random_images, GRID)] +
            [tf.zeros([MNIST.H, MNIST.W * GRID, MNIST.C])] +
            [tf.concat(list(images), axis=1) for images in tf.split(interpolated_images, GRID)], axis=0)
        with self._writer.as_default():
            tf.summary.image("vae/images", tf.expand_dims(image, 0))

    def train_epoch(self, dataset, args):
        loss = 0
        for batch in dataset.batches(args.batch_size):
            loss += self.train_batch(batch["images"])
        self.generate()
        return loss


if __name__ == "__main__":
    import argparse
    import datetime
    import os
    import re

    # Parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size", default=50, type=int, help="Batch size.")
    parser.add_argument("--dataset", default="mnist", type=str, help="MNIST-like dataset to use.")
    parser.add_argument("--decoder_layers", default="500,500,500", type=str, help="Decoder layers.")
    parser.add_argument("--encoder_layers", default="500,500,500", type=str, help="Encoder layers.")
    parser.add_argument("--epochs", default=100, type=int, help="Number of epochs.")
    parser.add_argument("--recodex", default=False, action="store_true", help="Evaluation in ReCodEx.")
    parser.add_argument("--threads", default=0, type=int, help="Maximum number of threads to use.")
    parser.add_argument("--z_dim", default=100, type=int, help="Dimension of Z.")
    args = parser.parse_args()
    args.decoder_layers = [int(decoder_layer) for decoder_layer in args.decoder_layers.split(",")]
    args.encoder_layers = [int(encoder_layer) for encoder_layer in args.encoder_layers.split(",")]

    # Fix random seeds
    np.random.seed(42)
    tf.random.set_seed(42)
    if args.recodex:
        tf.keras.utils.get_custom_objects()["glorot_uniform"] = lambda: tf.initializers.glorot_uniform(seed=42)
        tf.keras.utils.get_custom_objects()["orthogonal"] = lambda: tf.initializers.orthogonal(seed=42)
        tf.keras.utils.get_custom_objects()["uniform"] = lambda: tf.initializers.RandomUniform(seed=42)
    tf.config.threading.set_inter_op_parallelism_threads(args.threads)
    tf.config.threading.set_intra_op_parallelism_threads(args.threads)

    # Create logdir name
    args.logdir = os.path.join("logs", "{}-{}-{}".format(
        os.path.basename(__file__),
        datetime.datetime.now().strftime("%Y-%m-%d_%H%M%S"),
        ",".join(("{}={}".format(re.sub("(.)[^_]*_?", r"\1", key), value) for key, value in sorted(vars(args).items())))
    ))

    # Load data
    mnist = MNIST(args.dataset)

    # Create the network and train
    network = Network(args)
    for epoch in range(args.epochs):
        loss = network.train_epoch(mnist.train, args)
    with open("vae.out", "w") as out_file:
        print("{:.2f}".format(loss), file=out_file)
