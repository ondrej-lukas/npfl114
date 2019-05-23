#!/usr/bin/env python3
#bfc95faa-444e-11e9-b0fd-00505601122b
#3da961ed-4364-11e9-b0fd-00505601122b
import numpy as np
import tensorflow as tf

import cart_pole_evaluator

class Network:
    def __init__(self, env, args):
        # TODO: Define suitable model. Apart from the model defined in `reinforce`,
        # define also another model `baseline`, which processes the input using
        # a fully connected hidden layer with non-linear activation, and produces
        # one output (using a dense layer without activation).
        #
        # Use Adam optimizer with given `args.learning_rate` for both models.
        self.model = tf.keras.Sequential()
        self.model.add(tf.keras.layers.InputLayer(input_shape=env.state_shape))
        self.model.add(tf.keras.layers.Dense(args.hidden_layer, "relu"))
        self.model.add(tf.keras.layers.Dense(env.actions, "softmax"))
        self.model.compile(optimizer=tf.optimizers.Adam(args.learning_rate),
                           loss=tf.keras.losses.SparseCategoricalCrossentropy(),
                           metrics=[tf.keras.metrics.SparseCategoricalAccuracy()])

        self.baseline = tf.keras.Sequential()
        self.baseline.add(tf.keras.layers.Dense(args.hidden_layer,"relu"))
        self.baseline.add(tf.keras.layers.Dense(1))
        self.baseline.compile(optimizer=tf.optimizers.Adam(args.learning_rate),
                           loss=tf.keras.losses.MSE())

    def train(self, states, actions, returns):
        states, actions, returns = np.array(states), np.array(actions), np.array(returns)

        # TODO: Train the model using the states, actions and observed returns.
        # You should:
        # - compute the predicted baseline using the `baseline` model
        # - train the policy model, using `returns - predicted_baseline` as weights
        #   in the sparse crossentropy loss
        # - train the `baseline` model to predict `returns`
        baseline = self.baseline.predict_on_batch(states)
        weights = returns - baseline
        self.model.train_on_batch(states, actions, sample_weight=weights)
        self.baseline.train_on_batch(states, returns)

    def predict(self, states):
        states = np.array(states)

        # TODO: Predict distribution over actions for the given input states. Return
        # only the probabilities, not the baseline.
        predictions = self.model.predict_on_batch(states)
        return predictions


if __name__ == "__main__":
    # Parse arguments
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size", default=10, type=int, help="Number of episodes to train on.")
    parser.add_argument("--episodes", default=500, type=int, help="Training episodes.")
    parser.add_argument("--hidden_layer", default=16, type=int, help="Size of hidden layer.")
    parser.add_argument("--learning_rate", default=0.02, type=float, help="Learning rate.")
    parser.add_argument("--render_each", default=0, type=int, help="Render some episodes.")
    parser.add_argument("--threads", default=0, type=int, help="Maximum number of threads to use.")
    args = parser.parse_args()

    # Fix random seed
    np.random.seed(42)
    tf.random.set_seed(42)
    tf.config.threading.set_inter_op_parallelism_threads(args.threads)
    tf.config.threading.set_intra_op_parallelism_threads(args.threads)

    # Create the environment
    env = cart_pole_evaluator.environment(discrete=False)

    # Construct the network
    network = Network(env, args)

    # Training
    for _ in range(args.episodes // args.batch_size):
        batch_states, batch_actions, batch_returns = [], [], []
        for _ in range(args.batch_size):
            # Perform episode
            states, actions, rewards = [], [], []
            state, done = env.reset(), False
            while not done:
                if args.render_each and env.episode > 0 and env.episode % args.render_each == 0:
                    env.render()

                probabilities = network.predict([state])[0]
                # TODO(reinforce): Compute `action` according to the distribution returned by the network.
                # The `np.random.choice` method comes handy.
                action = np.random.choice(env.actions, 1, p=probabilities)[0]
                next_state, reward, done, _ = env.step(action)

                states.append(state)
                actions.append(action)
                rewards.append(reward)

                state = next_state

            # TODO(reinforce): Compute `returns` from the observed `rewards`.
            returns = []
            last_ret = 0
            for i in range(len(rewards) - 1, -1, -1):
                new_ret = rewards[i] + last_ret
                returns.append(new_ret)
                last_ret = new_ret
            returns = list(reversed(returns))

            batch_states += states
            batch_actions += actions
            batch_returns += returns

        network.train(batch_states, batch_actions, batch_returns)

    # Final evaluation
    while True:
        state, done = env.reset(True), False
        while not done:
            probabilities = network.predict([state])[0]
            action = np.argmax(probabilities)
            state, reward, done, _ = env.step(action)
