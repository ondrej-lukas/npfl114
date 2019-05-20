#!/usr/bin/env python3
import sys

import gym
import numpy as np

class GymEnvironment:
    def __init__(self, env, separators=None):
        self._env = gym.make(env)
        self._env.seed(42)

        self._separators = separators

        self._evaluating_from = None
        self._episode_return = 0
        self._episode_returns = []
        self._episode_ended = True

    def _maybe_discretize(self, observation):
        if self._separators is not None:
            buckets = np.array(observation, dtype=np.int)
            for i in range(len(observation)):
                buckets[i] = np.digitize(observation[i], self._separators[i])
            observation = 0
            for i in range(len(self._separators)):
                observation *= 1 + len(self._separators[i])
                observation += buckets[i]

        return observation

    @property
    def states(self):
        if self._separators is not None:
            states = 1
            for separator in self._separators:
                states *= 1 + len(separator)
            return states
        raise RuntimeError("Continuous environments have infinitely many states")

    @property
    def state_shape(self):
        if self._separators is not None:
            return []
        else:
            return list(self._env.observation_space.shape)

    @property
    def actions(self):
        return self._env.action_space.n

    @property
    def episode(self):
        return len(self._episode_returns)

    def reset(self, start_evaluate=False):
        if start_evaluate and self._evaluating_from is None:
            self._evaluating_from = self.episode

        self._episode_ended = False
        return self._maybe_discretize(self._env.reset())

    def step(self, action):
        if self._episode_ended:
            raise RuntimeError("Cannot run `step` on environments without an active episode, run `reset` first")

        observation, reward, done, info = self._env.step(action)

        self._episode_return += reward
        if done:
            self._episode_ended = True
            self._episode_returns.append(self._episode_return)

            if self.episode % 10 == 0:
                print("Episode {}, mean 100-episode return {}".format(
                    self.episode, np.mean(self._episode_returns[-100:])), file=sys.stderr)
            if self._evaluating_from is not None and self.episode >= self._evaluating_from + 100:
                print("The mean 100-episode return after evaluation {}".format(np.mean(self._episode_returns[-100:])))
                sys.exit(0)

            self._episode_return = 0

        return self._maybe_discretize(observation), reward, done, info

    def render(self):
        self._env.render()
