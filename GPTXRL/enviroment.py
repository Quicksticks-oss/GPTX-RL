import numpy as np
import random
import math

class TextEnv:
    def __init__(self, input_size):
        self.n_steps = 0
        self.input_size = input_size
        self.current_tokens = []
        self.last_actions = []
        self.n_games = 0

    def set_text(self, tokens):
        self.current_tokens = tokens
        while len(self.current_tokens) < self.input_size:
            self.current_tokens = np.append(self.current_tokens, 0)

    def get_observation(self):
        return np.array([_ for _ in self.current_tokens], dtype=np.float32)

    def reset(self, tokens):
        self.set_text(tokens)
        self.n_steps = 0
        self.n_games= 0
        return self.get_observation()

    def step(self, action, actual_next=0, repitition_penalty=5, inference=False):
        done = False
        won = False
        reward = -1

        if action == actual_next:
            reward = 100
            done = True
            won = True

        if self.n_games >= 32 and inference == False:
            done = True
            reward = -100
        self.n_games += 1
        return self.get_observation(), int(reward), done, won, self.n_games

    def close(self):
        pass
