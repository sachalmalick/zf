#this model is going to take a noise vector of [batches, latent_dim]
#it will output a modified noise vector and an integer 0 <= x <= latent_dim


import tensorflow as tf
import tensorflow_probability as tfp

# Drafting a basic DQN integration into the Cherrypicker class
import tensorflow as tf
import tensorflow_probability as tfp
import numpy as np
import random
from collections import deque

class Cherrypicker(tf.keras.Model):
    def __init__(self, range, state_size, action_size, learning_rate=0.001):
        super().__init__()
        self.range = range
        self.state_size = state_size
        self.action_size = action_size
        self.memory = deque(maxlen=2000)
        self.gamma = 0.95    # discount rate
        self.epsilon = 1.0  # exploration rate
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.learning_rate = learning_rate
        self.model = self._build_model()

    def _build_model(self):
        # Neural Net for Deep-Q learning Model
        model = tf.keras.Sequential()
        model.add(tf.keras.layers.Dense(24, input_dim=self.state_size, activation='relu'))
        model.add(tf.keras.layers.Dense(24, activation='relu'))
        model.add(tf.keras.layers.Dense(self.action_size, activation='linear'))
        model.compile(loss='mse', optimizer=tf.keras.optimizers.Adam(lr=self.learning_rate))
        return model

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state):
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)
        act_values = self.model.predict(state)
        return np.argmax(act_values[0])  # returns action

    def replay(self, batch_size):
        minibatch = random.sample(self.memory, batch_size)
        for state, action, reward, next_state, done in minibatch:
            target = reward
            if not done:
                target = (reward + self.gamma * np.amax(self.model.predict(next_state)[0]))
            target_f = self.model.predict(state)
            target_f[0][action] = target
            self.model.fit(state, target_f, epochs=1, verbose=0)
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def modify_noise_vector(self, noise_vector):
        # This method would use the DQN to decide how to modify the noise vector
        # Example: selecting an action based on the current state (noise vector)
        action = self.act(noise_vector)
        # Modify the noise_vector based on the chosen action
        # Return the modified noise_vector and the chosen action (or related information)
        return modified_noise_vector, action

# Displaying the drafted DQN integration code
print(dqn_code)
