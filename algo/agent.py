import random
from collections import deque
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model, Sequential, load_model, clone_model
from tensorflow.keras.layers import Input, Add, Lambda, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import Huber

class CustomAgent:
    def __init__(self, state_size, strategy="t-dqn", reset_every=100, pretrained=False, model_name="./custom_model/custom_model.h5"):
        self.strategy = strategy
        self.state_size = state_size + 11
        self.action_size = 3
        self.model_name = model_name
        self.inventory = []
        self.memory = deque(maxlen=64)
        self.first_iter = True
        self.gamma = 0.95
        self.epsilon = 1.0
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.learning_rate = 0.001
        self.loss = Huber(delta=1.0)
        self.custom_objects = {"huber_loss": Huber(delta=1.0)}
        self.optimizer = Adam(learning_rate=self.learning_rate)

        if pretrained and self.model_name is not None:
            self.model = self.load_model()
        else:
            self.model = self.build_model()

        if self.strategy in ["t-dqn", "double-dqn", 'dueling-dqn']:
            self.n_iter = 1
            self.reset_every = reset_every
            self.target_model = clone_model(self.model)
            self.target_model.set_weights(self.model.get_weights())

    def build_model(self):
        model = Sequential()
        model.add(Dense(units=16, activation="relu", input_dim=self.state_size))
        model.add(Dense(units=32, activation="relu"))
        # model.add(Dense(units=32, activation="relu"))
        model.add(Dense(units=16, activation="relu"))
        model.add(Dense(units=self.action_size))
        model.compile(loss=self.loss, optimizer=self.optimizer)
        return model

    def train_experience_replay(self, batch_size):
        mini_batch = random.sample(self.memory, batch_size)
        X_train, y_train = [], []

        if self.strategy == "dqn":
            for state, action, reward, next_state, done in mini_batch:
                target = reward if done else reward + self.gamma * np.amax(self.model.predict(next_state)[0])
                q_values = self.model.predict(state)
                q_values[0][action] = target
                X_train.append(state[0])
                y_train.append(q_values[0])

        elif self.strategy == "t-dqn":
            for state, action, reward, next_state, done in mini_batch:
                target = reward if done else reward + self.gamma * np.amax(self.target_model.predict(next_state)[0])
                q_values = self.model.predict(state)
                q_values[0][action] = target
                X_train.append(state[0])
                y_train.append(q_values[0])

        elif self.strategy == "double-dqn":
            if self.n_iter % self.reset_every == 0:
                self.target_model.set_weights(self.model.get_weights())

            for state, action, reward, next_state, done in mini_batch:
                target = reward if done else reward + self.gamma * self.target_model.predict(next_state)[0][np.argmax(self.model.predict(next_state)[0])]
                q_values = self.model.predict(state)
                q_values[0][action] = target
                X_train.append(state[0])
                y_train.append(q_values[0])

        elif self.strategy == "dueling-dqn":
            if self.n_iter % self.reset_every == 0:
                self.target_model.set_weights(self.model.get_weights())

            for state, action, reward, next_state, done in mini_batch:
                next_q_values = self.target_model.predict(next_state)
                best_actions = np.argmax(next_q_values, axis=1)
                next_best_state_values = self.model.predict(next_state)[np.arange(batch_size), best_actions]
                target = reward + self.gamma * next_best_state_values

                q_values = self.model.predict(state)
                q_values[0][action] = target

                X_train.append(state[0])
                y_train.append(q_values[0])

        else:
            raise NotImplementedError()

        loss = self.model.fit(
            np.array(X_train), np.array(y_train),
            epochs=1, verbose=0
        ).history["loss"][0]

        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

        return loss

    def take_action(self, state, is_eval=False):
        if not is_eval and random.random() <= self.epsilon:
            return random.randrange(self.action_size)

        if self.first_iter:
            self.first_iter = False
            return 1

        action_probs = self.model.predict(state)
        return np.argmax(action_probs[0])

    def store_memory(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def save_model(self, episode):
        self.model.save("{}_{}_{}.h5".format(self.model_name, self.state_size, episode))

    def load_model(self):
        return load_model(self.model_name, custom_objects=self.custom_objects)
