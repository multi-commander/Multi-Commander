"""
DQN and Double DQN agent implementation using Keras
"""

import random
import numpy as np
from collections import deque
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam
import keras.backend.tensorflow_backend as KTF
import tensorflow as tf
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"
KTF.set_session(tf.Session(config=tf.ConfigProto(device_count={'gpu':0})))

class DQNAgent:
    def __init__(self, config):
        self.state_size = config['state_size']
        self.action_size = config['action_size']
        self.memory = deque(maxlen=3000)
        self.gamma = 0.95    # discount rate
        self.epsilon = 0.9  # exploration rate
        self.epsilon_min = 0.1
        self.epsilon_decay = 0.995
        self.learning_rate = 0.001
        self.update_target_freq = 5
        self.batch_size = config['batch_size']
        self.model = self._build_model()
        self.target_model = self._build_model()
        self.update_target_network()

        intersection_id = list(config['lane_phase_info'].keys())[0]
        self.phase_list = config['lane_phase_info'][intersection_id]['phase']

    def _build_model(self):
        # Neural Net for Deep-Q learning Model
        model = Sequential()
        model.add(Dense(40, input_dim=self.state_size, activation='relu'))
        model.add(Dense(40, activation='relu'))
        model.add(Dense(self.action_size, activation='linear'))
        model.compile(loss='mse',
                      optimizer=Adam(lr=self.learning_rate))
        return model

    def update_target_network(self):
        weights = self.model.get_weights()
        self.target_model.set_weights(weights)

    def remember(self, state, action, reward, next_state):
        action = self.phase_list.index(action) # index
        self.memory.append((state, action, reward, next_state))

    def choose_action(self, state):
        # if np.random.rand() >= self.epsilon:
        #     return random.randrange(self.action_size)
        # act_values = self.model.predict(state)
        # return np.argmax(act_values[0])  # returns action
        state=state[0][0:8].tolist()
        action=np.argmax(state)
        return action

    def replay(self):
        minibatch = random.sample(self.memory, self.batch_size)
        for state, action, reward, next_state in minibatch:
            test=self.gamma * np.amax(self.target_model.predict(next_state)[0])
            target = (reward + self.gamma *
                      np.amax(self.target_model.predict(next_state)[0]))
            target_f = self.model.predict(state)
            target_f[0][action] = target # action is a action_list index
            self.model.fit(state, target_f, epochs=10, verbose=0)
        # if self.epsilon >= self.epsilon_min:
        #     self.epsilon *= self.epsilon_decay

    def load(self, name):
        self.model.load_weights(name, by_name=True)

    def save(self, name):
        self.model.save_weights(name)
        print("model saved:{}".format(name))


class DDQNAgent(DQNAgent):
    def __init__(self, config):
        super(DDQNAgent, self).__init__(config)

    # override
    def replay(self):
        minibatch = random.sample(self.memory, self.batch_size)
        for state, action, reward, next_state in minibatch:
            # compute target value, this is the key point of Double DQN
            
            # target
            target_f = self.model.predict(state)

            # choose best action for next state using current Q network
            actions_for_next_state = np.argmax(self.model.predict(next_state)[0])
            
            # compute target value
            target = (reward + self.gamma *
                      self.target_model.predict(next_state)[0][actions_for_next_state] )
            
            target_f[0][action] = target

            self.model.fit(state, target_f, epochs=1, verbose=0) # train on single sample
        
        if self.epsilon > self.epsilon_min:
            self.epsilon = self.epsilon_decay

    

