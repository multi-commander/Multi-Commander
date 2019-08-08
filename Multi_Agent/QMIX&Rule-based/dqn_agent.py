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

# os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"
# KTF.set_session(tf.Session(config=tf.ConfigProto(device_count={'gpu':0})))

class DQNAgent(object):
    def __init__(self, 
            intersection_id,
            state_size=9,
            action_size=8,
            batch_size=32,
            phase_list=[],
            env=None
            ):
        self.env = env
        self.intersection_id = intersection_id
        self.state_size = state_size
        self.action_size = action_size
        self.memory = deque(maxlen=3000)
        self.gamma = 0.95    # discount rate
        self.epsilon = 1.0  # exploration rate
        self.epsilon_min = 0.1
        self.epsilon_decay = 0.9995
        self.learning_rate = 0.001
        self.batch_size = batch_size
        self.model = self._build_model()
        self.target_model = self._build_model()
        self.update_target_network()
        
        self.phase_list = phase_list

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
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)
        act_values = self.model.predict(state)
        return np.argmax(act_values[0])  # returns action

    def choose_action_(self, state):
        '''
        choose phase with max waiting count in its start lanes
        '''
        intersection_info = self.env.intersection_info(self.intersection_id)
        phase_waiting_count = self.phase_list.copy()
        for index, phase in enumerate(self.phase_list):
            phase_start_lane = self.env.phase_startLane_mapping[self.intersection_id][phase] # {0:['road_0_1_1_1',], 1:[]...}
            phase_start_lane_waiting_count = [intersection_info["start_lane_vehicle_count"][lane] for lane  in phase_start_lane] # [num1, num2, ...]
            sum_count = sum(phase_start_lane_waiting_count)
            phase_waiting_count[index] = sum_count
        
        action = np.argmax(phase_waiting_count)
        return action

    def replay(self):
        minibatch = random.sample(self.memory, self.batch_size)
        states = []
        q_targets = []
        for state, action, reward, next_state in minibatch:
            target = (reward + self.gamma *
                      np.amax(self.target_model.predict(next_state)[0]))
            target_f = self.model.predict(state)
            target_f[0][action] = target # action is a action_list index

            states.append(state)
            q_targets.append(target_f)

            # self.model.fit(state, target_f, epochs=1, verbose=0)
        
        states = np.reshape(np.array(states), [-1, self.state_size])
        q_targets = np.reshape(np.array(q_targets), [-1, self.action_size])
        self.model.fit(state, target_f, epochs=2, verbose=0) # batch training

        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def load(self, name):
        self.model.load_weights(name)

    def save(self, name):
        self.model.save_weights(name)
        print("model saved:{}".format(name))


class DDQNAgent(DQNAgent):
    def __init__(self, config):
        super(DDQNAgent, self).__init__(config)

    # override
    def replay(self):
        minibatch = random.sample(self.memory, self.batch_size)
        states = []
        q_targets = []
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

            states.append(state)
            q_targets.append(target_f)
            # self.model.fit(state, target_f, epochs=1, verbose=0) # train on single sample

        states = np.reshape(np.array(states), [-1, self.state_size])
        q_targets = np.reshape(np.array(q_targets), [-1, self.action_size])
        self.model.fit(state, target_f, epochs=1, verbose=0) # batch training
        
        if self.epsilon > self.epsilon_min:
            self.epsilon = self.epsilon_decay

class MDQNAgent(object):
    def __init__(self,
        intersection, 
        state_size=17,
        batch_size=32,
        phase_list={},
        env=None):
        
        self.intersection = intersection
        self.agents =  {}
        self.make_agents(intersection, state_size, batch_size, phase_list, env)

    def make_agents(self, intersection, state_size, batch_size, phase_list, env):
        for id_ in self.intersection: 
            self.agents[id_] = DQNAgent(id_, 
                                state_size=state_size,
                                action_size=len(phase_list[id_]),
                                batch_size=batch_size,
                                phase_list=phase_list[id_],
                                env=env
                                )

    def update_target_network(self):
        for id_ in self.intersection:
            self.agents[id_].update_target_network()

    def remember(self, state, action, reward, next_state):
        for id_ in self.intersection:
            self.agents[id_].remember(state[id_],
                                    action[id_],
                                    reward[id_],
                                    next_state[id_])
    
    def choose_action(self, state):
        action = {}
        for id_ in self.intersection:
            action[id_] = self.agents[id_].choose_action_(state[id_])
        return action

    def replay(self):
        for id_ in self.intersection:
            self.agents[id_].replay()

    def load(self, name):
        for id_ in self.intersection:
            assert os.path.exists(name + '.' + id_), "Wrong checkpoint, file not exists!"
            self.agents[id_].load(name + '.' + id_)

    def save(self, name):
        for id_ in self.intersection:
            self.agents[id_].save(name + '.' + id_)

