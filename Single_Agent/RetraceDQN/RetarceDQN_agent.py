import random
import numpy as np
from collections import deque
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam

from Single_Agent.DQN_DDQN_DuelingDQN.dqn_agent import *

class RetraceDQNAgent(DQNAgent):
    def __init__(self,config):
        super(RetraceDQNAgent,self).__init__(config)

    def replay(self):
        minibatch = random.sample(self.memory, self.batch_size)
        for state, action, reward, next_state in minibatch:


            self.model.fit(state, target_f, epochs=1, verbose=0)  # train on single sample
        if self.epsilon > self.epsilon_min:
            self.epsilon = self.epsilon_decay