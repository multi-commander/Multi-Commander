"""
Dueling DQN implementation using tensorflow
"""

import tensorflow as tf
import numpy as np
import random 
from collections import deque
import copy

class DuelingDQNAgent(object):
    def __init__(self, config):
        self.state_size = config['state_size']
        self.action_size = config['action_size']
        self.memory = deque(maxlen=2000)
        self.gamma = 0.95 # discount factor
        self.epsilon = 1.0 # exploration rate 
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.learning_rate = 0.001
        # self.update_target_freq = 
        self.batch_size = 32
        self.qmodel = None
        self.target_model = None 
        self.layer_size = {'shared':[20],
                            'V':[1],
                            'A':[20, self.action_size]}
        self.global_step = 0

        
        self.sess = tf.Session()
        self.sess.__enter__()
        
        self._build_model()
        self.sess.run(tf.global_variables_initializer())
        self.update_target_network()

        self.saver = tf.train.Saver() # must after initializer

        intersection_id = list(config['lane_phase_info'].keys())[0]
        self.phase_list = config['lane_phase_info'][intersection_id]['phase']
    
    def _build_model(self):
        self.state = tf.placeholder(tf.float32, [None, ] + [self.state_size], name='state')
        self.state_ = tf.placeholder(tf.float32, [None, ] + [self.state_size], name='state_')
        self.q_target = tf.placeholder(tf.float32, [None, ] + [self.action_size], name='q_target')
        
        # with tf.variable_scope('qnet'):
        #     pass

        # with tf.variable_scope('target'):
        #     pass
       
        self.qmodel_output = self._build_network('qnet', self.state, self.layer_size)
        self.targte_model_output = self._build_network('target', self.state_, self.layer_size)
        
        # loss, and other operations
        with tf.variable_scope('loss'):
            self.q_loss = tf.reduce_mean(tf.squared_difference(self.qmodel_output, self.q_target))
        with tf.variable_scope('train'):
            self.train_op = tf.train.AdamOptimizer(self.learning_rate).minimize(self.q_loss)
        
        # replace target net with q net
        self.q_net_params = tf.get_collection('qnet')
        self.target_net_paprams = tf.get_collection('target')
        self.copy_target_op = [tf.assign(t, q) for t, q in zip(self.target_net_paprams, self.q_net_params)]
        

    def _build_network(self, scope, state, layer_size):
        with tf.variable_scope(scope):
            with tf.variable_scope('shared'):
                hidden = state
                shared_layer_size = layer_size['shared']
                for size in shared_layer_size:
                    hidden = tf.layers.dense(hidden, size,
                            bias_initializer=tf.constant_initializer(0.1),
                            kernel_initializer=tf.random_normal_initializer(0.1, 0.3))
                    hidden = tf.nn.relu(hidden)

            
            with tf.variable_scope('Value'):
                V = hidden
                V_size = layer_size['V']
                for size in V_size:
                    V = tf.layers.dense(V, size,
                        bias_initializer=tf.constant_initializer(0.1),
                        kernel_initializer=tf.random_normal_initializer(0.1, 0.3))
                    # no relu

            
            with tf.variable_scope('Advantage'):
                A = hidden
                A_size = layer_size['A']
                for size in A_size:
                    A = tf.layers.dense(A, size,
                        bias_initializer=tf.constant_initializer(0.1),
                        kernel_initializer=tf.random_normal_initializer(0.1, 0.3))

            with tf.variable_scope('Q'):
                out = V + (A - tf.reduce_mean(A, axis=1, keep_dims=True))
                
        return out
    
    
    def choose_action(self, state):
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)
        q_values = tf.get_default_session().run(self.qmodel_output, feed_dict={self.state: state})
        return np.argmax(q_values[0])

    def replay(self):
        minibatch = random.sample(self.memory, self.batch_size)
        states = []
        q_target = []
        for state, action, reward, next_state in minibatch:
            states.append(state)
            q_eval = tf.get_default_session().run(self.qmodel_output, feed_dict={self.state:state})
            q_next = tf.get_default_session().run(self.qmodel_output, feed_dict={self.state:next_state})

            target_value = reward + self.gamma * np.max(q_next)
            # q_target_ = copy.copy(q_eval)
            q_target_ = q_eval.copy()
            q_target_[0][action] = target_value
            q_target.append(q_target_)
        
        states = np.reshape(np.array(states), [-1, self.state_size])
        q_target = np.reshape(np.array(q_target), [-1, self.action_size])

        feed_dict = {self.state:states,
                    self.q_target:q_target}
        # batch training
        tf.get_default_session().run(self.train_op, feed_dict=feed_dict)
    
    def update_target_network(self):
        tf.get_default_session().run(self.copy_target_op)
    
    def remember(self, state, action, reward, next_state):
        action = self.phase_list.index(action)
        self.memory.append((state, action, reward, next_state))
    
    def save(self, ckpt, epoch):
        self.saver.save(self.sess, ckpt, global_step=epoch)
        print("model saved: {}-{}".format(ckpt, epoch))

    def load(self, ckpt):
        self.saver.restore(self.sess, ckpt)
        


