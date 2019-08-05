
import numpy as np
from keras.layers import Input, Dense, Conv2D, Flatten, BatchNormalization, Activation, Multiply, Add
from keras.models import Model, model_from_json, load_model
from keras.optimizers import RMSprop
from keras.layers.core import Dropout
from keras.layers.pooling import MaxPooling2D
from keras import backend as K
import random
from keras.engine.topology import Layer
import os
from keras.callbacks import EarlyStopping, TensorBoard
import traceback
import pickle as pkl



from agent import Agent

class Selector(Layer):

    def __init__(self, select, **kwargs):
        super(Selector, self).__init__(**kwargs)
        self.select = select
        self.select_neuron = K.constant(value=self.select)

    def build(self, input_shape):
        # Create a trainable weight variable for this layer.
        super(Selector, self).build(input_shape)  # Be sure to call this somewhere!

    def call(self, x):
        return K.cast(K.equal(x, self.select_neuron), dtype="float32")

    def get_config(self):
        config = {"select": self.select}
        base_config = super(Selector, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

    def compute_output_shape(self, input_shape):
        return input_shape


def conv2d_bn(input_layer, index_layer,
              filters=16,
              kernel_size=(3, 3),
              strides=(1, 1)):
    """Utility function to apply conv + BN.
    # Arguments
        x: input tensor.
        filters: filters in `Conv2D`.
        num_row: height of the convolution kernel.
        num_col: width of the convolution kernel.
        padding: padding mode in `Conv2D`.
        strides: strides in `Conv2D`.
        name: name of the ops; will become `name + '_conv'`
            for the convolution and `name + '_bn'` for the
            batch norm layer.
    # Returns
        Output tensor after applying `Conv2D` and `BatchNormalization`.
    """
    if K.image_data_format() == 'channels_first':
        bn_axis = 1
    else:
        bn_axis = 3
    conv = Conv2D(filters=filters,
                  kernel_size=kernel_size,
                  strides=strides,
                  padding='same',
                  use_bias=False,
                  name="conv{0}".format(index_layer))(input_layer)
    bn = BatchNormalization(axis=bn_axis, scale=False, name="bn{0}".format(index_layer))(conv)
    act = Activation('relu', name="act{0}".format(index_layer))(bn)
    pooling = MaxPooling2D(pool_size=2)(act)
    x = Dropout(0.3)(pooling)
    return x


class NetworkAgent(Agent):
    def __init__(self, dic_agent_conf, dic_traffic_env_conf, dic_path, cnt_round,
                 best_round=None, bar_round=None, intersection_id="0"):

        super(NetworkAgent, self).__init__(
            dic_agent_conf, dic_traffic_env_conf, dic_path, intersection_id=intersection_id)

        self.num_actions = len(dic_traffic_env_conf["PHASE"][dic_traffic_env_conf['SIMULATOR_TYPE']])
        self.num_phases = len(dic_traffic_env_conf["PHASE"][dic_traffic_env_conf['SIMULATOR_TYPE']])
        self.num_lanes = np.sum(np.array(list(self.dic_traffic_env_conf["LANE_NUM"].values())))

        self.memory = self.build_memory()
        self.cnt_round = cnt_round

        if cnt_round == 0: 
            # initialization

            if os.listdir(self.dic_path["PATH_TO_MODEL"]):
                if self.dic_traffic_env_conf['ONE_MODEL']:
                    self.load_network("round_0")
                else:
                    self.load_network("round_0_inter_{0}".format(intersection_id))
            elif self.dic_traffic_env_conf["TRANSFER"] and self.dic_traffic_env_conf['ONE_MODEL']:
                self.load_network_transfer(file_name="{0}".format(self.dic_traffic_env_conf["TRAFFIC_FILE"].split('.json')[0]))
            else:
                self.q_network = self.build_network()
            #self.load_network(self.dic_agent_conf["TRAFFIC_FILE"], file_path=self.dic_path["PATH_TO_PRETRAIN_MODEL"])
            self.q_network_bar = self.build_network_from_copy(self.q_network)
        else:
            try:
                self.load_network("round_{0}_inter_{1}".format(cnt_round-1, self.intersection_id))

                if "UPDATE_Q_BAR_EVERY_C_ROUND" in self.dic_agent_conf:
                    if self.dic_agent_conf["UPDATE_Q_BAR_EVERY_C_ROUND"]:
                        self.load_network_bar("round_{0}_inter_{1}".format(
                            max((cnt_round - 1) // self.dic_agent_conf["UPDATE_Q_BAR_FREQ"] * self.dic_agent_conf["UPDATE_Q_BAR_FREQ"], 0),
                            self.intersection_id))
                    else:
                        self.load_network_bar("round_{0}_inter_{1}".format(
                            max(cnt_round - self.dic_agent_conf["UPDATE_Q_BAR_FREQ"], 0),
                            self.intersection_id))
                else:
                    self.load_network_bar("round_{0}_inter_{1}".format(
                        max(cnt_round - self.dic_agent_conf["UPDATE_Q_BAR_FREQ"], 0), self.intersection_id))

            except Exception as e:
                print('traceback.format_exc():\n%s' % traceback.format_exc())
                return 0, None
        # decay the epsilon

        decayed_epsilon = self.dic_agent_conf["EPSILON"] * pow(self.dic_agent_conf["EPSILON_DECAY"], cnt_round)
        self.dic_agent_conf["EPSILON"] = max(decayed_epsilon, self.dic_agent_conf["MIN_EPSILON"])

    @staticmethod
    def _unison_shuffled_copies(Xs, Y, sample_weight):
        p = np.random.permutation(len(Y))
        new_Xs = []
        for x in Xs:
            assert len(x) == len(Y)
            new_Xs.append(x[p])
        return new_Xs, Y[p], sample_weight[p]

    @staticmethod
    def _cnn_network_structure(img_features):
        conv1 = conv2d_bn(img_features, 1, filters=32, kernel_size=(8, 8), strides=(4, 4))
        conv2 = conv2d_bn(conv1, 2, filters=16, kernel_size=(4, 4), strides=(2, 2))
        img_flatten = Flatten()(conv2)
        return img_flatten

    @staticmethod
    def _shared_network_structure(state_features, dense_d):
        hidden_1 = Dense(dense_d, activation="sigmoid", name="hidden_shared_1")(state_features)
        return hidden_1

    @staticmethod
    def _separate_network_structure(state_features, dense_d, num_actions, memo=""):
        hidden_1 = Dense(dense_d, activation="sigmoid", name="hidden_separate_branch_{0}_1".format(memo))(state_features)
        q_values = Dense(num_actions, activation="linear", name="q_values_separate_branch_{0}".format(memo))(hidden_1)
        return q_values

    def load_network(self, file_name, file_path=None):
        if file_path == None:
            file_path = self.dic_path["PATH_TO_MODEL"]
        self.q_network = load_model(os.path.join(file_path, "%s.h5" % file_name), custom_objects={"Selector": Selector})
        print("succeed in loading model %s"%file_name)

    def load_network_(self, file_name):

        self.q_network = load_model(file_name, custom_objects={"Selector": Selector})
        print("succeed in loading model %s"%file_name)

    def load_network_transfer(self, file_name, file_path=None):
        if file_path == None:
            file_path = self.dic_path["PATH_TO_TRANSFER_MODEL"]
        self.q_network = load_model(os.path.join(file_path, "%s.h5" % file_name), custom_objects={"Selector": Selector})
        print("succeed in loading model %s"%file_name)

    def load_network_bar(self, file_name, file_path=None):
        if file_path == None:
            file_path = self.dic_path["PATH_TO_MODEL"]
        self.q_network_bar = load_model(os.path.join(file_path, "%s.h5" % file_name), custom_objects={"Selector": Selector})
        print("succeed in loading model %s"%file_name)

    def save_network(self, file_name):
        self.q_network.save(os.path.join(self.dic_path["PATH_TO_MODEL"], "%s.h5" % file_name))

    def save_network_bar(self, file_name):
        self.q_network_bar.save(os.path.join(self.dic_path["PATH_TO_MODEL"], "%s.h5" % file_name))

    def build_network(self):

        raise NotImplementedError

    def build_memory(self):

        return []

    def build_network_from_copy(self, network_copy):

        '''Initialize a Q network from a copy'''

        network_structure = network_copy.to_json()
        network_weights = network_copy.get_weights()
        network = model_from_json(network_structure, custom_objects={"Selector": Selector})
        network.set_weights(network_weights)
        network.compile(optimizer=RMSprop(lr=self.dic_agent_conf["LEARNING_RATE"]),
                        loss=self.dic_agent_conf["LOSS_FUNCTION"])
        return network


    def prepare_Xs_Y(self, memory, dic_exp_conf):

        ind_end = len(memory)
        print("memory size before forget: {0}".format(ind_end))
        # use all the samples to pretrain, i.e., without forgetting
        if dic_exp_conf["PRETRAIN"] or dic_exp_conf["AGGREGATE"]:
            sample_slice = memory
        elif self.dic_agent_conf["PRIORITY"]:
            print("priority")
            sample_slice = []
            num_sample_list = [int(self.dic_agent_conf["MAX_MEMORY_LEN"] * 1 / 4),]*4
            for i in range(ind_end-1,-1,-1):
                one_sample = memory[i]
                print(one_sample)
                one_sample_lane_num_vehicle = one_sample[0]['lane_num_vehicle']
                num_veh = max(np.array(one_sample_lane_num_vehicle))
                interval_id = num_veh // 10
                if num_sample_list[interval_id] > 0:
                    sample_slice.append(one_sample)
                    num_sample_list[interval_id] -= 1

                if np.sum(np.array(num_sample_list)) == 0:
                    break
            print("end_priority", num_sample_list)
            sample_size = min(self.dic_agent_conf["SAMPLE_SIZE"],len(sample_slice))
            sample_slice = random.sample(sample_slice,sample_size)
            pkl.dump(sample_slice, file=open(
                os.path.join(self.dic_path['PATH_TO_WORK_DIRECTORY'], "train_round", "round_" + str(self.cnt_round),
                             "update_sample.pkl"), "ab"))

        # forget
        else:
            ind_sta = max(0, ind_end - self.dic_agent_conf["MAX_MEMORY_LEN"])
            memory_after_forget = memory[ind_sta: ind_end]
            print("memory size after forget:", len(memory_after_forget))

            # sample the memory
            sample_size = min(self.dic_agent_conf["SAMPLE_SIZE"], len(memory_after_forget))
            sample_slice = random.sample(memory_after_forget, sample_size)
            print("memory samples number:", sample_size)

        dic_state_feature_arrays = {}
        for feature_name in self.dic_traffic_env_conf["LIST_STATE_FEATURE"]:
            dic_state_feature_arrays[feature_name] = []
        Y = []

        for i in range(len(sample_slice)):
            state, action, next_state, reward, instant_reward, _, _ = sample_slice[i]
            # print(state)

            for feature_name in self.dic_traffic_env_conf["LIST_STATE_FEATURE"]:
                dic_state_feature_arrays[feature_name].append(state[feature_name])

            _state = []
            _next_state = []
            for feature_name in self.dic_traffic_env_conf["LIST_STATE_FEATURE"]:
                _state.append([state[feature_name]])
                _next_state.append([next_state[feature_name]])
            target = self.q_network.predict(_state)

            next_state_qvalues = self.q_network_bar.predict(_next_state)

            if self.dic_agent_conf["LOSS_FUNCTION"] == "mean_squared_error":
                final_target = np.copy(target[0])
                final_target[action] = reward / self.dic_agent_conf["NORMAL_FACTOR"] + self.dic_agent_conf["GAMMA"] * \
                                       np.max(next_state_qvalues[0])
            elif self.dic_agent_conf["LOSS_FUNCTION"] == "categorical_crossentropy":
                raise NotImplementedError

            Y.append(final_target)

        self.Xs = [np.array(dic_state_feature_arrays[feature_name]) for feature_name in
                   self.dic_traffic_env_conf["LIST_STATE_FEATURE"]]
        self.Y = np.array(Y)


    def convert_state_to_input(self, s):
        if self.dic_traffic_env_conf["BINARY_PHASE_EXPANSION"]:
            inputs = []
            # print(s)
            for feature in self.dic_traffic_env_conf["LIST_STATE_FEATURE"]:
                if "cur_phase" in feature:

                    inputs.append(np.array([self.dic_traffic_env_conf['PHASE']
                                            [self.dic_traffic_env_conf['SIMULATOR_TYPE']]
                                            [s[feature][0]]]))
                else:
                    inputs.append(np.array([s[feature]]))
            return inputs
        else:
            return [np.array([s[feature]]) for feature in self.dic_traffic_env_conf["LIST_STATE_FEATURE"]]


    def choose_action(self, count, state):

        ''' choose the best action for current state '''
        state_input = self.convert_state_to_input(state)
        q_values = self.q_network.predict(state_input)
        if random.random() <= self.dic_agent_conf["EPSILON"]:  # continue explore new Random Action
            action = random.randrange(len(q_values[0]))
        else:  # exploitation
            action = np.argmax(q_values[0])

        return action

    def train_network(self, dic_exp_conf):

        if dic_exp_conf["PRETRAIN"] or dic_exp_conf["AGGREGATE"]:
            epochs = 1000
        else:
            epochs = self.dic_agent_conf["EPOCHS"]
        batch_size = min(self.dic_agent_conf["BATCH_SIZE"], len(self.Y))

        early_stopping = EarlyStopping(
            monitor='val_loss', patience=self.dic_agent_conf["PATIENCE"], verbose=0, mode='min')

        hist = self.q_network.fit(self.Xs, self.Y, batch_size=batch_size, epochs=epochs,
                                  shuffle=False,
                                  verbose=2, validation_split=0.3, callbacks=[early_stopping])
