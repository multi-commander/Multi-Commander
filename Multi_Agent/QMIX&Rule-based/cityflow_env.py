import cityflow
import pandas as pd
import os
import json
import math
import numpy as np
import itertools
# from sim_setting import sim_setting_control

class CityFlowEnv(object):
    def __init__(self,
                lane_phase_info,
                intersection_id='xxx',
                num_step=1500,
                thread_num=1,
                cityflow_config_file='examples/config_1x1.json',
                replay_data_path='./replay'
                ):
        # cityflow_config['rlTrafficLight'] = rl_control # use RL to control the light or not
        self.eng = cityflow.Engine(cityflow_config_file, thread_num=thread_num)
        self.num_step = num_step
        self.state_size = None
        self.lane_phase_info = lane_phase_info # "intersection_1_1"
        self.intersection_id = intersection_id
        self.start_lane = self.lane_phase_info[self.intersection_id]['start_lane']
        self.end_lane = self.lane_phase_info[self.intersection_id]['end_lane']

        self.phase_list = self.lane_phase_info[self.intersection_id]["phase"]
        self.phase_startLane_mapping = self.lane_phase_info[self.intersection_id]["phase_startLane_mapping"]

        self.replay_data_path = replay_data_path
        self.current_phase = {self.intersection_id:self.phase_list[0]}
        self.current_phase_time = {self.intersection_id:0}
        self.yellow_time = 5
        self.state_store_i = 0
        self.get_state() # set self.state_size
        self.phase_log = []

    def reset(self):
        self.eng.reset()

    def step(self, next_phase):
        if self.current_phase[self.intersection_id] == next_phase:
            self.current_phase_time[self.intersection_id] += 1
        else:
            self.current_phase[self.intersection_id] = next_phase
            self.current_phase_time[self.intersection_id] = 1

        self.eng.set_tl_phase(self.intersection_id, self.current_phase[self.intersection_id]) # set phase of traffic light
        self.eng.next_step()
        self.phase_log.append(self.current_phase[self.intersection_id])
        return self.get_state(), self.get_reward() # return next_state and reward

    def get_state(self):
        intersection_info = self.intersection_info(self.intersection_id)
        state_dict = intersection_info['start_lane_vehicle_count']
        return_state = [state_dict[key] for key in sorted(state_dict.keys())] + [intersection_info['current_phase'][self.intersection_id]]
        return self.preprocess_state(return_state)

    def preprocess_state(self, state):
        return_state = np.array(state)
        if self.state_size is None:
            self.state_size = len(return_state.flatten())
        return_state = np.reshape(return_state, [1, self.state_size])
        return return_state

    def intersection_info(self, id_):
        '''
        info of intersection 'id_'
        '''
        state = {}
        get_lane_vehicle_count = self.eng.get_lane_vehicle_count()
        get_lane_waiting_vehicle_count = self.eng.get_lane_waiting_vehicle_count()
        get_lane_vehicles = self.eng.get_lane_vehicles()
        vehicle_speed = self.eng.get_vehicle_speed()

        state['start_lane_vehicle_count'] = {lane: get_lane_vehicle_count[lane] for lane in self.start_lane[id_]}
        state['end_lane_vehicle_count'] = {lane: get_lane_vehicle_count[lane] for lane in self.end_lane[id_]}
        
        state['start_lane_waiting_vehicle_count'] = {lane: get_lane_waiting_vehicle_count[lane] for lane in self.start_lane[id_]}
        state['end_lane_waiting_vehicle_count'] = {lane: get_lane_waiting_vehicle_count[lane] for lane in self.end_lane[id_]}
        
        state['start_lane_vehicles'] = {lane: get_lane_vehicles[lane] for lane in self.start_lane[id_]}
        state['end_lane_vehicles'] = {lane: get_lane_vehicles[lane] for lane in self.end_lane[id_]}
        
        state['start_lane_speed'] = {lane: np.sum(list(map(lambda vehicle:vehicle_speed[vehicle], get_lane_vehicles[lane]))) / (get_lane_vehicle_count[lane]+1e-5) for lane in self.start_lane[id_]} # compute start lane mean speed
        state['end_lane_speed'] = {lane: np.sum(list(map(lambda vehicle:vehicle_speed[vehicle], get_lane_vehicles[lane]))) / (get_lane_vehicle_count[lane]+1e-5) for lane in self.end_lane[id_]} # compute end lane mean speed
        
        state['current_phase'] = self.current_phase[id_]
        state['current_phase_time'] = self.current_phase_time[id_]

        return state

    # def get_reward(self):
    #     '''
    #     max vehicle count of the lanes, *-1
    #     '''
    #     lane_vehicle_count = self.eng.get_lane_vehicles()
    #     for key in lane_vehicle_count.keys():
    #         lane_vehicle_count[key] = len(lane_vehicle_count[key])
    #     reward = -1 * max(list(lane_vehicle_count.values()))
    #     return reward

    # def get_reward(self):
    #     '''
    #     max vehicle count of all start lanes
    #     '''
    #     start_lane_vehicle_count = {lane: self.eng.get_lane_vehicle_count()[lane] for lane in self.start_lane}
    #     reward = -1 * np.mean(list(start_lane_vehicle_count.values()))
    #     return reward

    # def get_reward(self):
    #     '''
    #     max waiting vehicle count of the lanes, *-1
    #     '''
    #     lane_waiting_vehicle_count = self.eng.get_lane_waiting_vehicle_count()
    #     lane_waiting_vehicle_count_list = list(lane_waiting_vehicle_count.values())
    #     reward = -1 * ( sum(lane_waiting_vehicle_count_list)/len(lane_waiting_vehicle_count_list)  )
    #     return reward
    
    def get_reward(self):
        '''
        mean speed of start lanes
        '''
        intersection_info = self.intersection_info(self.intersection_id)

        start_lane_vehicles = intersection_info["start_lane_vehicle_count"]

        start_lane_vehicles = list(itertools.chain(*start_lane_vehicles))
        vehicle_speed = self.eng.get_vehicle_speed()
        start_lane_vehicles_speed = [vehicle_speed[v] for v in start_lane_vehicles]
        reward = sum(start_lane_vehicles_speed)/(len(start_lane_vehicles_speed) + 1e-5) * 100
        return reward
    
    def _get_pressure(self):
        return [self.dic_lane_waiting_vehicle_count_current_step[lane] for lane in self.list_entering_lanes] + \
               [-self.dic_lane_waiting_vehicle_count_current_step[lane] for lane in self.list_exiting_lanes]
    
    # def get_reward(self):
    #     # reward function
    #     lane_waiting_vehicle_count = self.eng.get_lane_vehicle_count()
    #     lane_waiting_vehicle_count_list = list(lane_waiting_vehicle_count.values())
    #     reward = -1 * max(lane_waiting_vehicle_count_list)
    #     return reward

    def get_score(self):
        lane_waiting_vehicle_count = self.eng.get_lane_waiting_vehicle_count()
        reward = -1 * sum(list(lane_waiting_vehicle_count.values()))
        metric = (1/(1 + math.exp(-1 * reward))) / self.num_step
        return metric
        # return 0

    def log(self):
        if not os.path.exists(self.replay_data_path):
            os.makedirs(self.replay_data_path)
        # self.eng.print_log(self.config['replay_data_path'] + "/replay_roadnet.json",
        #                    self.config['replay_data_path'] + "/replay_flow.json")
        df = pd.DataFrame({self.intersection_id: self.phase_log[:self.num_step]})
        df.to_csv(os.path.join(self.replay_data_path, 'signal_plan.txt'), index=None)

class CityFlowEnvM(object):
    '''
    multi inersection cityflow environment
    '''
    def __init__(self,
                lane_phase_info,
                intersection_id,
                num_step=2000,
                thread_num=1,
                cityflow_config_file='example/config_1x2.json'
                ):
        self.eng = cityflow.Engine(cityflow_config_file, thread_num=thread_num)
        self.num_step = num_step
        self.intersection_id = intersection_id # list, [intersection_id, ...]
        self.state_size = None
        self.lane_phase_info = lane_phase_info # "intersection_1_1"

        self.current_phase = {}
        self.current_phase_time = {}
        self.start_lane = {}
        self.end_lane = {}
        self.phase_list = {}
        self.phase_startLane_mapping = {}
        self.intersection_lane_mapping = {} #{id_:[lanes]}

        for id_ in self.intersection_id:
            self.start_lane[id_] = self.lane_phase_info[id_]['start_lane']
            self.end_lane[id_] = self.lane_phase_info[id_]['end_lane']
            self.phase_startLane_mapping[id_] = self.lane_phase_info[id_]["phase_startLane_mapping"]

            self.phase_list[id_] = self.lane_phase_info[id_]["phase"]
            self.current_phase[id_] = self.phase_list[id_][0]
            self.current_phase_time[id_] = 0
        self.get_state() # set self.state_size
        
    def reset(self):
        self.eng.reset()

    def step(self, action):
        '''
        action: {intersection_id: phase, ...}
        '''
        for id_, a in action.items():
            if self.current_phase[id_] == a:
                self.current_phase_time[id_] += 1
            else:
                self.current_phase[id_] = a
                self.current_phase_time[id_] = 1
            self.eng.set_tl_phase(id_, self.current_phase[id_]) # set phase of traffic light
        self.eng.next_step()
        return self.get_state(), self.get_reward()

    def get_state(self):
        state =  {id_: self.get_state_(id_) for id_ in self.intersection_id}
        return state

    def get_state_(self, id_):
        state = self.intersection_info(id_)
        state_dict = state['start_lane_waiting_vehicle_count']
        sorted_keys = sorted(state_dict.keys())
        return_state = [state_dict[key] for key in sorted_keys] + [state['current_phase']]
        return self.preprocess_state(return_state)

    def intersection_info(self, id_):
        '''
        info of intersection 'id_'
        '''
        state = {}
        # state['lane_vehicle_count'] = self.eng.get_lane_vehicle_count()  # {lane_id: lane_count, ...}
        # state['start_lane_vehicle_count'] = {lane: self.eng.get_lane_vehicle_count()[lane] for lane in self.start_lane[id_]}
        # state['lane_waiting_vehicle_count'] = self.eng.get_lane_waiting_vehicle_count()  # {lane_id: lane_waiting_count, ...}
        # state['lane_vehicles'] = self.eng.get_lane_vehicles()  # {lane_id: [vehicle1_id, vehicle2_id, ...], ...}
        # state['vehicle_speed'] = self.eng.get_vehicle_speed()  # {vehicle_id: vehicle_speed, ...}
        # state['vehicle_distance'] = self.eng.get_vehicle_distance() # {vehicle_id: distance, ...}
        # state['current_time'] = self.eng.get_current_time()

        get_lane_vehicle_count = self.eng.get_lane_vehicle_count()
        get_lane_waiting_vehicle_count = self.eng.get_lane_waiting_vehicle_count()
        get_lane_vehicles = self.eng.get_lane_vehicles()
        vehicle_speed = self.eng.get_vehicle_speed()

        state['start_lane_vehicle_count'] = {lane: get_lane_vehicle_count[lane] for lane in self.start_lane[id_]}
        state['end_lane_vehicle_count'] = {lane: get_lane_vehicle_count[lane] for lane in self.end_lane[id_]}
        
        state['start_lane_waiting_vehicle_count'] = {lane: get_lane_waiting_vehicle_count[lane] for lane in self.start_lane[id_]}
        state['end_lane_waiting_vehicle_count'] = {lane: get_lane_waiting_vehicle_count[lane] for lane in self.end_lane[id_]}
        
        state['start_lane_vehicles'] = {lane: get_lane_vehicles[lane] for lane in self.start_lane[id_]}
        state['end_lane_vehicles'] = {lane: get_lane_vehicles[lane] for lane in self.end_lane[id_]}
        
        state['start_lane_speed'] = {lane: np.sum(list(map(lambda vehicle:vehicle_speed[vehicle], get_lane_vehicles[lane]))) / (get_lane_vehicle_count[lane]+1e-5) for lane in self.start_lane[id_]} # compute start lane mean speed
        state['end_lane_speed'] = {lane: np.sum(list(map(lambda vehicle:vehicle_speed[vehicle], get_lane_vehicles[lane]))) / (get_lane_vehicle_count[lane]+1e-5) for lane in self.end_lane[id_]} # compute end lane mean speed
        
        state['current_phase'] = self.current_phase[id_]
        state['current_phase_time'] = self.current_phase_time[id_]

        return state


    def preprocess_state(self, state):
        return_state = np.array(state)
        if self.state_size is None:
            self.state_size = len(return_state.flatten())
        return_state = np.reshape(return_state, [1, self.state_size])
        return return_state

    def get_reward(self):
        reward = {id_: self.get_reward_(id_) for id_ in self.intersection_id}
        return reward

    # def get_reward_(self, id_):
    #     '''
    #     every agent/intersection's reward
    #     '''
    #     state = self.intersection_info(id_)
    #     start_lane_waiting_vehicle_count = state['start_lane_waiting_vehicle_count']
    #     reward = -1 * np.sum(list(start_lane_waiting_vehicle_count.values()))
    #     return reward

    def get_reward_(self, id_):
        '''
        every agent/intersection's reward
        '''
        state = self.intersection_info(id_)
        start_lane_speed = state['start_lane_speed']
        reward = np.mean(list(start_lane_speed.values())) * 100
        return reward

    def get_score(self):
        score = {id_: self.get_score_(id_) for id_ in self.intersection_id}
        return score
    
    def get_score_(self, id_):
        state = self.intersection_info(id_)
        start_lane_waiting_vehicle_count = state['start_lane_waiting_vehicle_count']
        end_lane_waiting_vehicle_count = state['end_lane_waiting_vehicle_count']
        x = -1 * np.sum(list(start_lane_waiting_vehicle_count.values()) + list(end_lane_waiting_vehicle_count.values()))
        score = ( 1/(1 + np.exp(-1 * x)) )/self.num_step
        return score

import ray
from ray.rllib.env.multi_agent_env import MultiAgentEnv
from gym.spaces import Discrete, Box
class CityFlowEnvRay(MultiAgentEnv):
    '''
    multi inersection cityflow environment, for the Ray framework
    '''
    observation_space = Box(0.0*np.ones((13,)), 100*np.ones((13,)))
    action_space = Discrete(8) # num of agents

    def __init__(self, config):
        print("init")
        self.eng = cityflow.Engine(config["cityflow_config_file"], thread_num=config["thread_num"])
        # self.eng = config["eng"][0]
        self.num_step = config["num_step"]
        self.intersection_id = config["intersection_id"] # list, [intersection_id, ...]
        self.num_agents = len(self.intersection_id)
        self.state_size = None
        self.lane_phase_info = config["lane_phase_info"] # "intersection_1_1"
        self.congestion_thres = 30

        self.current_phase = {}
        self.current_phase_time = {}
        self.start_lane = {}
        self.end_lane = {}
        self.phase_list = {}
        self.phase_startLane_mapping = {}
        self.intersection_lane_mapping = {} #{id_:[lanes]}

        for id_ in self.intersection_id:
            self.start_lane[id_] = self.lane_phase_info[id_]['start_lane']
            self.end_lane[id_] = self.lane_phase_info[id_]['end_lane']
            self.phase_startLane_mapping[id_] = self.lane_phase_info[id_]["phase_startLane_mapping"]

            self.phase_list[id_] = self.lane_phase_info[id_]["phase"]
            self.current_phase[id_] = self.phase_list[id_][0]
            self.current_phase_time[id_] = 0
        self.get_state() # set self.state_size
        self.num_actions = len(self.phase_list[self.intersection_id[0]])

        # self.observation_space = Box(np.ones(0.0*(self.state_size,)), 20.0*np.ones((self.state_size)))
        # self.action_space = Discrete(self.num_actions) # num of agents
        
        
        self.count = 0
        self.done = False
        self.congestion = False
        self.reset()
        
    def reset(self):
        self.eng.reset()
        self.done = False
        self.congestion = False
        self.count = 0
        return {id_:np.zeros((self.state_size,)) for id_ in self.intersection_id}

    def step(self, action):
        '''
        action: {intersection_id: phase, ...}
        '''
        # print("action:", action)
        for id_, a in action.items():
            if self.current_phase[id_] == self.phase_list[id_][a]:
                self.current_phase_time[id_] += 1
            else:
                self.current_phase[id_] = self.phase_list[id_][a]
                self.current_phase_time[id_] = 1
            self.eng.set_tl_phase(id_, self.current_phase[id_]) # set phase of traffic light
        # print("after action:", action)
        self.eng.next_step()
        self.count += 1

        state = self.get_state()
        reward = self.get_reward()

        self.congestion = self.compute_congestion()
        self.done = {id_:False for id_ in self.intersection_id}
        self.done['__all__'] = False
        if self.count >= self.num_step:
            self.done = {id_:True for id_ in self.intersection_id}
            self.done['__all__'] = True
            # for id_ in self.intersection_id:
            #     reward[id_] = 0
        else:
            for id_ in self.intersection_id:
                if self.congestion[id_]:
                    self.done[id_] = True
                    reward[id_] = -1*50*(self.num_step-self.count) # if congestion, return a large penaty
            if all(list(self.congestion.values())) is True:
                self.done['__all__'] = True
            else:
                self.done['__all__'] = False
        
        return state, reward, self.done, {} 

    def compute_congestion(self):
        intersection_info = {}
        for id_ in self.intersection_id:
            intersection_info[id_] = self.intersection_info(id_)
        congestion = {id_:False for id_ in self.intersection_id}
        for id_ in self.intersection_id:
            if np.max(list(intersection_info[id_]["start_lane_waiting_vehicle_count"].values())) > self.congestion_thres:
                congestion[id_] = True
        return congestion

    def get_state(self):
        state =  {id_: self.get_state_(id_) for id_ in self.intersection_id}
        return state

    def get_state_(self, id_):
        state = self.intersection_info(id_)
        state_dict = state['start_lane_waiting_vehicle_count']
        sorted_keys = sorted(state_dict.keys())
        return_state = [state_dict[key] for key in sorted_keys] + [state['current_phase']]
        return self.preprocess_state(return_state)

    def intersection_info(self, id_):
        '''
        info of intersection 'id_'
        '''
        state = {}
        
        get_lane_vehicle_count = self.eng.get_lane_vehicle_count()
        get_lane_waiting_vehicle_count = self.eng.get_lane_waiting_vehicle_count()
        get_lane_vehicles = self.eng.get_lane_vehicles()
        get_vehicle_speed = self.eng.get_vehicle_speed()

        # print(self.intersection_id)
        # print(id_)
        # print("start lane", self.start_lane)
        # print("get_lane_vehicle_count key length:", get_lane_vehicle_count)
        # print("engine:", self.eng)
        # print("get_lane_waiting_vehicle_count:", get_lane_waiting_vehicle_count)
        # print("get_lane_vehicles:", get_lane_vehicles)
        # print("vehicle_speed:", vehicle_speed)

        state['start_lane_vehicle_count'] = {lane: get_lane_vehicle_count[lane] for lane in self.start_lane[id_]}
        state['end_lane_vehicle_count'] = {lane: get_lane_vehicle_count[lane] for lane in self.end_lane[id_]}
        # state['start_lane_vehicle_count'] = {}
        # state['end_lane_vehicle_count'] = {}

        state['start_lane_waiting_vehicle_count'] = {lane: get_lane_waiting_vehicle_count[lane] for lane in self.start_lane[id_]}
        state['end_lane_waiting_vehicle_count'] = {lane: get_lane_waiting_vehicle_count[lane] for lane in self.end_lane[id_]}
        
        state['start_lane_vehicles'] = {lane: get_lane_vehicles[lane] for lane in self.start_lane[id_]}
        state['end_lane_vehicles'] = {lane: get_lane_vehicles[lane] for lane in self.end_lane[id_]}
        
        state['start_lane_speed'] = {lane: np.sum(list(map(lambda v:get_vehicle_speed[v], get_lane_vehicles[lane])))/(len(get_lane_vehicles[lane])+1e-5) for lane in self.start_lane[id_]} # compute start lane mean speed
        state['end_lane_speed'] = {lane: np.sum(list(map(lambda v:get_vehicle_speed[v], get_lane_vehicles[lane])))/(len(get_lane_vehicles[lane])+1e-5) for lane in self.end_lane[id_]} # compute end lane mean speed
        
        state['current_phase'] = self.current_phase[id_]
        state['current_phase_time'] = self.current_phase_time[id_]

        return state


    def preprocess_state(self, state):
        return_state = np.array(state)
        if self.state_size is None:
            self.state_size = len(return_state.flatten())
        return_state = np.reshape(np.array(return_state), [1, self.state_size]).flatten()
        return return_state

    def get_reward(self):
        reward = {id_: self.get_reward_(id_) for id_ in self.intersection_id}
        mean_global_sum = np.mean(list(reward.values()))
        return reward
        # reward = {id_:mean_global_sum for id_ in self.intersection_id}
        # return reward

    def get_reward_(self, id_):
        '''
        every agent/intersection's reward
        '''
        state = self.intersection_info(id_)
        temp = state['start_lane_waiting_vehicle_count']
        reward = -1 * np.mean(list(temp.values())) 
        return reward

    def get_score(self):
        score = {id_: self.get_score_(id_) for id_ in self.intersection_id}
        return score
    
    def get_score_(self, id_):
        state = self.intersection_info(id_)
        start_lane_waiting_vehicle_count = state['start_lane_waiting_vehicle_count']
        end_lane_waiting_vehicle_count = state['end_lane_waiting_vehicle_count']
        x = -1 * np.sum(list(start_lane_waiting_vehicle_count.values()) + list(end_lane_waiting_vehicle_count.values()))
        score = ( 1/(1 + np.exp(-1 * x)) )/self.num_step
        return score