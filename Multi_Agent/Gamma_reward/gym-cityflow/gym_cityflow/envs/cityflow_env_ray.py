import cityflow
import pandas as pd
import os
import json
import math
import numpy as np
import itertools


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
        
        self.eng = cityflow.Engine(config["cityflow_config_file"], thread_num=config["thread_num"])
        # self.eng = config["eng"][0]
        self.num_step = config["num_step"]
        self.intersection_id = config["intersection_id"] # list, [intersection_id, ...]
        self.num_agents = len(self.intersection_id)
        self.state_size = None
        self.lane_phase_info = config["lane_phase_info"] # "intersection_1_1"

        self.current_phase = {}
        self.current_phase_time = {}
        self.start_lane = {}
        self.end_lane = {}
        self.phase_list = {}
        self.phase_startLane_mapping = {}
        self.intersection_lane_mapping = {} #{id_:[lanes]}

        #### dw
        self.gamma = 0.5
        self.R_buffer = [{id_: 0
                        for id_ in self.intersection_id},
                        {id_: 0
                        for id_ in self.intersection_id}]
        #### dw

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

        # print("######################")
        # print("after action:", action)
        # print("######################")
        
        self.eng.next_step()
        self.count += 1

        #if self.count >= self.num_step or self.congestion:
        if self.count >= self.num_step:
            self.done = True
        state = self.get_state()
        reward = self.get_reward()

        #### dw #####

        if self.count >1:
            if self.count % 2 ==0:
                for id in reward:
                    self.R_buffer[0][id] = reward[id] 
            else:
                for id in reward:
                    self.R_buffer[1][id] = reward[id] 

        print("##################")
        print(self.R_buffer)
        print("##################")

        #### dw #####


        self.congestion = self.compute_congestion()
        self.done = {id_:False for id_ in self.intersection_id}
        self.done['__all__'] = False
        if self.count >= self.num_step:
            self.done = {id_:True for id_ in self.intersection_id}
            self.done['__all__'] = True
        # else:
        #     for id_ in self.intersection_id:
        #         if self.congestion[id_]:
        #             self.done[id_] = True
        #     if any(list(self.congestion.values())) is False:
        #         self.done['__all__'] = True
        #     else:
        #         self.done['__all__'] = False
        
        return state, reward, self.done, {} 

    def compute_congestion(self):
        intersection_info = {}
        for id_ in self.intersection_id:
            intersection_info[id_] = self.intersection_info(id_)
        congestion = {id_:False for id_ in self.intersection_id}
        for id_ in self.intersection_id:
            if np.max(list(intersection_info[id_]["start_lane_waiting_vehicle_count"].values())) > 20:
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
        
        state['start_lane_speed'] = {lane: np.sum(list(map(lambda vehicle:get_vehicle_speed[vehicle], get_lane_vehicles[lane]))) / (get_lane_vehicle_count[lane]+1e-5) for lane in self.start_lane[id_]} # compute start lane mean speed
        state['end_lane_speed'] = {lane: np.sum(list(map(lambda vehicle:get_vehicle_speed[vehicle], get_lane_vehicles[lane]))) / (get_lane_vehicle_count[lane]+1e-5) for lane in self.end_lane[id_]} # compute end lane mean speed
        
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
        
        #every agent/intersection's reward
        
        state = self.intersection_info(id_)
        r = max(list(state['start_lane_vehicle_count'].values()))
    
        R = r

        flag = False
        

        #R_start = list(self.R_buffer.values())[0]
        #R_end = list(self.R_buffer.values())[5]
        
        pre_id = ""
        count = 0

        
        for i in self.R_buffer[0].keys():
            count +=1

            if flag == True:
                if self.count % 2 ==0:
                    R -= self.gamma*(self.R_buffer[0][i] - self.R_buffer[1][i])
                    break
                else:
                    R -= self.gamma*(self.R_buffer[1][i] - self.R_buffer[0][i])
                    break
            if i == id_:
                flag = True
                if count == 1:
                    #R -= self.gamma * R_end
                    continue
                if self.count % 2 ==0:
                    R -= self.gamma*(self.R_buffer[0][pre_id] - self.R_buffer[1][pre_id])
                else:
                    R -= self.gamma*(self.R_buffer[1][pre_id] - self.R_buffer[0][pre_id])
                # R -= self.gamma * (self.R_buffer[1][pre_id] - self.R_buffer[0][i])
                if count == 6:
                    #R -= self.gamma * R_start
                    break
            else:
                pre_id = i
       
        R = -R
        '''
        temp = state['start_lane_speed']
        reward = np.mean(list(temp.values())) 
        '''


        return R


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