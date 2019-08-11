import math

import cityflow
import gym
import numpy as np
from gym.spaces import Discrete, Box
from ray.rllib.env.multi_agent_env import MultiAgentEnv


class CityflowGymEnv(gym.Env):
    # metadata = {'render.modes': ['human']}
    observation_space = Box(-1 * np.ones(9), 100 * np.ones(9))
    action_space = Discrete(8)

    def __init__(self, config):
        self.eng = cityflow.Engine(config['cityflow_config_file'], thread_num=config['thread_num'])

        self.num_step = config['num_step']
        self.lane_phase_info = config['lane_phase_info']  # "intersection_1_1"

        self.intersection_id = list(self.lane_phase_info.keys())[0]
        self.start_lane = self.lane_phase_info[self.intersection_id]['start_lane']
        self.phase_list = self.lane_phase_info[self.intersection_id]["phase"]
        self.phase_startLane_mapping = self.lane_phase_info[self.intersection_id]["phase_startLane_mapping"]

        self.current_phase = self.phase_list[0]
        self.current_phase_time = 0

        self.phase_log = []

        self.step_count = 1

        self.congestion_thres = 30

    def step(self, action):
        state = self._get_state()
        reward = self._get_reward()

        if self.current_phase == action:
            self.current_phase_time += 1
        else:
            self.current_phase = action
            self.current_phase_time = 1

        self.eng.set_tl_phase(self.intersection_id, self.current_phase)  # set phase of traffic light
        self.eng.next_step()
        self.step_count += 1
        congestion = self._compute_congestion()
        self.phase_log.append(self.current_phase)
        done = False
        if self.step_count > self.num_step:
            done = True
        else:
            if congestion:
                done = True
                reward = -1 * self.congestion_thres * (self.num_step - self.step_count)
        return state, reward, done, {}  # return next_state and reward, whether done and info

    def _compute_congestion(self):
        if np.max(list(self.eng.get_lane_waiting_vehicle_count().values())) > self.congestion_thres:
            return True
        return False

    def _get_state(self):
        d = self.eng.get_lane_waiting_vehicle_count()
        start_lane_waiting_count = {lane: d[lane] for lane in self.start_lane}
        sorted_keys = sorted(start_lane_waiting_count.keys())
        return_state = np.array([start_lane_waiting_count[key] for key in sorted_keys] + [self.current_phase])

        return return_state

    def _get_reward(self):
        reward = -1 * np.mean(self._get_state()[:8])

        return reward

    def _get_score(self):
        lane_waiting_vehicle_count = self.eng.get_lane_waiting_vehicle_count()
        reward = max(list(lane_waiting_vehicle_count.values()))
        metric = 1 / ((1 + math.exp(-1 * reward)) * self.num_step)
        return reward

    def reset(self):
        self.eng.reset()
        self.step_count = 0
        return self._get_state()

    def render(self, mode='human', close=False):
        pass


class CityFlowEnvRay(MultiAgentEnv):
    '''
    multi inersection cityflow environment, for the Ray framework
    '''
    observation_space = Box(0.0 * np.ones((13,)), 100 * np.ones((13,)))
    action_space = Discrete(8)  # num of agents

    def __init__(self, config):
        print("init")
        self.eng = cityflow.Engine(config["cityflow_config_file"], thread_num=config["thread_num"])
        # self.eng = config["eng"][0]
        self.num_step = config["num_step"]
        self.intersection_id = config["intersection_id"]  # list, [intersection_id, ...]
        self.num_agents = len(self.intersection_id)
        self.state_size = None
        self.lane_phase_info = config["lane_phase_info"]  # "intersection_1_1"
        self.congestion_thres = 30

        self.current_phase = {}
        self.current_phase_time = {}
        self.start_lane = {}
        self.end_lane = {}
        self.phase_list = {}
        self.phase_startLane_mapping = {}
        self.intersection_lane_mapping = {}  # {id_:[lanes]}

        for id_ in self.intersection_id:
            self.start_lane[id_] = self.lane_phase_info[id_]['start_lane']
            self.end_lane[id_] = self.lane_phase_info[id_]['end_lane']
            self.phase_startLane_mapping[id_] = self.lane_phase_info[id_]["phase_startLane_mapping"]

            self.phase_list[id_] = self.lane_phase_info[id_]["phase"]
            self.current_phase[id_] = self.phase_list[id_][0]
            self.current_phase_time[id_] = 0
        self.get_state()  # set self.state_size
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
        return {id_: np.zeros((self.state_size,)) for id_ in self.intersection_id}

    def step(self, action):
        """
        action: {intersection_id: phase, ...}
        """
        # print("action:", action)
        for id_, a in action.items():
            if self.current_phase[id_] == self.phase_list[id_][a]:
                self.current_phase_time[id_] += 1
            else:
                self.current_phase[id_] = self.phase_list[id_][a]
                self.current_phase_time[id_] = 1
            self.eng.set_tl_phase(id_, self.current_phase[id_])  # set phase of traffic light
        # print("after action:", action)
        self.eng.next_step()
        self.count += 1

        state = self.get_state()
        reward = self.get_reward()

        self.congestion = self.compute_congestion()
        self.done = {id_: False for id_ in self.intersection_id}
        self.done['__all__'] = False
        if self.count >= self.num_step:
            self.done = {id_: True for id_ in self.intersection_id}
            self.done['__all__'] = True
            # for id_ in self.intersection_id:
            #     reward[id_] = 0
        else:
            for id_ in self.intersection_id:
                if self.congestion[id_]:
                    self.done[id_] = True
                    reward[id_] = -1 * 50 * (self.num_step - self.count)  # if congestion, return a large penaty
            if all(list(self.congestion.values())) is True:
                self.done['__all__'] = True
            else:
                self.done['__all__'] = False

        return state, reward, self.done, {}

    def compute_congestion(self):
        intersection_info = {}
        for id_ in self.intersection_id:
            intersection_info[id_] = self.intersection_info(id_)
        congestion = {id_: False for id_ in self.intersection_id}
        for id_ in self.intersection_id:
            if np.max(
                    list(intersection_info[id_]["start_lane_waiting_vehicle_count"].values())) > self.congestion_thres:
                congestion[id_] = True
        return congestion

    def get_state(self):
        state = {id_: self.get_state_(id_) for id_ in self.intersection_id}
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

        state['start_lane_waiting_vehicle_count'] = {lane: get_lane_waiting_vehicle_count[lane] for lane in
                                                     self.start_lane[id_]}
        state['end_lane_waiting_vehicle_count'] = {lane: get_lane_waiting_vehicle_count[lane] for lane in
                                                   self.end_lane[id_]}

        state['start_lane_vehicles'] = {lane: get_lane_vehicles[lane] for lane in self.start_lane[id_]}
        state['end_lane_vehicles'] = {lane: get_lane_vehicles[lane] for lane in self.end_lane[id_]}

        state['start_lane_speed'] = {
            lane: np.sum(list(map(lambda v: get_vehicle_speed[v], get_lane_vehicles[lane]))) / (
                        len(get_lane_vehicles[lane]) + 1e-5) for lane in
            self.start_lane[id_]}  # compute start lane mean speed
        state['end_lane_speed'] = {lane: np.sum(list(map(lambda v: get_vehicle_speed[v], get_lane_vehicles[lane]))) / (
                    len(get_lane_vehicles[lane]) + 1e-5) for lane in self.end_lane[id_]}  # compute end lane mean speed

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
        score = (1 / (1 + np.exp(-1 * x))) / self.num_step
        return score
