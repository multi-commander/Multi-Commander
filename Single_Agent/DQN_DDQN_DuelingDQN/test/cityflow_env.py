import cityflow
import pandas as pd
import os
import math
import numpy as np



class CityFlowEnv:
    """
    Simulator Environment with CityFlow
    """

    def __init__(self, config):

        self.config = config

        self.eng = cityflow.Engine(config['cityflow_config_path'], thread_num=1)

        self.num_step = config['num_step']
        self.lane_phase_info = config['lane_phase_info']  # "intersection_1_1"
        self.state_size = len(config['lane_phase_info'][config["intersection_id"]]['start_lane']) + 1 + 4 ## state size

        self.intersection_id = list(self.lane_phase_info.keys())[0]
        self.start_lane = self.lane_phase_info[self.intersection_id]['start_lane']
        self.phase_list = self.lane_phase_info[self.intersection_id]["phase"]
        self.phase_startLane_mapping = self.lane_phase_info[self.intersection_id]["phase_startLane_mapping"]

        self.current_phase = self.phase_list[0]
        self.current_phase_time = 0
        self.yellow_time = 5
        self.state_store_i = 0
        self.time_span = 50
        self.state_time_span = 5
        self.num_span_1 = 0
        self.num_span_2 = 0
        self.state_size_single = len(self.start_lane)
        self.count = np.zeros([8, self.time_span])
        self.accum_s = np.zeros([self.state_size_single, self.state_time_span])
        self.phase_log = []

    def reset(self):
        self.eng.reset()
        self.phase_log = []

    def step(self, next_phase):
        if self.current_phase == next_phase:
            self.current_phase_time += 1
        else:
            self.current_phase = next_phase
            self.current_phase_time = 1

        self.eng.set_tl_phase(self.intersection_id, self.current_phase)
        self.eng.next_step()
        self.phase_log.append(self.current_phase)

    def get_state(self):

        state = {'lane_vehicle_count': self.eng.get_lane_vehicle_count(),
                 'start_lane_vehicle_count': {lane: self.eng.get_lane_vehicle_count()[lane] for lane in
                                              self.start_lane},
                 'lane_waiting_vehicle_count': self.eng.get_lane_waiting_vehicle_count(),
                 'lane_vehicles': self.eng.get_lane_vehicles(),
                 'vehicle_speed': self.eng.get_vehicle_speed(),
                 'vehicle_distance': self.eng.get_vehicle_distance(),
                 'current_time': self.eng.get_current_time(),
                 'current_phase': self.current_phase,
                 'current_phase_time': self.current_phase_time,
                 'dimension': self.dimension(),
                }

        return_state = np.array(list(state["start_lane_vehicle_count"].values()) + [state["current_phase"]] + state['dimension'])
        return_state = np.reshape(return_state, [-1, self.state_size])
        return return_state
    
    # def get_state_size():
    #     pass

    
    def span_count(self):
        self.count[:,self.num_span_1]=0
        self.count[self.current_phase-1, self.num_span_1] = 1
        self.num_span_1 = self.num_span_1 + 1
        if self.num_span_1 is self.time_span:
            self.num_span_1 = 0
        self.countsum = self.count.sum(axis=1)
        return self.countsum.reshape(8, 1)
    
    # def span_state(self):
    #     self.state = self.get_state()
    #     self.state = np.array(list(self.state['start_lane_vehicle_count'].values()))
    #     self.state = np.reshape(self.state, [self.state_size_single, 1])
    #     for i in range(self.state_size_single):
    #         self.accum_s[i, self.num_span_2] = self.state[i]
    #     self.num_span_2 = self.num_span_2 + 1
    #     if self.num_span_2 is self.state_time_span:
    #         self.num_span_2 = 0
    #     return self.accum_s.reshape(1, self.state_size)
    
    def dimension(self):
        var = 0
        onemoment = 0
        twomoment = 0
        threemoment = 0
        State = {'lane_vehicle_count': self.eng.get_lane_vehicle_count(),
                 'start_lane_vehicle_count': {lane: self.eng.get_lane_vehicle_count()[lane] for lane in
                                              self.start_lane},
                 'lane_waiting_vehicle_count': self.eng.get_lane_waiting_vehicle_count(),
                 'lane_vehicles': self.eng.get_lane_vehicles(),
                 'vehicle_speed': self.eng.get_vehicle_speed(),
                 'vehicle_distance': self.eng.get_vehicle_distance(),
                 'current_time': self.eng.get_current_time(),
                 'current_phase': self.current_phase,
                 'current_phase_time': self.current_phase_time,
                }

        State1= np.array(list(State['start_lane_vehicle_count'].values()))
        State1= np.reshape(State1, [self.state_size_single, 1])
        for i in range(self.state_size_single):
            self.accum_s[i, self.num_span_2] = State1[i]
        self.num_span_2 = self.num_span_2 + 1
        if self.num_span_2 is self.state_time_span:
            self.num_span_2 = 0

        for x in range(self.state_size_single):
            var1 = np.var(self.accum_s[x,:])
            var=var+var1
            for w in range(self.state_time_span):
                onemoment = onemoment + self.accum_s[x,w]
                twomoment = twomoment + self.accum_s[x, w]**2
                threemoment = threemoment + self.accum_s[x, w]**3
        var = var/(self.state_size_single*self.state_time_span)
        onemoment = onemoment / (self.state_size_single * self.state_time_span)
        twomoment = twomoment / (self.state_size_single * self.state_time_span)
        threemoment = threemoment / (self.state_size_single * self.state_time_span)

        return [var,onemoment,twomoment,threemoment]
    

    def get_reward(self):
        # a sample reward function which calculates the total of waiting vehicles
        self.state = self.get_state()
        lane_waiting_vehicle_count = self.eng.get_lane_waiting_vehicle_count()
        speed = sum(list(self.eng.get_vehicle_speed().values()))
        number = sum(list(self.eng.get_lane_vehicle_count().values()))
        reward1 = -1 * sum(list(lane_waiting_vehicle_count.values()))
        reward2 = reward1/(1+math.exp(reward1))
        reward3 = speed/(number+2)#10
        reward4 = sum(list(self.span_count()[[self.state['current_phase']-1]]))
        return reward2+reward3*100-reward4*10
    
    def get_score(self):
        number = self.eng.get_lane_waiting_vehicle_count()
        reward = sum(list(number.values()))
        number2 = 1/((1+math.exp(-reward))*3600)
        return number2

    def log(self):
        # self.eng.print_log(self.config['replay_data_path'] + "/replay_roadnet.json",
        #                   self.config['replay_data_path'] + "/replay_flow.json")
        df = pd.DataFrame({self.intersection_id: self.phase_log[:self.num_step]})
        if not os.path.exists(self.config['data']):
            os.makedirs(self.config["data"])
        df.to_csv(os.path.join(self.config['data'], 'signal_plan_template.txt'), index=None)
