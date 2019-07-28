import cityflow
import pandas as pd
import os


class SingleIntersectionEnv():
    '''
    Simulator Environment with CityFlow
    '''
    def __init__(self, config_path):
        self.eng = cityflow.Engine(config_path, threadNum = 1)
        # self.eng.load_roadnet(config['roadnet'])
        # self.eng.load_flow(config['flow'])
        # self.config = config
        # self.num_step = config['num_step']
        # self.lane_phase_info = config['lane_phase_info'] # "intersection_1_1"

        self.intersection_id = list(self.lane_phase_info.keys())[0]
        # self.start_lane = self.lane_phase_info[self.intersection_id]['start_lane']
        # self.phase_list = self.lane_phase_info[self.intersection_id]["phase"]
        # self.phase_startLane_mapping = self.lane_phase_info[self.intersection_id]["phase_startLane_mapping"]

        self.current_phase = self.phase_list[0]
        self.current_phase_time = 0
        # self.yellow_time = 5

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
        state = {}
        state['lane_vehicle_count'] = self.eng.get_lane_vehicle_count()  # {lane_id: lane_count, ...}
        state['start_lane_vehicle_count'] = {lane: self.eng.get_lane_vehicle_count()[lane] for lane in self.start_lane}
        state['lane_waiting_vehicle_count'] = self.eng.get_lane_waiting_vehicle_count()  # {lane_id: lane_waiting_count, ...}
        state['lane_vehicles'] = self.eng.get_lane_vehicles()  # {lane_id: [vehicle1_id, vehicle2_id, ...], ...}
        state['vehicle_speed'] = self.eng.get_vehicle_speed()  # {vehicle_id: vehicle_speed, ...}
        state['vehicle_distance'] = self.eng.get_vehicle_distance() # {vehicle_id: distance, ...}
        state['current_time'] = self.eng.get_current_time()
        state['current_phase'] = self.current_phase
        state['current_phase_time'] = self.current_phase_time

        return state

    def get_reward(self):
        # a sample reward function which calculates the total of waiting vehicles
        lane_waiting_vehicle_count = self.eng.get_lane_waiting_vehicle_count()
        reward = -1 * sum(list(lane_waiting_vehicle_count.values()))
        '''
        # 以平均速度作为奖励
        speeds = self.eng.get_vehicle_speed()
        avg_speed = sum(list(speeds.values()))/len(speeds)
        reward = avg_speed
        '''
        return reward

    def log(self):
        #self.eng.print_log("/replay_roadnet.json",
        #                   "/replay_flow.json")
        #df = pd.DataFrame({self.intersection_id: self.phase_log[:self.num_step]})
        #if not os.path.exists(self.config['data']):
        #    os.makedirs(self.config["data"])
        #df.to_csv(os.path.join(self.config['data'], 'signal_plan_template.txt'), index=None)
