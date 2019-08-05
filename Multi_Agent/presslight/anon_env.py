import pickle
import numpy as np
import json
import sys
import pandas as pd
import os
from utility import get_cityflow_config
import time
import threading
from multiprocessing import Process, Pool
from script import get_traffic_volume
from copy import deepcopy
import cityflow


# engine = cdll.LoadLibrary("./engine.cpython-36m-x86_64-linux-gnu.so")

class Intersection:
    DIC_PHASE_MAP = {
        0: 1,
        1: 2,
        2: 3,
        3: 4,
        -1: 0
    }

    def __init__(self, inter_id, dic_traffic_env_conf, eng, light_id_dict):
        self.inter_id = inter_id
        self.inter_name = "intersection_{0}_{1}".format(inter_id[0], inter_id[1])

        self.eng = eng

        self.fast_compute = dic_traffic_env_conf['FAST_COMPUTE']

        # =====  intersection settings =====
        self.list_approachs = ["W", "E", "N", "S"]
        self.dic_approach_to_node = {"W": 0, "E": 2, "S": 1, "N": 3}
        self.dic_entering_approach_to_edge = {"W": "road_{0}_{1}_0".format(inter_id[0] - 1, inter_id[1])}
        self.dic_entering_approach_to_edge.update({"E": "road_{0}_{1}_2".format(inter_id[0] + 1, inter_id[1])})
        self.dic_entering_approach_to_edge.update({"N": "road_{0}_{1}_3".format(inter_id[0], inter_id[1] + 1)})
        self.dic_entering_approach_to_edge.update({"S": "road_{0}_{1}_1".format(inter_id[0], inter_id[1] - 1)})

        self.dic_exiting_approach_to_edge = {
            approach: "road_{0}_{1}_{2}".format(inter_id[0], inter_id[1], self.dic_approach_to_node[approach]) for
            approach in self.list_approachs}

        # grid settings
        self.length_lane = 300
        self.length_terminal = 50
        self.length_grid = 5
        self.num_grid = int(self.length_lane // self.length_grid)

        self.list_phases = dic_traffic_env_conf["PHASE"][dic_traffic_env_conf['SIMULATOR_TYPE']]

        # generate all lanes
        self.list_entering_lanes = []
        for approach in self.list_approachs:
            self.list_entering_lanes += [self.dic_entering_approach_to_edge[approach] + '_' + str(i) for i in
                                         range(sum(list(dic_traffic_env_conf["LANE_NUM"].values())))]
        self.list_exiting_lanes = []
        for approach in self.list_approachs:
            self.list_exiting_lanes += [self.dic_exiting_approach_to_edge[approach] + '_' + str(i) for i in
                                        range(sum(list(dic_traffic_env_conf["LANE_NUM"].values())))]

        self.list_lanes = self.list_entering_lanes + self.list_exiting_lanes

        self.adjacency_row = light_id_dict['adjacency_row']
        self.neighbor_ENWS = light_id_dict['neighbor_ENWS']

        # previous & current
        self.dic_lane_vehicle_previous_step = {}
        self.dic_lane_waiting_vehicle_count_previous_step = {}

        self.dic_vehicle_speed_previous_step = {}
        self.dic_vehicle_distance_previous_step = {}

        self.dic_lane_vehicle_current_step = {}
        self.dic_lane_waiting_vehicle_count_current_step = {}
        self.dic_vehicle_speed_current_step = {}
        self.dic_vehicle_distance_current_step = {}

        self.list_lane_vehicle_previous_step = []
        self.list_lane_vehicle_current_step = []

        # -1: all yellow, -2: all red, -3: none
        self.all_yellow_phase_index = -1
        self.all_red_phase_index = -2

        self.current_phase_index = 1
        self.previous_phase_index = 1
        self.eng.set_tl_phase(self.inter_name, self.current_phase_index)

        self.next_phase_to_set_index = None
        self.current_phase_duration = -1
        self.all_red_flag = False
        self.all_yellow_flag = False
        self.flicker = 0

        self.dic_vehicle_min_speed = {}  # this second
        self.dic_vehicle_arrive_leave_time = dict()  # cumulative

        self.dic_feature = {}  # this second
        self.dic_feature_previous_step = {}  # this second

    # set
    def set_signal(self, action, action_pattern, yellow_time, all_red_time):
        if self.all_yellow_flag:
            # in yellow phase
            self.flicker = 0
            if self.current_phase_duration >= yellow_time:  # yellow time reached
                self.current_phase_index = self.next_phase_to_set_index
                self.eng.set_tl_phase(self.inter_name, self.current_phase_index)  # if multi_phase, need more adjustment
                self.all_yellow_flag = False
            else:
                pass
        else:
            # determine phase
            if action_pattern == "switch":  # switch by order
                if action == 0:  # keep the phase
                    self.next_phase_to_set_index = self.current_phase_index
                elif action == 1:  # change to the next phase
                    self.next_phase_to_set_index = (self.current_phase_index + 1) % len(
                        self.list_phases)  # if multi_phase, need more adjustment
                else:
                    sys.exit("action not recognized\n action must be 0 or 1")

            elif action_pattern == "set":  # set to certain phase
                self.next_phase_to_set_index = self.DIC_PHASE_MAP[action]  # if multi_phase, need more adjustment

            # set phase
            if self.current_phase_index == self.next_phase_to_set_index:  # the light phase keeps unchanged
                pass
            else:  # the light phase needs to change
                # change to yellow first, and activate the counter and flag
                self.eng.set_tl_phase(self.inter_name, 0)
                self.current_phase_index = self.all_yellow_phase_index
                self.all_yellow_flag = True
                self.flicker = 1

    # update inner measurements
    def update_previous_measurements(self):
        self.previous_phase_index = self.current_phase_index

        self.dic_lane_vehicle_previous_step = self.dic_lane_vehicle_current_step
        self.dic_lane_waiting_vehicle_count_previous_step = self.dic_lane_waiting_vehicle_count_current_step
        self.dic_vehicle_speed_previous_step = self.dic_vehicle_speed_current_step
        self.dic_vehicle_distance_previous_step = self.dic_vehicle_distance_current_step

    def update_current_measurements_map(self, simulator_state, path_to_log, test_flag):
        ## need change, debug in seeing format
        def _change_lane_vehicle_dic_to_list(dic_lane_vehicle):
            list_lane_vehicle = []

            for value in dic_lane_vehicle.values():
                list_lane_vehicle.extend(value)

            return list_lane_vehicle

        if self.current_phase_index == self.previous_phase_index:
            self.current_phase_duration += 1
        else:
            self.current_phase_duration = 1

        self.dic_lane_vehicle_current_step = {}
        self.dic_lane_waiting_vehicle_count_current_step = {}
        for lane in self.list_entering_lanes:
            self.dic_lane_vehicle_current_step[lane] = simulator_state["get_lane_vehicles"][lane]
            self.dic_lane_waiting_vehicle_count_current_step[lane] = simulator_state["get_lane_waiting_vehicle_count"][
                lane]

        for lane in self.list_exiting_lanes:
            self.dic_lane_waiting_vehicle_count_current_step[lane] = simulator_state["get_lane_waiting_vehicle_count"][
                lane]

        self.dic_vehicle_speed_current_step = simulator_state['get_vehicle_speed']
        self.dic_vehicle_distance_current_step = simulator_state['get_vehicle_distance']

        # get vehicle list
        self.list_lane_vehicle_current_step = _change_lane_vehicle_dic_to_list(self.dic_lane_vehicle_current_step)
        self.list_lane_vehicle_previous_step = _change_lane_vehicle_dic_to_list(self.dic_lane_vehicle_previous_step)

        list_vehicle_new_arrive = list(
            set(self.list_lane_vehicle_current_step) - set(self.list_lane_vehicle_previous_step))
        list_vehicle_new_left = list(
            set(self.list_lane_vehicle_previous_step) - set(self.list_lane_vehicle_current_step))
        list_vehicle_new_left_entering_lane_by_lane = self._update_leave_entering_approach_vehicle()
        list_vehicle_new_left_entering_lane = []
        for l in list_vehicle_new_left_entering_lane_by_lane:
            list_vehicle_new_left_entering_lane += l

        # update vehicle arrive and left time
        self._update_arrive_time(list_vehicle_new_arrive)
        self._update_left_time(list_vehicle_new_left_entering_lane, path_to_log)

        # update feature
        self._update_feature_map(simulator_state, test_flag)

    def update_current_measurements(self, path_to_log):
        ## need change, debug in seeing format
        def _change_lane_vehicle_dic_to_list(dic_lane_vehicle):
            list_lane_vehicle = []

            for value in dic_lane_vehicle.values():
                list_lane_vehicle.extend(value)

            return list_lane_vehicle

        if self.current_phase_index == self.previous_phase_index:
            self.current_phase_duration += 1
        else:
            self.current_phase_duration = 1

        self.dic_lane_vehicle_current_step = []  # = self.eng.get_lane_vehicles()
        # not implement
        flow_tmp = self.eng.get_lane_vehicles()
        self.dic_lane_vehicle_current_step = {key: None for key in self.list_entering_lanes}
        for lane in self.list_entering_lanes:
            self.dic_lane_vehicle_current_step[lane] = flow_tmp[lane]

        self.dic_lane_waiting_vehicle_count_current_step = self.eng.get_lane_waiting_vehicle_count()
        self.dic_vehicle_speed_current_step = self.eng.get_vehicle_speed()
        self.dic_vehicle_distance_current_step = self.eng.get_vehicle_distance()

        # get vehicle list
        self.list_lane_vehicle_current_step = _change_lane_vehicle_dic_to_list(self.dic_lane_vehicle_current_step)
        self.list_lane_vehicle_previous_step = _change_lane_vehicle_dic_to_list(self.dic_lane_vehicle_previous_step)

        list_vehicle_new_arrive = list(
            set(self.list_lane_vehicle_current_step) - set(self.list_lane_vehicle_previous_step))
        list_vehicle_new_left = list(
            set(self.list_lane_vehicle_previous_step) - set(self.list_lane_vehicle_current_step))
        list_vehicle_new_left_entering_lane_by_lane = self._update_leave_entering_approach_vehicle()
        list_vehicle_new_left_entering_lane = []
        for l in list_vehicle_new_left_entering_lane_by_lane:
            list_vehicle_new_left_entering_lane += l

        # update vehicle arrive and left time
        self._update_arrive_time(list_vehicle_new_arrive)
        self._update_left_time(list_vehicle_new_left_entering_lane, path_to_log)

        # update feature
        self._update_feature()

    def _update_leave_entering_approach_vehicle(self):

        list_entering_lane_vehicle_left = []

        # update vehicles leaving entering lane
        if not self.dic_lane_vehicle_previous_step:
            for lane in self.list_entering_lanes:
                list_entering_lane_vehicle_left.append([])
        else:
            last_step_vehicle_id_list = []
            current_step_vehilce_id_list = []
            for lane in self.list_entering_lanes:
                last_step_vehicle_id_list.extend(self.dic_lane_vehicle_previous_step[lane])
                current_step_vehilce_id_list.extend(self.dic_lane_vehicle_current_step[lane])

            list_entering_lane_vehicle_left.append(
                list(set(last_step_vehicle_id_list) - set(current_step_vehilce_id_list))
            )

        return list_entering_lane_vehicle_left

    def _update_arrive_time(self, list_vehicle_arrive):

        ts = self.get_current_time()
        # get dic vehicle enter leave time
        for vehicle in list_vehicle_arrive:
            if vehicle not in self.dic_vehicle_arrive_leave_time:
                self.dic_vehicle_arrive_leave_time[vehicle] = \
                    {"enter_time": ts, "leave_time": np.nan}
            else:
                # print("vehicle: %s already exists in entering lane!"%vehicle)
                # sys.exit(-1)
                pass

    def _update_left_time(self, list_vehicle_left, path_to_log):

        ts = self.get_current_time()
        # update the time for vehicle to leave entering lane
        for vehicle in list_vehicle_left:
            try:
                self.dic_vehicle_arrive_leave_time[vehicle]["leave_time"] = ts
                ## TODO log one vehicle and then pop
                self.log_one_vehicle(vehicle, path_to_log)
                self.dic_vehicle_arrive_leave_time.pop(vehicle)
            except KeyError:
                print("vehicle not recorded when entering")
                sys.exit(-1)

    def log_one_vehicle(self, vehicle, path_to_log):
        inter = str(self.inter_id[0]) + '_' + str(self.inter_id[1])
        path_to_log_file = os.path.join(path_to_log, "vehicle_inter_{0}.csv".format(inter))
        df = [vehicle, self.dic_vehicle_arrive_leave_time[vehicle]["enter_time"],
              self.dic_vehicle_arrive_leave_time[vehicle]["leave_time"]]
        df = pd.DataFrame(df)
        df = df.transpose()
        df.to_csv(path_to_log_file, mode='a', header=False, index=False)

    def _update_feature(self):

        dic_feature = dict()

        dic_feature["cur_phase"] = [self.current_phase_index]
        dic_feature["time_this_phase"] = [self.current_phase_duration]
        dic_feature["vehicle_position_img"] = None  # self._get_lane_vehicle_position(self.list_entering_lanes)
        dic_feature["vehicle_speed_img"] = None  # self._get_lane_vehicle_speed(self.list_entering_lanes)
        dic_feature["vehicle_acceleration_img"] = None
        dic_feature[
            "vehicle_waiting_time_img"] = None  # self._get_lane_vehicle_accumulated_waiting_time(self.list_entering_lanes)

        dic_feature["lane_num_vehicle"] = self._get_lane_num_vehicle(self.list_entering_lanes)

        dic_feature["lane_num_vehicle_downstream"] = self._get_lane_num_vehicle(self.list_exiting_lanes)

        dic_feature["coming_vehicle"] = self._get_coming_vehicles()
        dic_feature["leaving_vehicle"] = self._get_leaving_vehicles()
        dic_feature["pressure"] = self._get_pressure()
        dic_feature["adjacency_matrix"] = None  # self._get_adjacency_row()

    def update_neighbor_info(self, neighbors, dic_feature):
        # print(dic_feature)
        none_dic_feature = deepcopy(dic_feature)
        for key in none_dic_feature.keys():
            if none_dic_feature[key] is not None:
                if "cur_phase" in key:
                    none_dic_feature[key] = [1] * len(none_dic_feature[key])
                elif "num_total_veh" in key:
                    none_dic_feature[key] = []
                else:
                    none_dic_feature[key] = [0] * len(none_dic_feature[key])
            else:
                none_dic_feature[key] = None
        for i in range(len(neighbors)):
            neighbor = neighbors[i]
            example_dic_feature = {}
            if neighbor is None:
                example_dic_feature["cur_phase_{0}".format(i)] = none_dic_feature["cur_phase"]
                example_dic_feature["time_this_phase_{0}".format(i)] = none_dic_feature["time_this_phase"]
                example_dic_feature["lane_num_vehicle_{0}".format(i)] = none_dic_feature["lane_num_vehicle"]
                example_dic_feature["lane_num_vehicle_downstream_{0}".format(i)] = none_dic_feature[
                    "lane_num_vehicle_downstream"]
            else:
                example_dic_feature["cur_phase_{0}".format(i)] = neighbor.dic_feature["cur_phase"]
                example_dic_feature["time_this_phase_{0}".format(i)] = neighbor.dic_feature["time_this_phase"]
                example_dic_feature["lane_num_vehicle_{0}".format(i)] = neighbor.dic_feature["lane_num_vehicle"]
                example_dic_feature["lane_num_vehicle_downstream_{0}".format(i)] = neighbor.dic_feature[
                    "lane_num_vehicle_downstream"]
            dic_feature.update(example_dic_feature)
        return dic_feature

    @staticmethod
    def _add_suffix_to_dict_key(target_dict, suffix):
        keys = list(target_dict.keys())
        for key in keys:
            target_dict[key + "_" + suffix] = target_dict.pop(key)
        return target_dict

    def _update_feature_map(self, simulator_state, test_flag):

        dic_feature = dict()

        dic_feature["cur_phase"] = [self.current_phase_index]
        dic_feature["time_this_phase"] = [self.current_phase_duration]
        dic_feature["vehicle_position_img"] = None  # self._get_lane_vehicle_position(self.list_entering_lanes)
        dic_feature["vehicle_speed_img"] = None  # self._get_lane_vehicle_speed(self.list_entering_lanes)
        dic_feature["vehicle_acceleration_img"] = None
        dic_feature[
            "vehicle_waiting_time_img"] = None  # self._get_lane_vehicle_accumulated_waiting_time(self.list_entering_lanes)

        dic_feature["lane_num_vehicle"] = self._get_lane_num_vehicle(self.list_entering_lanes)
        dic_feature["lane_num_vehicle_downstream"] = self._get_lane_num_vehicle_downstream(simulator_state)
        dic_feature["delta_lane_num_vehicle"] = [
            dic_feature["lane_num_vehicle"][i] - dic_feature["lane_num_vehicle_downstream"][i] for i in
            range(len(dic_feature["lane_num_vehicle_downstream"]))]
        # dic_feature["pressure"] = None # [self._get_pressure()]

        if self.fast_compute or test_flag:
            dic_feature["coming_vehicle"] = None
            dic_feature["leaving_vehicle"] = None
            # dic_feature["num_total_veh"] = simulator_state['num_total_veh']
        else:
            dic_feature["coming_vehicle"] = self._get_coming_vehicles(simulator_state)
            dic_feature["leaving_vehicle"] = self._get_leaving_vehicles(simulator_state)
            # print(simulator_state['num_total_veh'])

        dic_feature["num_total_veh"] = simulator_state['num_total_veh']

        dic_feature["pressure"] = self._get_pressure()

        dic_feature[
            "lane_num_vehicle_been_stopped_thres01"] = None  # self._get_lane_num_vehicle_been_stopped(0.1, self.list_entering_lanes)
        dic_feature["lane_num_vehicle_been_stopped_thres1"] = self._get_lane_num_vehicle_been_stopped(1,
                                                                                                      self.list_entering_lanes)
        dic_feature["lane_queue_length"] = None  # self._get_lane_queue_length(self.list_entering_lanes)
        dic_feature["lane_num_vehicle_left"] = None
        dic_feature["lane_sum_duration_vehicle_left"] = None
        dic_feature["lane_sum_waiting_time"] = None  # self._get_lane_sum_waiting_time(self.list_entering_lanes)
        dic_feature["terminal"] = None

        dic_feature["adjacency_matrix"] = self._get_adjacency_row()

        self.dic_feature = dic_feature

    def _get_adjacency_row(self):
        return self.adjacency_row

    def lane_position_mapper(self, lane_pos, bins):
        lane_pos_np = np.array(lane_pos)
        digitized = np.digitize(lane_pos_np, bins)
        position_counter = [len(lane_pos_np[digitized == i]) for i in range(1, len(bins))]
        return position_counter

    def _get_coming_vehicles(self, simulator_state):

        coming_distribution = []
        lane_vid_mapping_dict = simulator_state['get_lane_vehicles']
        vid_distance_mapping_dict = simulator_state['get_vehicle_distance']

        bins = np.linspace(0, 300, 4).tolist()

        for lane in self.list_entering_lanes:
            coming_vehicle_position = []
            vehicle_position_lane = lane_vid_mapping_dict[lane]
            for vehicle in vehicle_position_lane:
                coming_vehicle_position.append(vid_distance_mapping_dict[vehicle])
            coming_distribution.extend(self.lane_position_mapper(coming_vehicle_position, bins))

        return coming_distribution

    def _get_leaving_vehicles(self, simulator_state):
        leaving_distribution = []
        ## dimension = num_lane*3*num_list_entering_lanes

        lane_vid_mapping_dict = simulator_state['get_lane_vehicles']
        vid_distance_mapping_dict = simulator_state['get_vehicle_distance']

        ## TODO LANE LENGTH = 300
        bins = np.linspace(0, 300, 4).tolist()

        for lane in self.list_exiting_lanes:
            coming_vehicle_position = []
            vehicle_position_lane = lane_vid_mapping_dict[lane]
            for vehicle in vehicle_position_lane:
                coming_vehicle_position.append(vid_distance_mapping_dict[vehicle])
            leaving_distribution.extend(self.lane_position_mapper(coming_vehicle_position, bins))

        return leaving_distribution

    def _get_pressure(self):
        return [self.dic_lane_waiting_vehicle_count_current_step[lane] for lane in self.list_entering_lanes] + \
               [-self.dic_lane_waiting_vehicle_count_current_step[lane] for lane in self.list_exiting_lanes]

    def _get_lane_queue_length(self, list_lanes):
        '''
        queue length for each lane
        '''
        return [self.dic_lane_waiting_vehicle_count_current_step[lane] for lane in list_lanes]

    def _get_lane_num_vehicle(self, list_lanes):
        '''
        vehicle number for each lane
        '''
        return [len(self.dic_lane_vehicle_current_step[lane]) for lane in list_lanes]

    def _get_lane_num_vehicle_downstream(self, simulator_state):
        '''
        vehicle number for each lane
        '''
        lane_vid_mapping_dict = simulator_state['get_lane_vehicles']
        return [len(lane_vid_mapping_dict[lane]) for lane in self.list_exiting_lanes]

    def _get_lane_sum_waiting_time(self, list_lanes):
        '''
        waiting time for each lane
        '''
        raise NotImplementedError

    def _get_lane_list_vehicle_left(self, list_lanes):
        '''
        get list of vehicles left at each lane
        ####### need to check
        '''

        raise NotImplementedError

    # non temporary
    def _get_lane_num_vehicle_left(self, list_lanes):

        list_lane_vehicle_left = self._get_lane_list_vehicle_left(list_lanes)
        list_lane_num_vehicle_left = [len(lane_vehicle_left) for lane_vehicle_left in list_lane_vehicle_left]
        return list_lane_num_vehicle_left

    def _get_lane_sum_duration_vehicle_left(self, list_lanes):

        ## not implemented error
        raise NotImplementedError

    def _get_lane_num_vehicle_been_stopped(self, thres, list_lanes):
        return [self.dic_lane_waiting_vehicle_count_current_step[lane] for lane in list_lanes]

    def _get_lane_vehicle_position(self, list_lanes):

        list_lane_vector = []
        for lane in list_lanes:
            lane_vector = np.zeros(self.num_grid)
            list_vec_id = self.dic_lane_vehicle_current_step[lane]
            for vec in list_vec_id:
                pos = int(self.dic_vehicle_distance_current_step[vec])
                pos_grid = min(pos // self.length_grid, self.num_grid)
                lane_vector[pos_grid] = 1
            list_lane_vector.append(lane_vector)
        return np.array(list_lane_vector)

    # debug
    def _get_vehicle_info(self, veh_id):
        try:
            pos = self.dic_vehicle_distance_current_step[veh_id]
            speed = self.dic_vehicle_speed_current_step[veh_id]
            return pos, speed
        except:
            return None, None

    def _get_lane_vehicle_speed(self, list_lanes):
        return [self.dic_vehicle_speed_current_step[lane] for lane in list_lanes]

    def _get_lane_vehicle_accumulated_waiting_time(self, list_lanes):

        raise NotImplementedError

    # ================= get functions from outside ======================
    def get_current_time(self):
        return self.eng.get_current_time()

    def get_dic_vehicle_arrive_leave_time(self):

        return self.dic_vehicle_arrive_leave_time

    def get_feature(self):

        return self.dic_feature

    def get_state(self, list_state_features):
        dic_state = {state_feature_name: self.dic_feature[state_feature_name] for state_feature_name in
                     list_state_features}

        return dic_state

    def get_reward(self, dic_reward_info):

        dic_reward = dict()
        dic_reward["flickering"] = None
        dic_reward["sum_lane_queue_length"] = None
        dic_reward["sum_lane_wait_time"] = None
        dic_reward["sum_lane_num_vehicle_left"] = None
        dic_reward["sum_duration_vehicle_left"] = None
        dic_reward["sum_num_vehicle_been_stopped_thres01"] = None
        dic_reward["sum_num_vehicle_been_stopped_thres1"] = np.sum(
            self.dic_feature["lane_num_vehicle_been_stopped_thres1"])

        dic_reward['pressure'] = np.absolute(np.sum(self.dic_feature["pressure"]))

        reward = 0
        for r in dic_reward_info:
            if dic_reward_info[r] != 0:
                reward += dic_reward_info[r] * dic_reward[r]
        return reward


class AnonEnv:
    list_intersection_id = [
        "intersection_1_1"
    ]

    def __init__(self, path_to_log, path_to_work_directory, dic_traffic_env_conf):
        self.path_to_log = path_to_log
        self.path_to_work_directory = path_to_work_directory
        self.dic_traffic_env_conf = dic_traffic_env_conf
        self.simulator_type = self.dic_traffic_env_conf["SIMULATOR_TYPE"]

        self.list_intersection = None
        self.list_inter_log = None
        self.list_lanes = None
        self.system_states = None
        self.feature_name_for_neighbor = self._reduce_duplicates(self.dic_traffic_env_conf["LIST_STATE_FEATURE"])

        # check min action time
        if self.dic_traffic_env_conf["MIN_ACTION_TIME"] <= self.dic_traffic_env_conf["YELLOW_TIME"]:
            print("MIN_ACTION_TIME should include YELLOW_TIME")
            pass
            # raise ValueError

        # touch new inter_{}.pkl (if exists, remove)
        for inter_ind in range(self.dic_traffic_env_conf["NUM_INTERSECTIONS"]):
            path_to_log_file = os.path.join(self.path_to_log, "inter_{0}.pkl".format(inter_ind))
            f = open(path_to_log_file, "wb")
            f.close()

    def reset(self):
        # self.eng.reset() to be implemented

        # self.eng = engine.Engine(self.dic_traffic_env_conf["INTERVAL"],
        #                          self.dic_traffic_env_conf["THREADNUM"],
        #                          self.dic_traffic_env_conf["SAVEREPLAY"],
        #                          self.dic_traffic_env_conf["RLTRAFFICLIGHT"],
        #                          False,
        #                          0)

        #self.load_roadnet(self.dic_traffic_env_conf["ROADNET_FILE"])
        #self.load_flow(self.dic_traffic_env_conf["TRAFFIC_FILE"])

        get_cityflow_config(self.dic_traffic_env_conf["INTERVAL"],
                            0,
                            "./data/template_lsr/1_6/",
                            self.dic_traffic_env_conf["ROADNET_FILE"],
                            self.dic_traffic_env_conf["TRAFFIC_FILE"],
                            self.dic_traffic_env_conf["RLTRAFFICLIGHT"],
                            self.dic_traffic_env_conf["SAVEREPLAY"])

        self.eng = cityflow.Engine("./config/cityflow_config.json", self.dic_traffic_env_conf["THREADNUM"])


        # get adjacency
        self.traffic_light_node_dict = self._adjacency_extraction()

        # initialize intersections (grid)
        self.list_intersection = [Intersection((i + 1, j + 1), self.dic_traffic_env_conf, self.eng,
                                               self.traffic_light_node_dict[
                                                   "intersection_{0}_{1}".format(i + 1, j + 1)])
                                  for i in range(self.dic_traffic_env_conf["NUM_ROW"])
                                  for j in range(self.dic_traffic_env_conf["NUM_COL"])]
        self.list_inter_log = [[] for i in range(self.dic_traffic_env_conf["NUM_ROW"] *
                                                 self.dic_traffic_env_conf["NUM_COL"])]

        self.id_to_index = {}
        count = 0
        for i in range(self.dic_traffic_env_conf["NUM_ROW"]):
            for j in range(self.dic_traffic_env_conf["NUM_COL"]):
                self.id_to_index['intersection_{0}_{1}'.format(i + 1, j + 1)] = count
                count += 1

        self.list_lanes = []
        for inter in self.list_intersection:
            self.list_lanes += inter.list_lanes
        self.list_lanes = np.unique(self.list_lanes).tolist()
        # print(self.list_lanes)

        # get new measurements
        system_state_start_time = time.time()
        if self.dic_traffic_env_conf["FAST_COMPUTE"]:
            self.system_states = {"get_lane_vehicles": self.eng.get_lane_vehicles(),
                                  "get_lane_waiting_vehicle_count": self.eng.get_lane_waiting_vehicle_count(),
                                  "get_vehicle_speed": None,
                                  "get_vehicle_distance": None
                                  }
        else:
            self.system_states = {"get_lane_vehicles": self.eng.get_lane_vehicles(),
                                  "get_lane_waiting_vehicle_count": self.eng.get_lane_waiting_vehicle_count(),
                                  "get_vehicle_speed": None,  # self.eng.get_vehicle_speed(),
                                  "get_vehicle_distance": self.eng.get_vehicle_distance(),
                                  # "num_total_veh": np.sum([self.system_states["get_lane_vehicles"][lane] for lane in self.list_lanes])
                                  }

        self.system_states["num_total_veh"] = np.sum(
            [len(self.system_states["get_lane_vehicles"][lane]) for lane in self.list_lanes])

        update_start_time = time.time()
        for inter in self.list_intersection:
            inter.update_current_measurements_map(self.system_states, self.path_to_log, False)
        # print("Update_current_measurements_map time: ", time.time()-update_start_time)

        # update neighbor's info
        neighbor_start_time = time.time()
        if self.dic_traffic_env_conf["NEIGHBOR"]:
            for inter in self.list_intersection:
                neighbor_inter_ids = inter.neighbor_ENWS
                neighbor_inters = []
                for neighbor_inter_id in neighbor_inter_ids:
                    if neighbor_inter_id is not None:
                        neighbor_inters.append(self.list_intersection[self.id_to_index[neighbor_inter_id]])
                    else:
                        neighbor_inters.append(None)
                inter.dic_feature = inter.update_neighbor_info(neighbor_inters, deepcopy(inter.dic_feature))

        state, done = self.get_state()
        return state

    def reset_test(self):

        # self.eng.reset() to be implemented
        # self.eng = engine.Engine(self.dic_traffic_env_conf["INTERVAL"],
        #                          self.dic_traffic_env_conf["THREADNUM"],
        #                          self.dic_traffic_env_conf["SAVEREPLAY"],
        #                          self.dic_traffic_env_conf["RLTRAFFICLIGHT"],
        #                          False,
        #                          0)
        #
        # self.load_roadnet(self.dic_traffic_env_conf["ROADNET_FILE"])
        #
        # self.load_flow(self.dic_traffic_env_conf["TRAFFIC_SEPARATE"])

        get_cityflow_config(self.dic_traffic_env_conf["INTERVAL"],
                            0,
                            "./data/template_lsr/1_6/",
                            self.dic_traffic_env_conf["ROADNET_FILE"],
                            self.dic_traffic_env_conf["TRAFFIC_FILE"],
                            self.dic_traffic_env_conf["RLTRAFFICLIGHT"],
                            self.dic_traffic_env_conf["SAVEREPLAY"])

        self.eng = cityflow.Engine("./config/cityflow_config.json", self.dic_traffic_env_conf["THREADNUM"])

        # get adjacency
        self.traffic_light_node_dict = self._adjacency_extraction()

        # initialize intersections (grid)
        self.list_intersection = [Intersection((i + 1, j + 1), self.dic_traffic_env_conf, self.eng,
                                               self.traffic_light_node_dict[
                                                   "intersection_{0}_{1}".format(i + 1, j + 1)])
                                  for i in range(self.dic_traffic_env_conf["NUM_ROW"])
                                  for j in range(self.dic_traffic_env_conf["NUM_COL"])]
        self.list_inter_log = [[] for i in range(self.dic_traffic_env_conf["NUM_ROW"] *
                                                 self.dic_traffic_env_conf["NUM_COL"])]
        # get lanes list
        self.list_lanes = []
        for inter in self.list_intersection:
            self.list_lanes += inter.list_lanes
        self.list_lanes = np.unique(self.list_lanes).tolist()

        # get new measurements
        system_state_start_time = time.time()
        if self.dic_traffic_env_conf["FAST_COMPUTE"]:
            self.system_states = {"get_lane_vehicles": self.eng.get_lane_vehicles(),
                                  "get_lane_waiting_vehicle_count": self.eng.get_lane_waiting_vehicle_count(),
                                  "get_vehicle_speed": None,
                                  "get_vehicle_distance": None
                                  }
        else:
            self.system_states = {"get_lane_vehicles": self.eng.get_lane_vehicles(),
                                  "get_lane_waiting_vehicle_count": self.eng.get_lane_waiting_vehicle_count(),
                                  "get_vehicle_speed": self.eng.get_vehicle_speed(),
                                  "get_vehicle_distance": self.eng.get_vehicle_distance(),
                                  # "num_total_veh": np.sum([self.system_states["get_lane_vehicles"][lane] for lane in self.list_lanes])
                                  }

        self.system_states["num_total_veh"] = np.sum(
            [len(self.system_states["get_lane_vehicles"][lane]) for lane in self.list_lanes])

        update_start_time = time.time()
        for inter in self.list_intersection:
            inter.update_current_measurements_map(self.system_states, self.path_to_log, False)
        # print("Update_current_measurements_map time: ", time.time()-update_start_time)

        # update neighbor's info
        neighbor_start_time = time.time()
        if self.dic_traffic_env_conf["NEIGHBOR"]:
            for inter in self.list_intersection:
                neighbor_inter_ids = inter.neighbor_ENWS
                neighbor_inters = []
                for neighbor_inter_id in neighbor_inter_ids:
                    if neighbor_inter_id is not None:
                        neighbor_inters.append(self.list_intersection[self.id_to_index[neighbor_inter_id]])
                    else:
                        neighbor_inters.append(None)
                inter.dic_feature = inter.update_neighbor_info(neighbor_inters, deepcopy(inter.dic_feature))

        state, done = self.get_state()
        # print(state)
        return state

    def step(self, action, test_flag):

        step_start_time = time.time()
        list_action_in_sec = [action]
        list_action_in_sec_display = [action]
        for i in range(self.dic_traffic_env_conf["MIN_ACTION_TIME"] - 1):
            if self.dic_traffic_env_conf["ACTION_PATTERN"] == "switch":
                list_action_in_sec.append(np.zeros_like(action).tolist())
            elif self.dic_traffic_env_conf["ACTION_PATTERN"] == "set":
                list_action_in_sec.append(np.copy(action).tolist())
            list_action_in_sec_display.append(np.full_like(action, fill_value=-1).tolist())

        average_reward_action_list = [0] * len(action)
        for i in range(self.dic_traffic_env_conf["MIN_ACTION_TIME"]):

            action_in_sec = list_action_in_sec[i]
            action_in_sec_display = list_action_in_sec_display[i]

            instant_time = self.get_current_time()
            self.current_time = self.get_current_time()

            before_action_feature = self.get_feature()
            # state = self.get_state()

            if self.dic_traffic_env_conf['DEBUG']:
                print("time: {0}".format(instant_time))
            else:

                if i == 0:
                    print("time: {0}".format(instant_time))

            self._inner_step(action_in_sec, test_flag)

            # get reward
            if self.dic_traffic_env_conf['DEBUG']:
                start_time = time.time()

            reward = self.get_reward()

            if self.dic_traffic_env_conf['DEBUG']:
                print("Reward time: {}".format(time.time() - start_time))

            for j in range(len(reward)):
                average_reward_action_list[j] = (average_reward_action_list[j] * i + reward[j]) / (i + 1)

            # log
            self.log(cur_time=instant_time, before_action_feature=before_action_feature, action=action_in_sec_display)

            next_state, done = self.get_state()

        print("Step time: ", time.time() - step_start_time)
        return next_state, reward, done, average_reward_action_list

    def _inner_step(self, action, test_flag):

        # copy current measurements to previous measurements
        for inter in self.list_intersection:
            inter.update_previous_measurements()

        # set signals
        # multi_intersection decided by action {inter_id: phase}
        for inter_ind, inter in enumerate(self.list_intersection):
            inter.set_signal(
                action=action[inter_ind],
                action_pattern=self.dic_traffic_env_conf["ACTION_PATTERN"],
                yellow_time=self.dic_traffic_env_conf["YELLOW_TIME"],
                all_red_time=self.dic_traffic_env_conf["ALL_RED_TIME"]
            )

        # run one step
        for i in range(int(1 / self.dic_traffic_env_conf["INTERVAL"])):
            self.eng.next_step()

        if self.dic_traffic_env_conf['DEBUG']:
            start_time = time.time()

        system_state_start_time = time.time()
        if self.dic_traffic_env_conf["FAST_COMPUTE"] or test_flag:
            self.system_states = {"get_lane_vehicles": self.eng.get_lane_vehicles(),
                                  "get_lane_waiting_vehicle_count": self.eng.get_lane_waiting_vehicle_count(),
                                  "get_vehicle_speed": None,
                                  "get_vehicle_distance": None
                                  }
        else:
            self.system_states = {"get_lane_vehicles": self.eng.get_lane_vehicles(),
                                  "get_lane_waiting_vehicle_count": self.eng.get_lane_waiting_vehicle_count(),
                                  "get_vehicle_speed": None,  # self.eng.get_vehicle_speed(),
                                  "get_vehicle_distance": self.eng.get_vehicle_distance()
                                  }

        self.system_states["num_total_veh"] = np.sum(
            [len(self.system_states["get_lane_vehicles"][lane]) for lane in self.list_lanes])

        if self.dic_traffic_env_conf['DEBUG']:
            print("Get system state time: {}".format(time.time() - start_time))
        # get new measurements

        if self.dic_traffic_env_conf['DEBUG']:
            start_time = time.time()

        update_start_time = time.time()
        for inter in self.list_intersection:
            inter.update_current_measurements_map(self.system_states, self.path_to_log, test_flag)

        # update neighbor's info
        if self.dic_traffic_env_conf["NEIGHBOR"]:
            for inter in self.list_intersection:
                neighbor_inter_ids = inter.neighbor_ENWS
                neighbor_inters = []
                for neighbor_inter_id in neighbor_inter_ids:
                    if neighbor_inter_id is not None:
                        neighbor_inters.append(self.list_intersection[self.id_to_index[neighbor_inter_id]])
                    else:
                        neighbor_inters.append(None)
                inter.dic_feature = inter.update_neighbor_info(neighbor_inters, deepcopy(inter.dic_feature))

        if self.dic_traffic_env_conf['DEBUG']:
            print("Update measurements time: {}".format(time.time() - start_time))

        # self.log_lane_vehicle_position()
        # self.log_first_vehicle()
        # self.log_phase()

    def load_roadnet(self, roadnetFile=None):
        print("Start load roadnet")
        start_time = time.time()
        if not roadnetFile:
            roadnetFile = "roadnet_1_1.json"
        self.eng.load_roadnet(os.path.join(self.path_to_work_directory, roadnetFile))
        print("successfully load roadnet:{0}, time: {1}".format(roadnetFile, time.time() - start_time))

    def load_flow(self, flowFile=None):
        print("Start load flowFile")
        start_time = time.time()
        if not flowFile:
            flowFile = "flow_1_1.json"
        self.eng.load_flow(os.path.join(self.path_to_work_directory, flowFile))
        print("successfully load flowFile: {0}, time: {1}".format(flowFile, time.time() - start_time))

    def _check_episode_done(self, list_state):

        # ======== to implement ========

        return False

    @staticmethod
    def convert_dic_to_df(dic):
        list_df = []
        for key in dic:
            df = pd.Series(dic[key], name=key)
            list_df.append(df)
        return pd.DataFrame(list_df)

    def get_feature(self):
        list_feature = [inter.get_feature() for inter in self.list_intersection]
        return list_feature

    def get_state(self):

        # consider neighbor info
        list_state = [inter.get_state(self.dic_traffic_env_conf["LIST_STATE_FEATURE"]) for inter in
                      self.list_intersection]

        done = self._check_episode_done(list_state)

        return list_state, done

    @staticmethod
    def _reduce_duplicates(feature_name_list):
        new_list = set()
        for feature_name in feature_name_list:
            if feature_name[-1] in ["0", "1", "2", "3"]:
                new_list.add(feature_name[:-2])
        return list(new_list)

    def get_reward(self):

        list_reward = [inter.get_reward(self.dic_traffic_env_conf["DIC_REWARD_INFO"]) for inter in
                       self.list_intersection]

        return list_reward

    def get_current_time(self):
        return self.eng.get_current_time()

    def log(self, cur_time, before_action_feature, action):

        for inter_ind in range(len(self.list_intersection)):
            self.list_inter_log[inter_ind].append({"time": cur_time,
                                                   "state": before_action_feature[inter_ind],
                                                   "action": action[inter_ind]})

    def batch_log(self, start, stop):
        for inter_ind in range(start, stop):
            path_to_log_file = os.path.join(self.path_to_log, "inter_{0}.pkl".format(inter_ind))
            f = open(path_to_log_file, "wb")
            pickle.dump(self.list_inter_log[inter_ind], f)
            f.close()

        vol = get_traffic_volume(self.dic_traffic_env_conf["TRAFFIC_FILE"])
        self.eng.print_log(os.path.join(self.path_to_log, self.dic_traffic_env_conf["ROADNET_FILE"]),
                           os.path.join(self.path_to_log, "replay_1_1_%s.txt" % vol))

    def bulk_log_multi_process(self, batch_size=100):
        assert len(self.list_intersection) == len(self.list_inter_log)
        if batch_size > len(self.list_intersection):
            batch_size_run = len(self.list_intersection)
        else:
            batch_size_run = batch_size
        process_list = []
        for batch in range(0, len(self.list_intersection), batch_size_run):
            start = batch
            stop = min(batch + batch_size, len(self.list_intersection))
            p = Process(target=self.batch_log, args=(start, stop))
            print("before")
            p.start()
            print("end")
            process_list.append(p)
        print("before join")

        for t in process_list:
            t.join()

        f = open(os.path.join(self.path_to_log, "log_done.txt"), "a")
        f.close()

    def bulk_log(self):

        for inter_ind in range(len(self.list_intersection)):
            path_to_log_file = os.path.join(self.path_to_log, "vehicle_inter_{0}.csv".format(inter_ind))
            dic_vehicle = self.list_intersection[inter_ind].get_dic_vehicle_arrive_leave_time()
            df = self.convert_dic_to_df(dic_vehicle)
            df.to_csv(path_to_log_file, na_rep="nan")

        for inter_ind in range(len(self.list_inter_log)):
            path_to_log_file = os.path.join(self.path_to_log, "inter_{0}.pkl".format(inter_ind))
            f = open(path_to_log_file, "wb")
            pickle.dump(self.list_inter_log[inter_ind], f)
            f.close()

        vol = get_traffic_volume(self.dic_traffic_env_conf["TRAFFIC_FILE"])
        self.eng.print_log(os.path.join(self.path_to_log, self.dic_traffic_env_conf["ROADNET_FILE"]),
                           os.path.join(self.path_to_log, "replay_1_1_%s.txt" % vol))

    def log_attention(self, attention_dict):
        path_to_log_file = os.path.join(self.path_to_log, "attention.pkl")
        f = open(path_to_log_file, "wb")
        pickle.dump(attention_dict, f)
        f.close()

    def log_hidden_state(self, hidden_states):
        path_to_log_file = os.path.join(self.path_to_log, "hidden_states.pkl")

        with open(path_to_log_file, "wb") as f:
            pickle.dump(hidden_states, f)

    def log_lane_vehicle_position(self):
        def list_to_str(alist):
            new_str = ""
            for s in alist:
                new_str = new_str + str(s) + " "
            return new_str

        dic_lane_map = {
            "road_0_1_0_0": "w",
            "road_2_1_2_0": "e",
            "road_1_0_1_0": "s",
            "road_1_2_3_0": "n"
        }
        for inter in self.list_intersection:
            for lane in inter.list_entering_lanes:
                print(str(self.get_current_time()) + ", " + lane + ", " + list_to_str(
                    inter._get_lane_vehicle_position([lane])[0]),
                      file=open(os.path.join(self.path_to_log, "lane_vehicle_position_%s.txt" % dic_lane_map[lane]),
                                "a"))

    def log_lane_vehicle_position(self):
        def list_to_str(alist):
            new_str = ""
            for s in alist:
                new_str = new_str + str(s) + " "
            return new_str

        dic_lane_map = {
            "road_0_1_0_0": "w",
            "road_2_1_2_0": "e",
            "road_1_0_1_0": "s",
            "road_1_2_3_0": "n"
        }
        for inter in self.list_intersection:
            for lane in inter.list_entering_lanes:
                print(str(self.get_current_time()) + ", " + lane + ", " + list_to_str(
                    inter._get_lane_vehicle_position([lane])[0]),
                      file=open(os.path.join(self.path_to_log, "lane_vehicle_position_%s.txt" % dic_lane_map[lane]),
                                "a"))

    def log_first_vehicle(self):
        _veh_id = "flow_0_"
        _veh_id_2 = "flow_2_"
        _veh_id_3 = "flow_4_"
        _veh_id_4 = "flow_6_"

        for inter in self.list_intersection:
            for i in range(100):
                veh_id = _veh_id + str(i)
                veh_id_2 = _veh_id_2 + str(i)
                pos, speed = inter._get_vehicle_info(veh_id)
                pos_2, speed_2 = inter._get_vehicle_info(veh_id_2)
                # print(i, veh_id, pos, veh_id_2, speed, pos_2, speed_2)
                if not os.path.exists(os.path.join(self.path_to_log, "first_vehicle_info_a")):
                    os.makedirs(os.path.join(self.path_to_log, "first_vehicle_info_a"))

                if not os.path.exists(os.path.join(self.path_to_log, "first_vehicle_info_b")):
                    os.makedirs(os.path.join(self.path_to_log, "first_vehicle_info_b"))

                if pos and speed:
                    print("%f, %f, %f" % (self.get_current_time(), pos, speed),
                          file=open(
                              os.path.join(self.path_to_log, "first_vehicle_info_a", "first_vehicle_info_a_%d.txt" % i),
                              "a"))
                if pos_2 and speed_2:
                    print("%f, %f, %f" % (self.get_current_time(), pos_2, speed_2),
                          file=open(
                              os.path.join(self.path_to_log, "first_vehicle_info_b", "first_vehicle_info_b_%d.txt" % i),
                              "a"))

                veh_id_3 = _veh_id_3 + str(i)
                veh_id_4 = _veh_id_4 + str(i)
                pos_3, speed_3 = inter._get_vehicle_info(veh_id_3)
                pos_4, speed_4 = inter._get_vehicle_info(veh_id_4)
                # print(i, veh_id, pos, veh_id_2, speed, pos_2, speed_2)
                if not os.path.exists(os.path.join(self.path_to_log, "first_vehicle_info_c")):
                    os.makedirs(os.path.join(self.path_to_log, "first_vehicle_info_c"))

                if not os.path.exists(os.path.join(self.path_to_log, "first_vehicle_info_d")):
                    os.makedirs(os.path.join(self.path_to_log, "first_vehicle_info_d"))

                if pos_3 and speed_3:
                    print("%f, %f, %f" % (self.get_current_time(), pos_3, speed_3),
                          file=open(
                              os.path.join(self.path_to_log, "first_vehicle_info_c", "first_vehicle_info_a_%d.txt" % i),
                              "a"))
                if pos_4 and speed_4:
                    print("%f, %f, %f" % (self.get_current_time(), pos_4, speed_4),
                          file=open(
                              os.path.join(self.path_to_log, "first_vehicle_info_d", "first_vehicle_info_b_%d.txt" % i),
                              "a"))

    def log_phase(self):
        for inter in self.list_intersection:
            print("%f, %f" % (self.get_current_time(), inter.current_phase_index),
                  file=open(os.path.join(self.path_to_log, "log_phase.txt"), "a"))

    def _adjacency_extraction(self):
        traffic_light_node_dict = {}
        file = os.path.join(self.path_to_work_directory, self.dic_traffic_env_conf["ROADNET_FILE"])
        with open('{0}'.format(file)) as json_data:
            net = json.load(json_data)
            # print(net)
            for inter in net['intersections']:
                if not inter['virtual']:
                    traffic_light_node_dict[inter['id']] = {'location': {'x': float(inter['point']['x']),
                                                                         'y': float(inter['point']['y'])},
                                                            "total_inter_num": None, 'adjacency_row': None,
                                                            "inter_id_to_index": None,
                                                            "neighbor_ENWS": None}

            top_k = self.dic_traffic_env_conf["TOP_K_ADJACENCY"]
            total_inter_num = len(traffic_light_node_dict.keys())
            inter_id_to_index = {}

            edge_id_dict = {}
            for road in net['roads']:
                if road['id'] not in edge_id_dict.keys():
                    edge_id_dict[road['id']] = {}
                edge_id_dict[road['id']]['from'] = road['startIntersection']
                edge_id_dict[road['id']]['to'] = road['endIntersection']

            index = 0
            for i in traffic_light_node_dict.keys():
                inter_id_to_index[i] = index
                index += 1

            for i in traffic_light_node_dict.keys():
                location_1 = traffic_light_node_dict[i]['location']

                row = np.array([0] * total_inter_num)
                # row = np.zeros((self.dic_traffic_env_conf["NUM_ROW"],self.dic_traffic_env_conf["NUM_col"]))
                for j in traffic_light_node_dict.keys():
                    location_2 = traffic_light_node_dict[j]['location']
                    dist = AnonEnv._cal_distance(location_1, location_2)
                    row[inter_id_to_index[j]] = dist
                if len(row) == top_k:
                    adjacency_row_unsorted = np.argpartition(row, -1)[:top_k].tolist()
                elif len(row) > top_k:
                    adjacency_row_unsorted = np.argpartition(row, top_k)[:top_k].tolist()
                else:
                    adjacency_row_unsorted = [k for k in range(total_inter_num)]
                adjacency_row_unsorted.remove(inter_id_to_index[i])
                traffic_light_node_dict[i]['adjacency_row'] = [inter_id_to_index[i]] + adjacency_row_unsorted
                traffic_light_node_dict[i]['total_inter_num'] = total_inter_num

            for i in traffic_light_node_dict.keys():
                traffic_light_node_dict[i]['total_inter_num'] = inter_id_to_index
                traffic_light_node_dict[i]['neighbor_ENWS'] = []
                for j in range(4):
                    road_id = i.replace("intersection", "road") + "_" + str(j)
                    if edge_id_dict[road_id]['to'] not in traffic_light_node_dict.keys():
                        traffic_light_node_dict[i]['neighbor_ENWS'].append(None)
                    else:
                        traffic_light_node_dict[i]['neighbor_ENWS'].append(edge_id_dict[road_id]['to'])

        return traffic_light_node_dict

    @staticmethod
    def _cal_distance(loc_dict1, loc_dict2):
        a = np.array((loc_dict1['x'], loc_dict1['y']))
        b = np.array((loc_dict2['x'], loc_dict2['y']))
        return np.sqrt(np.sum((a - b) ** 2))

    def end_sumo(self):
        print("anon process end")
        pass


if __name__ == '__main__':
    pass
    inter_and_neighbor_state = {}
    inter_state = {"aaa": [122, 1, 2, 3, 3],
                   "bbb": [122, 1, 2, 3, 3],
                   "ccc": [122, 1, 2, 3, 3],
                   "ddd": [122, 1, 2, 3, 3]
                   }
    none_state = deepcopy(inter_state)
    for key in none_state.keys():
        none_state[key] = [0] * len(none_state[key])


    def _add_suffix_to_dict_key(target_dict, suffix):
        keys = list(target_dict.keys())
        for key in keys:
            target_dict[key + "_" + suffix] = target_dict.pop(key)
        return target_dict


    inter_and_neighbor_state.update(inter_state)

    id_to_index = [None, 1, 2, None]

    for i in range(4):
        if id_to_index[i] is None:  # if one's neighbor is None, fill in with zero values
            example_value = _add_suffix_to_dict_key(deepcopy(none_state), str(i))
        else:
            example_value = _add_suffix_to_dict_key(deepcopy(inter_state), str(i))
        inter_and_neighbor_state.update(example_value)
