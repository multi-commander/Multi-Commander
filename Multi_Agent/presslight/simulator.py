import numpy as np
import os
import sys
import pickle
from sys import platform
from math import floor, ceil
import pandas as pd
import engine

class Anon:
    def __init__(self, path_to_log, path_to_work_directory, dic_traffic_env_conf):
        self.path_to_log = path_to_log
        self.path_to_work_directory = path_to_work_directory
        self.dic_traffic_env_conf = dic_traffic_env_conf

        self.eng = engine.Engine(self.dic_traffic_env_conf["INTERVAL"],
                                    self.dic_traffic_env_conf["THREADNUM"],
                                    self.dic_traffic_env_conf["SAVEREPLAY"],
                                    self.dic_traffic_env_conf["RLTRAFFICLIGHT"])
        self.load_roadnet()
        self.load_flow()



    def reset(self):
        self.load_roadnet()
        self.load_flow()

    def load_roadnet(self, roadnetFile=None):
        if not roadnetFile:
            roadnetFile = "roadnet_1_1.json"
        #print("/n/n", os.path.join(self.path_to_work_directory, roadnetFile))
        self.eng.load_roadnet(os.path.join(self.path_to_work_directory, roadnetFile))
        print("successfully load roadnet: ", roadnetFile)

    def load_flow(self, flowFile=None):
        if not flowFile:
            flowFile = "flow_1_1.json"
        self.eng.load_flow(os.path.join(self.path_to_work_directory, flowFile))
        print("successfully load flowFile: ", flowFile)

    def set_tl_phase(self, intersection_id, phase):
        self.eng.set_tl_phase(intersection_id, phase)

    def next_step(self):
        self.eng.next_step()

    def get_current_time(self):
        return self.eng.get_current_time()

    def get_lane_vehicle_count(self):
        return self.eng.get_lane_vehicle_count()

    def get_lane_vehicle(self):
        return self.eng.get_lane_vehicles()

    def get_lane_waiting_vehicle_count(self):
        return self.eng.get_lane_waiting_vehicle_count()

    def get_vehicle_speed(self):
        return self.eng.get_vehicle_speed()

    def get_vehicle_distance(self):
        return self.eng.get_vehicle_distance

    def replay_log(self, roadFile, flowFile):
        self.eng.print_log(roadFile, flowFile)




