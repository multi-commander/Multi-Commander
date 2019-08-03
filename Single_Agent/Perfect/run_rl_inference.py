import argparse
import json
import logging
import os
import numpy as np
from datetime import datetime

import cityflow
from cityflow_env import CityFlowEnv
from utility import parse_roadnet
from dqn_agent import DQNAgent


def main():
    logging.getLogger().setLevel(logging.INFO)
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='config/global_config.json')
    parser.add_argument('--num_step', type=int, default=2000)
    parser.add_argument('--ckpt', type=str)
    parser.add_argument('--algo', type=str, default='DQN', choices=['DQN', 'DDQN', 'DuelDQN'], help='choose an algorithm')


    args = parser.parse_args()

    # preparing config
    # # for environment
    config = json.load(open(args.config))
    config["num_step"] = args.num_step
    cityflow_config = json.load(open(config['cityflow_config_file']))
    roadnetFile = cityflow_config['dir'] + cityflow_config['roadnetFile']
    config["lane_phase_info"] = parse_roadnet(roadnetFile)

    # # for agent
    intersection_id = "intersection_1_1"
    config["intersection_id"] = intersection_id
    config["state_size"] = len(config['lane_phase_info'][intersection_id]['start_lane']) + 1  # 1 is for the current phase. [vehicle_count for each start lane] + [current_phase]
    phase_list = config['lane_phase_info'][intersection_id]['phase']
    config["action_size"] = len(phase_list)
    config["batch_size"] = args.batch_size
    
    logging.info(phase_list)

    # build cityflow environment
    env = CityFlowEnv(config)

    # build agent
    agent = DQNAgent(config)
    
    # inference
    agent.load(args.ckpt)
    env.reset()
    state = env.get_state()
    
    for i in range(args.num_step): 
        action = agent.choose_action(state) # index of action
        action_phase = phase_list[action] # actual action
        next_state, reward = env.step(action_phase) # one step

        state = next_state

        # logging
        logging.info("step:{}/{}, action:{}, reward:{}"
                        .format(i, args.num_step, action, reward))
    
if __name__ == '__main__':
    main()
