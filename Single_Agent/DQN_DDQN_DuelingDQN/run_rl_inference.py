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
    parser.add_argument('--num_step', type=int, default=10**3)
    parser.add_argument('--ckpt', type=str)

    args = parser.parse_args()

    # preparing config
    # # for environment
    config = json.load(open(args.config))
    config["num_step"] = args.num_step
    # config["replay_data_path"] = "replay"
    cityflow_config = json.load(open(config['cityflow_config_file']))
    roadnetFile = cityflow_config['dir'] + cityflow_config['roadnetFile']
    config["lane_phase_info"] = parse_roadnet(roadnetFile)

    # # for agent
    intersection_id = list(config['lane_phase_info'].keys())[0]
    phase_list = config['lane_phase_info'][intersection_id]['phase']
    logging.info(phase_list)
    state_size = config["state_size"] = len(config['lane_phase_info'][intersection_id]['start_lane']) + 1 # 1 is for the current phase. [vehicle_count for each start lane] + [current_phase]
    config["action_size"] = len(phase_list)

    # build cotyflow environment
    env = CityFlowEnv(config)

    # build agent
    agent = DQNAgent(config)
    
    # inference
    agent.load(args.ckpt)
    env.reset()
    state = env.get_state()
    state = np.array(list(state['start_lane_vehicle_count'].values()) + [state['current_phase']] )
    state = np.reshape(state, [1, state_size])
    
    for i in range(args.num_step): 
        action = agent.choose_action(state) # index of action
        action_phase = phase_list[action] # actual action
        next_state, reward = env.step(action_phase) # one step

        next_state = np.array(list(next_state['start_lane_vehicle_count'].values()) + [next_state['current_phase']])
        next_state = np.reshape(next_state, [1, state_size])

        state = next_state

        # logging
        logging.info("step:{}/{}, action:{}, reward:{}"
                        .format(i, args.num_step, action, reward))
    
    # copy file to front/replay
    # roadnetLog = os.path.join(cityflow_config['dir'], cityflow_config['roadnetLogFile'])
    # replayLog = os.path.join(cityflow_config['dir'], cityflow_config['replayLogFile'])



if __name__ == '__main__':
    main()
