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


# import ray

def main():
    logging.getLogger().setLevel(logging.INFO)
    date = datetime.now().strftime('%Y%m%d_%H%M%S')
    parser = argparse.ArgumentParser()
    # parser.add_argument('--scenario', type=str, default='PongNoFrameskip-v4')
    parser.add_argument('--config', type=str, default='config/global_config.json')
    parser.add_argument('--num_step', type=int, default=10 ** 3)
    args = parser.parse_args()

    # preparing config
    # # for rnvironment
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
    config["state_size"] = len(config['lane_phase_info'][intersection_id][
                                   'start_lane']) + 1  # 1 is for the current phase. [vehicle_count for each start lane] + [current_phase]
    config["action_size"] = len(phase_list)

    # build cotyflow environment
    env = CityFlowEnv(config)

    # build agent
    agent = DQNAgent(config)

    # training
    batch_size = 32
    EPISODES = 11
    learning_start = 300
    update_model_freq = 300
    update_target_model_freq = 1500
    num_step = config['num_step']
    state_size = config['state_size']

    ### the dqp learning code
    if not os.path.exists("model"):
        os.makedirs("model")
    model_dir = "model/{}".format(date)
    os.makedirs(model_dir)

    total_step = 0
    for i in range(EPISODES):
        env.reset()
        state = env.get_state()
        state = np.array(list(state['start_lane_vehicle_count'].values()) + [state['current_phase']])
        state = np.reshape(state, [1, state_size])

        episode_length = 0
        while episode_length < num_step:
            action = agent.choose_action(state)  # index of action
            action_phase = phase_list[action]  # actual action
            # no yellow light
            next_state, reward = env.step(action_phase)  # one step
            last_action_phase = action_phase
            episode_length += 1
            total_step += 1

            # store to replay buffer
            next_state = np.array(list(next_state['start_lane_vehicle_count'].values()) + [next_state['current_phase']])
            next_state = np.reshape(next_state, [1, state_size])
            agent.remember(state, action_phase, reward, next_state)

            state = next_state

            # training
            if total_step > learning_start and total_step % update_model_freq == 0:
                agent.replay()

            # update target Q netwark
            if total_step > learning_start and total_step % update_target_model_freq == 0:
                agent.update_target_network()

            # log
            logging.info("episode:{}/{}, total_step:{}, action:{}, reward:{}"
                         .format(i, EPISODES, total_step, action, reward))

        # save model
        if i % 10 == 0:
            agent.model.save(model_dir + "/dqn-{}.h5".format(i))

    # save simulation replay
    # env.log()
    # automatically


if __name__ == '__main__':
    main()
