from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import ray
import gym
from gym.envs.registration import register
import ray.rllib.agents.dqn as dqn
from ray.rllib.agents.dqn import DQNTrainer
from ray import tune
from ray.tune.logger import pretty_print
from cityflow_env import CityflowGymEnv
from utility import parse_roadnet
import logging
from datetime import datetime
import argparse
import json
from ray.tune.registry import register_env
from ray.tune import grid_search
from ray.tune import run_experiments

def env_config(args):
    # preparing config
    # # for environment
    config = json.load(open(args.config))

    config["num_step"] = args.num_step

    # config["replay_data_path"] = "replay"
    cityflow_config = json.load(open(config['cityflow_config_file']))
    roadnetFile = cityflow_config['dir'] + cityflow_config['roadnetFile']
    config["lane_phase_info"] = parse_roadnet(roadnetFile)
    config["state_time_span"] = args.state_time_span
    config["time_span"] = args.time_span

    # # for agent
    intersection_id = list(config['lane_phase_info'].keys())[0]
    phase_list = config['lane_phase_info'][intersection_id]['phase']
    logging.info(phase_list)
    # config["state_size"] = len(config['lane_phase_info'][intersection_id]['start_lane']) + 1 # 1 is for the current phase. [vehicle_count for each start lane] + [current_phase]
    config["state_size"] = len(config['lane_phase_info'][intersection_id]['start_lane'])
    config["action_size"] = len(phase_list)
    config["batch_size"] = args.batch_size
    return config


if __name__ == "__main__":
    # Can also register the env creator function explicitly with:
    register_env("Cityflow-v0", lambda config: CityflowGymEnv(config))
    ray.init()
    logging.getLogger().setLevel(logging.INFO)
    date = datetime.now().strftime('%Y%m%d_%H%M%S')
    parser = argparse.ArgumentParser()
    # parser.add_argument('--scenario', type=str, default='PongNoFrameskip-v4')
    parser.add_argument('--config', type=str, default='config/global_config.json', help='config file')
    parser.add_argument('--algo', type=str, default='DQN', choices=['DQN', 'DDQN', 'DuelDQN'],
                        help='choose an algorithm')
    parser.add_argument('--inference', action="store_true", help='inference or training')
    parser.add_argument('--ckpt', type=str, help='inference or training')
    parser.add_argument('--epoch', type=int, default=10, help='number of training epochs')
    parser.add_argument('--num_step', type=int, default=10 ** 3,
                        help='number of timesteps for one episode, and for inference')
    parser.add_argument('--save_freq', type=int, default=100, help='model saving frequency')
    parser.add_argument('--batch_size', type=int, default=128, help='model saving frequency')
    parser.add_argument('--state_time_span', type=int, default=5, help='state interval to receive long term state')
    parser.add_argument('--time_span', type=int, default=30, help='time interval to collect data')

    args = parser.parse_args()

    model_dir = "model/{}_{}".format(args.algo, date)
    result_dir = "result/{}_{}".format(args.algo, date)

    config_env = env_config(args)

    tune.run(
        "DQN",
        checkpoint_freq=2,
        checkpoint_at_end=True,
        config={
                "env": "Cityflow-v0",  # or "corridor" if registered above
                "num_gpus": 0,
                "num_workers": 1,  # parallelism
                "env_config": config_env
        },
    )
