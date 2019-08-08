'''
Multi-intersection control, using Ray multi-agent algorithms
'''
import ray
from ray import tune
from ray.tune import register_env, grid_search
import ray.rllib.agents.dqn as dqn
from ray.rllib.agents.dqn import DQNTrainer
import ray.rllib.env.group_agents_wrapper
import os

from ray.tune.logger import pretty_print
from gym.spaces import Tuple
from utility import parse_roadnet
import logging
from datetime import datetime
from tqdm import tqdm
import argparse
import json
from cityflow_env import CityFlowEnvRay
import cityflow

parser = argparse.ArgumentParser()
# parser.add_argument('--scenario', type=str, default='PongNoFrameskip-v4')
parser.add_argument('--config', type=str, default='config/global_config_multi.json', help='config file')
parser.add_argument('--algo', type=str, default='QMIX', choices=['QMIX', 'APEX_QMIX'],
                    help='choose an algorithm')
parser.add_argument('--inference', action="store_true", help='inference or training')
parser.add_argument('--ckpt', type=str, help='inference or training')
parser.add_argument('--epoch', type=int, default=500, help='number of training epochs')
parser.add_argument('--num_step', type=int, default=1500,help='number of timesteps for one episode, and for inference')
parser.add_argument('--save_freq', type=int, default=50, help='model saving frequency')
parser.add_argument('--batch_size', type=int, default=128, help='model saving frequency')
parser.add_argument('--state_time_span', type=int, default=5, help='state interval to receive long term state')
parser.add_argument('--time_span', type=int, default=30, help='time interval to collect data')

os.environ["CUDA_VISIBLE_DEVICES"]="0, 1" # use GPU

def generate_config(args):
    with open(args.config) as f:
        config = json.load(f)
    with open(config['cityflow_config_file']) as f:
        cityflow_config = json.load(f)
    roadnetFile = cityflow_config['dir'] + cityflow_config['roadnetFile']
    config["num_step"] = args.num_step
    config["lane_phase_info"] = parse_roadnet(roadnetFile)
    intersection_id = list(config['lane_phase_info'].keys())
    config["intersection_id"] = intersection_id
    config["state_time_span"] = args.state_time_span
    config["time_span"] = args.time_span
    config["thread_num"] = 1
    config["state_time_span"] = args.state_time_span
    config["time_span"] = args.time_span
    # phase_list = config['lane_phase_info'][intersection_id]['phase']
    # logging.info(phase_list)
    # config["state_size"] = len(config['lane_phase_info'][intersection_id]['start_lane']) + 1 # 1 is for the current phase. [vehicle_count for each start lane] + [current_phase]
    return config

def main():
    args = parser.parse_args()
    config = generate_config(args)

    # env = CityFlowEnvRay(config)
    # eng = cityflow.Engine(config["cityflow_config_file"], thread_num = config["thread_num"])
    # config["eng"] = [eng,]
    # print(config["eng"])
    num_agents = len(config["intersection_id"])
    grouping = {
        "group_1":[id_ for id_ in config["intersection_id"]]
    }
    obs_space = Tuple([
        CityFlowEnvRay.observation_space for _ in range(num_agents)
    ])
    act_space = Tuple([
        CityFlowEnvRay.action_space for _ in range(num_agents)
    ])
    register_env(
        "cityflow_multi",
        lambda config_: CityFlowEnvRay(config_).with_agent_groups(
            grouping, obs_space=obs_space, act_space=act_space))

    if args.algo == "QMIX":
        config_ = {
            # "num_workers": 2,
            "num_gpus_per_worker":0,
            "sample_batch_size": 4,
            "num_cpus_per_worker": 3,
            "train_batch_size": 32,
            "exploration_final_eps": 0.0,
            "num_workers": 8,
            "mixer": grid_search(["qmix"]),
            "env_config":config
        }
        group = True
    elif args.algo == "APEX_QMIX":
        config_ = {
            "num_gpus": 1,
            "num_workers": 2,
            "optimizer": {
                "num_replay_buffer_shards": 1,
            },
            "min_iter_time_s": 3,
            "buffer_size": 2000,
            "learning_starts": 300,
            "train_batch_size": 64,
            "sample_batch_size": 32,
            "target_network_update_freq": 100,
            "timesteps_per_iteration": 1000,
            "env_config":config
        }
        group = True
    else:
        config_ = {}
        group = False
    
    ray.init()
    tune.run(
        args.algo,
        stop={
            "timesteps_total":args.epoch*args.num_step
        },
        checkpoint_freq=args.save_freq,
        config=dict(config_,
        **{"env":"cityflow_multi"}),
    )


if __name__ == '__main__':
    main()
