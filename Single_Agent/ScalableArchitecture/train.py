import os
import argparse
import json
import logging
from datetime import datetime

import ray
import ray.rllib.agents.a3c as a3c
import ray.rllib.agents.dqn as dqn
import ray.rllib.agents.impala as impala
import ray.rllib.agents.ppo as ppo
from gym_cityflow.envs.cityflow_env import CityflowGymEnv
from ray import tune
from ray.rllib.agents.registry import get_agent_class
from ray.tune import grid_search, register_env

from utility import parse_roadnet

os.environ['CUDA_VISIBLE_DEVICES'] = '0'

def gen_env_config():
    # preparing config
    # for environment
    with open(args.config) as f:
        config = json.load(f)
    with open(config['cityflow_config_file']) as f:
        cityflow_config = json.load(f)

    roadnet_file = cityflow_config['dir'] + cityflow_config['roadnetFile']

    config["num_step"] = args.num_step
    config["state_time_span"] = args.state_time_span
    config["time_span"] = args.time_span

    config["lane_phase_info"] = parse_roadnet(roadnet_file)

    intersection_id = list(config['lane_phase_info'].keys())[0]
    # intersection_id = list(config['lane_phase_info'].keys())

    config["intersection_id"] = intersection_id
    phase_list = config['lane_phase_info'][intersection_id]['phase']
    # logging.info(phase_list)

    config["state_size"] = len(config['lane_phase_info'][intersection_id]['start_lane'])
    config["action_size"] = len(phase_list)
    config["batch_size"] = args.batch_size
    return config


def gen_trainer_config(env_config):
    # if args.algo == 'DQN':
    #     config = dqn.DEFAULT_CONFIG.copy()
    # elif args.algo == 'PPO':
    #     config = ppo.DEFAULT_CONFIG.copy()
    # elif args.algo == 'APEX':
    #     config = dqn.apex.APEX_DEFAULT_CONFIG.copy()
    # elif args.algo == 'APPO':
    #     config = ppo.appo.DEFAULT_CONFIG.copy()
    # elif args.algo == 'IMPALA':
    #     config = impala.DEFAULT_CONFIG.copy()
    # elif args.algo == 'A3C':
    #     config = a3c.DEFAULT_CONFIG.copy()
    # elif args.algo == 'A2C':
    #     config = a3c.a2c.A2C_DEFAULT_CONFIG.copy()
    # else:
    #     assert 0 == 1, 'Unexpected args.algo.'
    config = {"ignore_worker_failures": True,
              "env": CityflowGymEnv, "env_config": env_config,
              "num_gpus": 0, "num_workers": 24,
              "num_cpus_per_worker": 1,  # "num_gpus_per_worker": 0.03125,
              "num_cpus_for_driver": 1}

    # config['lr'] = grid_search([1e-2, 1e-3, 1e-4])
    return config


def training_workflow(config_, reporter):
    # build trainer
    cls = get_agent_class(args.algo)
    trainer = cls(env=CityflowGymEnv, config=config_)
    for i in range(args.epoch):
        res = trainer.train()
        reporter(**res)

        # if i % 100 == 0:
        #     checkpoint = trainer.save()
        #     print(f'checkpoint saved at {checkpoint}')


def create_parser():
    parser = argparse.ArgumentParser()

    parser.add_argument('--config', type=str, default='config/global_config.json', help='config file')

    parser.add_argument('--algo', type=str, default='DQN',
                        choices=['DQN', 'PPO', 'APEX', 'APPO', 'IMPALA', 'A3C', 'A2C'])
    parser.add_argument('--inference', action="store_true", help='inference or training')
    parser.add_argument('--ckpt', type=str, help='inference or training')
    parser.add_argument('--epoch', type=int, default=10, help='number of training epochs')
    parser.add_argument('--num_step', type=int, default=10 ** 3,
                        help='number of time steps for one episode, and for inference')
    parser.add_argument('--save_freq', type=int, default=100, help='model saving frequency')
    parser.add_argument('--batch_size', type=int, default=128, help='model saving frequency')
    parser.add_argument('--state_time_span', type=int, default=5, help='state interval to receive long term state')
    parser.add_argument('--time_span', type=int, default=30, help='time interval to collect data')

    return parser


ray.init()

parser = create_parser()
args = parser.parse_args()

env_config = gen_env_config()
trainer_config = gen_trainer_config(env_config)

# register_env('cityflow_single_agent',
#              lambda config_: CityflowGymEnv(config_))

tune.run(
        args.algo,
        checkpoint_freq=args.save_freq,
        checkpoint_at_end=True,
        stop={'training_iteration': args.epoch},
        config=trainer_config,
    )

