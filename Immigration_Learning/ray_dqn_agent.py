import ray
import ray.rllib.agents.dqn as dqn
from ray.rllib.agents.dqn import DQNTrainer
from ray.tune.logger import pretty_print
import gym
import gym_cityflow
from gym_cityflow.envs.cityflow_env import CityflowGymEnv
from utility import parse_roadnet
from ray import tune
import logging
from datetime import datetime
from tqdm import tqdm
import argparse
import json
import tensorflow as tf 


def env_config(args,clock=False):
    # preparing config
    # # for environment
    config = json.load(open(args.config))

    config["num_step"] = args.num_step

    # config["replay_data_path"] = "replay"
    if clock:
        cityflow_config = json.load(open(config['new_cityflow_config_file']))
    else:
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


def agent_config(config_env):
    config = dqn.DEFAULT_CONFIG.copy()
    config["num_gpus"] = 0
    config["num_workers"] = 1
    config["env"] = CityflowGymEnv
    config["env_config"] = config_env
    return config

# def get_episode_reward(info):
#     episode=info



def main():
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
    parser.add_argument('--batch_size', type=int, default=1, help='model saving frequency')
    parser.add_argument('--state_time_span', type=int, default=5, help='state interval to receive long term state')
    parser.add_argument('--time_span', type=int, default=30, help='time interval to collect data')

    args = parser.parse_args()

    model_dir = "model/{}_{}".format(args.algo, date)
    result_dir = "result/{}_{}".format(args.algo, date)

    config_env = env_config(args)
    # ray.tune.register_env('gym_cityflow', lambda env_config:CityflowGymEnv(config_env))
    new_config_env=env_config(args,True)
    config_agent = agent_config(config_env)
    new_config_agent=agent_config(new_config_env)
    # # build cityflow environment
    
    tune.run(
        'DQN',
        stop={"training_iteration": 100},
        config=config_agent,
        local_dir='~/ray_results/training/',
        checkpoint_freq=1
    )
    
    print('-------------------------------training over----------------------------')
    tune.run(
        'DQN',
        stop={"training_iteration": 1000},
        config=new_config_agent,
        restore='~/ray_results/training/DQN',
        checkpoint_freq=1
    )
        
    
#     trainer = DQNTrainer(
#         env=CityflowGymEnv,
#         config=config_agent)
#     for i in range(10):
#         # Perform one iteration of training the policy with DQN
#         result = trainer.train()
#         print(pretty_print(result))
#         #each step
#         checkpoint = trainer.save()
#         print("checkpoint saved at", checkpoint)

#     new_trainer=DQNTrainer(
#         env=CityflowGymEnv,
#         config=new_config_agent
#     )
#     result = trainer.train()
#     print(pretty_print(result))
#     checkpoint = trainer.save()
#     print("checkpoint saved at", checkpoint)

if __name__ == '__main__':
    main()
