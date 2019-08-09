import ray
import ray.rllib.agents.dqn as dqn
from ray.rllib.agents.dqn import DQNTrainer
from ray.tune.logger import pretty_print
import gym
import gym_cityflow
#from gym_cityflow.envs.cityflow_env import CityflowGymEnv
from gym_cityflow.envs.cityflow_env_ray import CityFlowEnvRay
from gym.spaces import Tuple
import random
from ray.rllib.agents.dqn.dqn_policy import DQNTFPolicy

####### dw
import cityflow
import re

from utility import parse_roadnet
import logging
from datetime import datetime
from tqdm import tqdm
import argparse
import json


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
    intersection_id = list(config['lane_phase_info'].keys())
    config['intersection_id'] = intersection_id
    config["thread_num"] = 1
    #phase_list = config['lane_phase_info'][intersection_id]['phase']
    #logging.info(phase_list)
    # config["state_size"] = len(config['lane_phase_info'][intersection_id]['start_lane']) + 1 # 1 is for the current phase. [vehicle_count for each start lane] + [current_phase]
    #config["state_size"] = len(config['lane_phase_info'][intersection_id]['start_lane'])
    #config["action_size"] = len(phase_list)
    #config["batch_size"] = args.batch_size
    return config

def policy_mapping_fn(agent_id):

 
	# print("############")
	# print(agent_id)
	# print("############")

	if agent_id.startswith("intersection_1"):
		return "policy_0"
	if agent_id.startswith("intersection_2"):
		return "policy_1"
	if agent_id.startswith("intersection_3"):
		return "policy_2"
	if agent_id.startswith("intersection_4"):
		return "policy_3"
	if agent_id.startswith("intersection_5"):
		return "policy_4"
	else:
		return "policy_5"




def agent_config(config_env, policies, policy_ids):
    config = dqn.DEFAULT_CONFIG.copy()
    config["num_gpus"] = 0
    config["num_workers"] = 1
    config["env"] = CityFlowEnvRay
    config["env_config"] = config_env
    config["multiagent"] = {
            "policies":policies,
            "policy_mapping_fn":
			# "policy_mapping_fn":
			# 	lambda agent_id:
			# 		"policy_{}".format(agent_id[13:14]-1)  # Traffic lights are always controlled by this policy
				lambda agent_id:
					"policy_{}".format(int(''.join(re.findall(r"intersection_(\d)_1", agent_id)))-int(1))

            
                    
        }

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
    parser.add_argument('--batch_size', type=int, default=128, help='model saving frequency')
    parser.add_argument('--state_time_span', type=int, default=5, help='state interval to receive long term state')
    parser.add_argument('--time_span', type=int, default=30, help='time interval to collect data')

    args = parser.parse_args()

    ### dw ###
   	#parser.add_argument("--num-agents", type=int, default=6)

    model_dir = "model/{}_{}".format(args.algo, date)
    result_dir = "result/{}_{}".format(args.algo, date)

    config_env = env_config(args)

    num_agents = len(config_env["intersection_id"])

    '''
    obs_space = Tuple([
        CityFlowEnvRay.observation_space for _ in range(num_agents)
    ])
    act_space = Tuple([
        CityFlowEnvRay.action_space for _ in range(num_agents)
    ])
    '''

     ### dw ###
    obs_space = CityFlowEnvRay.observation_space
    act_space = CityFlowEnvRay.action_space


    ray.tune.register_env('gym_cityflow', lambda env_config: CityFlowEnvRay(env_config))

    #config_agent = agent_config(config_env)

    # # build cityflow environment

    '''
    trainer = DQNTrainer(
        env=CityFlowEnvRay,
        config=config_agent)
    '''

 
    policies = {
        #"dqn_policy":(None, obs_space, act_space, config_env)
        #"policy_{}".format(i): (None, obs_space, act_space, config_env)
        "policy_{}".format(i): (DQNTFPolicy, obs_space, act_space, {})
        
        for i in range(num_agents)
        

    }
    policy_ids = list(policies.keys())

    config_agent = agent_config(config_env, policies,policy_ids)



    
    trainer = DQNTrainer(env='gym_cityflow', 
                        config=config_agent)

    


    for i in range(1000):
        # Perform one iteration of training the policy with DQN
        result = trainer.train()
        print(pretty_print(result))

        if i % 30 == 0:
            checkpoint = trainer.save()
            print("checkpoint saved at", checkpoint)





if __name__ == '__main__':
    main()
