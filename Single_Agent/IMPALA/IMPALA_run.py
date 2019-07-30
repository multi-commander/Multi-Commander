import ray
import logging
import json
import os
import argparse
from datetime import datetime
from utility import parse_roadnet
import ray.rllib.agents.impala as impala
from ray.tune.logger import pretty_print

from cityflow_env import CityFlowEnv

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

ray.init()
config=impala.DEFAULT_CONFIG.copy()
trainer=impala.ImpalaTrainer(config=config, env=env)

for i in range(1000):
   # Perform one iteration of training the policy with PPO
   result = trainer.train()
   print(pretty_print(result))

   if i % 100 == 0:
       checkpoint = trainer.save()
       print("checkpoint saved at", checkpoint)
