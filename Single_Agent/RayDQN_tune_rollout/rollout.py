import argparse
import json
import os
import pickle
import ray
from ray.rllib.agents.registry import get_agent_class
from ray.tune.registry import register_env
from cityflow_env import CityflowGymEnv
import logging
from utility import parse_roadnet


EXAMPLE_USAGE = """
    python rollout.py /tmp/ray/checkpoint_dir/checkpoint-0 --run DQN
    --steps 100 --env Cityflow-v0 --out rollouts.pkl
"""


def create_parser(parser_creator=None):
    parser_creator = parser_creator or argparse.ArgumentParser
    parser = parser_creator(
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description="Roll out a reinforcement learning agent "
        "given a checkpoint.",
        epilog=EXAMPLE_USAGE)

    parser.add_argument(
        "checkpoint", type=str, help="Checkpoint from which to roll out.")
    required_named = parser.add_argument_group("required named arguments")
    required_named.add_argument(
        "--run",
        type=str,
        required=True,
        help="The algorithm or model to train. This may refer to the name "
        "of a built-on algorithm (e.g. RLLib's DQN or PPO), or a "
        "user-defined trainable function or class registered in the "
        "tune registry.")
    required_named.add_argument(
        "--env", type=str, required=True, help="The gym environment to use.")
    parser.add_argument(
        "--steps", default=10000, help="Number of steps to roll out.")
    parser.add_argument("--out", default=None, help="Output filename.")
    return parser


def run(args, config_env):
    config = {}
    # Load configuration from file
    config_dir = os.path.dirname(args.checkpoint)
    config_path = os.path.join(config_dir, "params.pkl")
    if not os.path.exists(config_path):
        config_path = os.path.join(config_dir, "../params.pkl")
    if not os.path.exists(config_path):
        if not args.config:
            raise ValueError(
                "Could not find params.pkl in either the checkpoint dir or "
                "its parent directory.")
    else:
        with open(config_path, "rb") as f:
            config = pickle.load(f)
    if "num_workers" in config:
        config["num_workers"] = min(2, config["num_workers"])
#    config = merge_dicts(config, config_env)

    ray.init()
    cls = get_agent_class(args.run)
    register_env(args.env, lambda config: CityflowGymEnv(config))
    agent = cls(env=args.env, config=config)
    agent.restore(args.checkpoint)
    env = CityflowGymEnv(config_env)

    num_steps = int(args.steps)

    if args.out is not None:
        rollouts = []
    steps = 0
    while steps < (num_steps or steps + 1):
        if args.out is not None:
            rollout = []
        state = env.reset()
        done = False
        reward_total = 0.0
        while not done and steps < (num_steps or steps + 1):
            action = agent.compute_action(state)
            next_state, reward, done, _ = env.step(action)
            reward_total += reward
            if args.out is not None:
                rollout.append([state, action, next_state, reward, done])
            steps += 1
            state = next_state
        if args.out is not None:
            rollouts.append(rollout)
        print('Episode reward', reward_total)
    if args.out is not None:
        pickle.dump(rollouts, open(args.out, 'wb'))


def env_config(config_env):
    # preparing config
    # # for environment
    config = json.load(open(config_env['config']))

    config["num_step"] = config_env['num_step']

    # config["replay_data_path"] = "replay"
    cityflow_config = json.load(open(config['cityflow_config_file']))
    roadnetFile = cityflow_config['dir'] + cityflow_config['roadnetFile']
    config["lane_phase_info"] = parse_roadnet(roadnetFile)
    config["state_time_span"] = config_env['state_time_span']
    config["time_span"] = config_env['time_span']

    # # for agent
    intersection_id = list(config['lane_phase_info'].keys())[0]
    phase_list = config['lane_phase_info'][intersection_id]['phase']
    logging.info(phase_list)
    # config["state_size"] = len(config['lane_phase_info'][intersection_id]['start_lane']) + 1 # 1 is for the current phase. [vehicle_count for each start lane] + [current_phase]
    config["state_size"] = len(config['lane_phase_info'][intersection_id]['start_lane'])
    config["action_size"] = len(phase_list)
    config["batch_size"] = config_env['batch_size']
    return config


if __name__ == "__main__":
    parser = create_parser()
    args = parser.parse_args()
    config_1 = {
        'config': '/home/skylark/test/config/global_config.json',
        'num_step': 10**3,
        'state_time_span': 5,
        'time_span': 30,
        'batch_size': 128
    }
    config_env = env_config(config_1)
    run(args, config_env)
