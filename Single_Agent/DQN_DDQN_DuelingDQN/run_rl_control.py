import argparse
import json
import logging
import os
import numpy as np
from datetime import datetime
from tqdm import tqdm
import pandas as pd


import cityflow
from cityflow_env import CityFlowEnv
from utility import parse_roadnet
from dqn_agent import DQNAgent, DDQNAgent
from duelingDQN import DuelingDQNAgent
# import ray

# os.environ["CUDA_VISIBLE_DEVICES"]="0" # use GPU

def main():
    logging.getLogger().setLevel(logging.INFO)
    date = datetime.now().strftime('%Y%m%d_%H%M%S')
    parser = argparse.ArgumentParser()
    # parser.add_argument('--scenario', type=str, default='PongNoFrameskip-v4')
    parser.add_argument('--config', type=str, default='config/global_config.json', help='config file')
    parser.add_argument('--algo', type=str, default='DQN', choices=['DQN', 'DDQN', 'DuelDQN'], help='choose an algorithm')
    parser.add_argument('--inference', action="store_true", help='inference or training')
    parser.add_argument('--ckpt', type=str, help='inference or training')
    parser.add_argument('--epoch', type=int, default=10, help='number of training epochs')
    parser.add_argument('--num_step', type=int, default=10**3, help='number of timesteps for one episode, and for inference')
    parser.add_argument('--save_freq', type=int, default=100, help='model saving frequency')
    parser.add_argument('--batch_size', type=int, default=128, help='model saving frequency')
    
    args = parser.parse_args()

    # preparing config
    # # for environment
    config = json.load(open(args.config))
    config["num_step"] = args.num_step
    

    # config["replay_data_path"] = "replay"
    cityflow_config = json.load(open(config['cityflow_config_file']))
    roadnetFile = cityflow_config['dir'] + cityflow_config['roadnetFile']
    config["lane_phase_info"] = parse_roadnet(roadnetFile)

    # build cityflow environment
    env = CityFlowEnv(config)

    # # for agent
    intersection_id = list(config['lane_phase_info'].keys())[0]
    phase_list = config['lane_phase_info'][intersection_id]['phase']
    logging.info(phase_list)
    # config["state_size"] = len(config['lane_phase_info'][intersection_id]['start_lane']) + 1 # 1 is for the current phase. [vehicle_count for each start lane] + [current_phase]
    config["state_size"] = len(config['lane_phase_info'][intersection_id]['start_lane']) * env.state_time_span
    config["action_size"] = len(phase_list)
    config["batch_size"] = args.batch_size



    # build agent
    if args.algo == 'DQN':
        agent = DQNAgent(config)
    elif args.algo == 'DDQN':
        agent = DDQNAgent(config)
    elif args.algo == 'DuelDQN':
        agent = DuelingDQNAgent(config)

    # parameters for training and inference
    # batch_size = 32
    EPISODES = args.epoch
    learning_start = 300
    update_model_freq = args.batch_size
    update_target_model_freq = 500
    num_step = config['num_step']
    state_size = config['state_size']

    if not args.inference:
        # training
        if not os.path.exists("model"):
            os.makedirs("model")
        if not os.path.exists("result"):
            os.makedirs("result")
        model_dir = "model/{}_{}".format(args.algo, date)
        result_dir = "result/{}_{}".format(args.algo, date)

        os.makedirs(model_dir)
        os.makedirs(result_dir)

        total_step = 0
        episode_rewards = []
        with tqdm(total=EPISODES*args.num_step) as pbar:
            for i in range(EPISODES):
                print(i)
                env.reset()
                state = env.get_state()

                state = np.array(list(state['start_lane_vehicle_count'].values()) + [state['current_phase']] )
                state = np.reshape(state, [1, state_size])

                episode_length = 0
                episode_reward = 0
                while episode_length < num_step:
                    
                    action = agent.choose_action(state) # index of action
                    action_phase = phase_list[action] # actual action

                    # no yellow light
                    next_state, reward = env.step(action_phase) # one step
                    last_action_phase = action_phase
                    episode_length += 1
                    total_step += 1
                    episode_reward += reward

                    pbar.update(1)

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

                    # logging
                    # logging.info("\repisode:{}/{}, total_step:{}, action:{}, reward:{}"
                    #             .format(i+1, EPISODES, total_step, action, reward))
                    pbar.set_description(
                        "total_step:{}, episode:{}, episode_step:{}, reward:{}".format(total_step, i+1, episode_length, reward))

                # save episode rewards
                episode_rewards.append(episode_reward)


                # save model
                if (i + 1) % args.save_freq == 0:
                    if args.algo != 'DuelDQN':
                        agent.model.save(model_dir + "/{}-{}.h5".format(args.algo, i+1))
                    else:
                        agent.save(model_dir + "/{}-ckpt".format(args.algo), i+1)
            
            # save reward to file
            df = pd.DataFrame({"rewards": episode_rewards})
            df.to_csv(result_dir + '/rewards.csv', index=None)
        

    else:
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
                            .format(i+1, args.num_step, action, reward))


if __name__ == '__main__':
    main()
