'''
multiple intersection, independent dqn/rule based policy
'''
import argparse
import json
import logging
import os
import numpy as np
from datetime import datetime
from tqdm import tqdm
import pandas as pd

import cityflow
from cityflow_env import CityFlowEnvM
# from test.cityflow_env import CityFlowEnv
from utility import parse_roadnet, plot_data_lists
from dqn_agent import MDQNAgent
# import ray

os.environ["CUDA_VISIBLE_DEVICES"]="0, 1, 2, 3" # use GPU

def main():
    logging.getLogger().setLevel(logging.INFO)
    date = datetime.now().strftime('%Y%m%d_%H%M%S')
    parser = argparse.ArgumentParser()
    # parser.add_argument('--scenario', type=str, default='PongNoFrameskip-v4')
    parser.add_argument('--config', type=str, default='config/global_config_multi.json', help='config file')
    parser.add_argument('--algo', type=str, default='MDQN', choices=['MDQN',], help='choose an algorithm')
    parser.add_argument('--inference', action="store_true", help='inference or training')
    parser.add_argument('--ckpt', type=str, help='inference or training')
    parser.add_argument('--epoch', type=int, default=10, help='number of training epochs')
    parser.add_argument('--num_step', type=int, default=1500, help='number of timesteps for one episode, and for inference')
    parser.add_argument('--save_freq', type=int, default=1, help='model saving frequency')
    parser.add_argument('--batch_size', type=int, default=32, help='batchsize for training')
    parser.add_argument('--phase_step', type=int, default=15, help='seconds of one phase')
    
    args = parser.parse_args()

    config = json.load(open(args.config))
    config["num_step"] = args.num_step
    cityflow_config = json.load(open(config['cityflow_config_file']))
    roadnetFile = cityflow_config['dir'] + cityflow_config['roadnetFile']

    cityflow_config["saveReplay"] = True if args.inference else False
    json.dump(cityflow_config, open(config["cityflow_config_file"], 'w'))
    
    config["lane_phase_info"] = parse_roadnet(roadnetFile)
    config["batch_size"] = args.batch_size
    intersection_id = list(config['lane_phase_info'].keys()) # all intersections
    config["intersection_id"] = intersection_id
    phase_list = {id_:config["lane_phase_info"][id_]["phase"] for id_ in intersection_id}
    config["phase_list"] = phase_list

    model_dir = "model/{}_{}".format(args.algo, date)
    result_dir = "result/{}_{}".format(args.algo, date)
    config["result_dir"] = result_dir
    
    # parameters for training and inference
    EPISODES = args.epoch
    learning_start = 300
    update_model_freq = args.batch_size//3
    update_target_model_freq = 300//args.phase_step

    # make dirs
    if not os.path.exists("model"):
        os.makedirs("model")
    if not os.path.exists("result"):
        os.makedirs("result") 
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
    if not os.path.exists(result_dir):
        os.makedirs(result_dir)
    
    env = CityFlowEnvM(config["lane_phase_info"],
                        intersection_id,
                        num_step=config["num_step"],
                        thread_num=1,
                        cityflow_config_file=config["cityflow_config_file"]
                        )
    
    config["state_size"] = env.state_size
    if args.algo == 'MDQN':
        Magent = MDQNAgent(intersection_id,
                            state_size=config["state_size"],
                            batch_size=config["batch_size"],
                            phase_list=config["phase_list"], # action_size is len(phase_list[id_])
                            env=env
                            )
    else:
        raise Exception("{} algorithm not implemented now".format(args.algo))

    if not args.inference:  # training
        total_step = 0
        episode_rewards = {id_:[] for id_ in intersection_id}
        episode_scores = {id_:[] for id_ in intersection_id}
        with tqdm(total=EPISODES*args.num_step) as pbar:
            for i in range(EPISODES):
                # print("episode: {}".format(i))
                env.reset()
                state = env.get_state()

                episode_length = 0
                episode_reward = {id_:0 for id_ in intersection_id} # for every agent
                episode_score = {id_:0 for id_ in intersection_id} # for everg agent
                while episode_length < args.num_step:
                    
                    action = Magent.choose_action(state) # index of action
                    action_phase = {}
                    for id_, a in action.items():
                        action_phase[id_] = phase_list[id_][a]
                    
                    next_state, reward = env.step(action_phase) # one step
                    score = env.get_score()

                    # consistent time of every phase
                    for _ in range(args.phase_step-1):
                        next_state, reward_ = env.step(action_phase)
                        score_ = env.get_score()
                        for id_ in intersection_id:
                            reward[id_] += reward_[id_]
                            score[id_] += score_[id_]

                    for id_ in intersection_id:
                        reward[id_] /= args.phase_step
                        score[id_] /= args.phase_step

                    for id_ in intersection_id:
                        episode_reward[id_] += reward[id_]
                        episode_score[id_] += score[id_]

                    episode_length += 1
                    total_step += 1
                    pbar.update(1)

                    # store to replay buffer
                    if episode_length > learning_start:
                        Magent.remember(state, action_phase, reward, next_state)

                    state = next_state

                    # training
                    if episode_length > learning_start and total_step % update_model_freq == 0 :
                        if len(Magent.agents[intersection_id[0]].memory) > args.batch_size:
                            Magent.replay()

                    # update target Q netwark
                    if episode_length > learning_start and total_step % update_target_model_freq == 0 :
                        Magent.update_target_network()

                    # logging.info("\repisode:{}/{}, total_step:{}, action:{}, reward:{}"
                    #             .format(i+1, EPISODES, total_step, action, reward))
                    print_reward = {'_'.join(k.split('_')[1:]):v for k, v in reward.items()}
                    pbar.set_description(
                        "t_st:{}, epi:{}, st:{}, r:{}".format(total_step, i+1, episode_length, print_reward))

                # compute episode mean reward
                for id_ in intersection_id:
                    episode_reward[id_] /= args.num_step
                
                # save episode rewards
                for id_ in intersection_id:
                    episode_rewards[id_].append(episode_reward[id_])
                    episode_scores[id_].append(episode_score[id_])
                
                print_episode_reward = {'_'.join(k.split('_')[1:]):v for k, v in episode_reward.items()}
                print_episode_score = {'_'.join(k.split('_')[1:]):v for k, v in episode_score.items()}
                print('\n')
                print("Episode:{}, Mean reward:{}, Score: {}".format(i+1, print_episode_reward, print_episode_score))

                # save model
                if (i + 1) % args.save_freq == 0:
                    if args.algo == 'MDQN':
                        # Magent.save(model_dir + "/{}-ckpt".format(args.algo), i+1)
                        Magent.save(model_dir + "/{}-{}.h5".format(args.algo, i+1))
                        
                    # save reward to file
                    df = pd.DataFrame(episode_rewards)
                    df.to_csv(result_dir + '/rewards.csv', index=None)

                    df = pd.DataFrame(episode_scores)
                    df.to_csv(result_dir + '/scores.csv', index=None)

                    # save figure
                    plot_data_lists([episode_rewards[id_] for id_ in intersection_id], intersection_id, figure_name=result_dir + '/rewards.pdf')
                    plot_data_lists([episode_scores[id_] for id_ in intersection_id], intersection_id, figure_name=result_dir + '/scores.pdf')
        

    else: # inference
        Magent.load(args.ckpt)   
        
        episode_reward = {id_:[] for id_ in intersection_id} # for every agent
        episode_score = {id_:[] for id_ in intersection_id} # for everg agent

        state = env.get_state()
        for i in range(args.num_step): 
            action = Magent.choose_action(state) # index of action
            action_phase = {}

            for id_, a in action.items():
                action_phase[id_] = phase_list[id_][a]

            # one step #####  
            next_state, reward = env.step(action_phase) # one step
            score = env.get_score()

            for _ in range(args.phase_step-1):
                next_state, reward_ = env.step(action_phase)
                score_ = env.get_score()
                for id_ in intersection_id:
                    reward[id_] += reward_[id_]
                    score[id_] += score_[id_]

            for id_ in intersection_id:
                reward[id_] /= args.phase_step
                score[id_] /= args.phase_step
            # one step #####

            for id_ in intersection_id:
                episode_reward[id_].append(reward[id_])
                episode_score[id_].append(score[id_])
            
            state = next_state

            print("step:{}/{}, action:{}, reward:{}, score:{}"
                            .format(i+1, args.num_step, action, reward, score))
        
        mean_reward = {}
        mean_score = {}
        for id_ in intersection_id:
            mean_reward[id_] = np.mean(episode_reward[id_])
            mean_score[id_] = np.mean(episode_score[id_])
        print('\n')
        print("[Inference] Mean reward:{}, Mean score:{},".format(mean_reward, mean_score))

        # inf_result_dir = "result/" + args.ckpt.split("/")[1] 
        # df = pd.DataFrame(scores})
        # df.to_csv(inf_result_dir + '/inf_scores.csv', index=None) 
        # plot_data_lists([scores], ['inference scores'], figure_name=inf_result_dir + '/inf_scores.pdf')


if __name__ == '__main__':
    main()
