import tensorflow as tf
import numpy as np
import threading
import impala
import config
import multiprocessing
import utils
import tensorboardX
import numpy
import gym
import copy


class Agent(threading.Thread):
    def __init__(self, session, coord, name, global_network, reward_clip, lock):
        super(Agent, self).__init__()
        self.lock = lock
        self.sess = session
        self.coord = coord
        self.name = name
        self.global_network = global_network
        self.local_network = impala.IMPALA(
            sess=self.sess,
            name=name,
            unroll=config.unroll,
            state_shape=config.state_shape,
            output_size=config.output_size,
            activation=config.activation,
            final_activation=config.final_activation,
            hidden=config.hidden,
            coef=config.entropy_coef,
            reward_clip=reward_clip
        )
        self.global_to_local = utils.copy_src_to_dst('global', name)

    def run(self):
        self.sess.run(self.global_to_local)
        self.env = gym.make('PongDeterministic-v4')
        if self.name == 'thread_0':
            self.env = gym.wrappers.Monitor(self.env, 'save-mov',
                                            video_callable=lambda episode_id: episode_id % 10 == 0)

        done = False
        frame = self.env.reset()
        frame = utils.pipeline(frame)
        history = np.stack((frame, frame, frame, frame), axis=2)
        state = copy.deepcopy(history)
        episode = 0
        score = 0
        episode_step = 0
        total_max_prob = 0
        loss_step = 0

        writer = tensorboardX.SummaryWriter('runs/' + self.name)

        while True:
            loss_step += 1
            episode_state = []
            episode_next_state = []
            episode_reward = []
            episode_done = []
            episode_action = []
            episode_behavior_policy = []
            for i in range(128):
                action, behavior_policy, max_prob = self.local_network.get_policy_and_action(state)

                episode_step += 1
                total_max_prob += max_prob

                frame, reward, done, _ = self.env.step(action + 1)
                frame = utils.pipeline(frame)
                history[:, :, :-1] = history[:, :, 1:]
                history[:, :, -1] = frame
                next_state = copy.deepcopy(history)

                score += reward

                d = False
                if reward == 1 or reward == -1:
                    d = True

                episode_state.append(state)
                episode_next_state.append(next_state)
                episode_reward.append(reward)
                episode_done.append(d)
                episode_action.append(action)
                episode_behavior_policy.append(behavior_policy)

                state = next_state

                if done:
                    print(self.name, episode, score, total_max_prob / episode_step, episode_step)
                    writer.add_scalar('score', score, episode)
                    writer.add_scalar('max_prob', total_max_prob / episode_step, episode)
                    writer.add_scalar('episode_step', episode_step, episode)
                    episode_step = 0
                    total_max_prob = 0
                    episode += 1
                    score = 0
                    done = False
                    if self.name == 'thread_0':
                        self.env.close()
                    frame = self.env.reset()
                    frame = utils.pipeline(frame)
                    history = np.stack((frame, frame, frame, frame), axis=2)
                    state = copy.deepcopy(history)

            pi_loss, value_loss, entropy = self.global_network.train(
                state=np.stack(episode_state),
                next_state=np.stack(episode_next_state),
                reward=np.stack(episode_reward),
                done=np.stack(episode_done),
                action=np.stack(episode_action),
                behavior_policy=np.stack(episode_behavior_policy))
            self.sess.run(self.global_to_local)
            writer.add_scalar('pi_loss', pi_loss, loss_step)
            writer.add_scalar('value_loss', value_loss, loss_step)
            writer.add_scalar('entropy', entropy, loss_step)