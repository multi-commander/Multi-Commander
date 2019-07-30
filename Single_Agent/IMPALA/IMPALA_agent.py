import core
import tensorflow as tf
import numpy as np
import agent_config
import vtrace


class IMPALAAgent:
    def __init__(self, sess, name, unroll, state_shape, output_size, activation, final_activation, hidden, coef,
                 reward_clip):
        self.sess = sess
        self.state_shape = state_shape
        self.output_size = output_size
        self.activation = activation
        self.final_activation = final_activation
        self.hidden = hidden
        self.clip_rho_threshold = 1.0
        self.clip_pg_rho_threshold = 1.0
        self.discount_factor = 0.99
        self.lr = 0.001
        self.unroll = unroll
        self.trajectory_size = unroll + 1
        self.coef = coef
        self.reward_clip = reward_clip

        self.s_ph = tf.placeholder(tf.float32, shape=[None, self.unroll, *self.state_shape])
        self.ns_ph = tf.placeholder(tf.float32, shape=[None, self.unroll, *self.state_shape])
        self.a_ph = tf.placeholder(tf.int32, shape=[None, self.unroll])
        self.d_ph = tf.placeholder(tf.bool, shape=[None, self.unroll])
        self.behavior_policy = tf.placeholder(tf.float32, shape=[None, self.unroll, self.output_size])
        self.r_ph = tf.placeholder(tf.float32, shape=[None, self.unroll])

        if self.reward_clip == 'tanh':
            squeezed = tf.tanh(self.r_ph / 5.0)
            self.clipped_rewards = tf.where(self.r_ph < 0, .3 * squeezed, squeezed) * 5.
        elif self.reward_clip == 'abs_one':
            self.clipped_rewards = tf.clip_by_value(self.r_ph, -1.0, 1.0)
        elif self.reward_clip == 'no_clip':
            self.clipped_rewards = self.r_ph

        self.discounts = tf.to_float(~self.d_ph) * self.discount_factor

        self.policy, self.value, self.next_value = core.build_model(
            self.s_ph, self.ns_ph, self.hidden, self.activation, self.output_size,
            self.final_activation, self.state_shape, self.unroll, name)

        self.transpose_vs, self.transpose_clipped_rho = vtrace.from_softmax(
            behavior_policy_softmax=self.behavior_policy,
            target_policy_softmax=self.policy,
            actions=self.a_ph, discounts=self.discounts, rewards=self.clipped_rewards,
            values=self.value, next_value=self.next_value, action_size=self.output_size)

        self.vs = tf.transpose(self.transpose_vs, perm=[1, 0])
        self.rho = tf.transpose(self.transpose_clipped_rho, perm=[1, 0])

        self.vs_ph = tf.placeholder(tf.float32, shape=[None, self.unroll])
        self.pg_advantage_ph = tf.placeholder(tf.float32, shape=[None, self.unroll])

        self.value_loss = vtrace.compute_value_loss(self.vs_ph, self.value)
        self.entropy = vtrace.compute_entropy_loss(self.policy)
        self.pi_loss = vtrace.compute_policy_loss(self.policy, self.a_ph, self.pg_advantage_ph, self.output_size)

        self.total_loss = self.pi_loss + self.value_loss + self.entropy * self.coef
        self.optimizer = tf.train.RMSPropOptimizer(self.lr, epsilon=0.01, momentum=0.0, decay=0.99)
        self.train_op = self.optimizer.minimize(self.total_loss)

    def train(self, state, next_state, reward, done, action, behavior_policy):
        unrolled_state = np.stack(
            [state[i:i + self.trajectory_size] for i in range(len(state) - self.trajectory_size + 1)])
        unrolled_next_state = np.stack(
            [next_state[i:i + self.trajectory_size] for i in range(len(state) - self.trajectory_size + 1)])
        unrolled_reward = np.stack(
            [reward[i:i + self.trajectory_size] for i in range(len(state) - self.trajectory_size + 1)])
        unrolled_done = np.stack(
            [done[i:i + self.trajectory_size] for i in range(len(state) - self.trajectory_size + 1)])
        unrolled_behavior_policy = np.stack(
            [behavior_policy[i:i + self.trajectory_size] for i in range(len(state) - self.trajectory_size + 1)])
        unrolled_action = np.stack(
            [action[i:i + self.trajectory_size] for i in range(len(state) - self.trajectory_size + 1)])

        unrolled_length = len(unrolled_state)
        sampled_range = np.arange(unrolled_length)
        np.random.shuffle(sampled_range)
        shuffled_idx = sampled_range[:32]

        # get vs_plus_1
        s_ph = np.stack([unrolled_state[i, 1:] for i in shuffled_idx])
        ns_ph = np.stack([unrolled_next_state[i, 1:] for i in shuffled_idx])
        r_ph = np.stack([unrolled_reward[i, 1:] for i in shuffled_idx])
        d_ph = np.stack([unrolled_done[i, 1:] for i in shuffled_idx])
        b_ph = np.stack([unrolled_behavior_policy[i, 1:] for i in shuffled_idx])
        a_ph = np.stack([unrolled_action[i, 1:] for i in shuffled_idx])

        feed_dict = {
            self.s_ph: s_ph,
            self.ns_ph: ns_ph,
            self.r_ph: r_ph,
            self.d_ph: d_ph,
            self.a_ph: a_ph,
            self.behavior_policy: b_ph}

        vs_plus_1 = self.sess.run(
            self.vs,
            feed_dict)

        # get vs
        s_ph = np.stack([unrolled_state[i, :-1] for i in shuffled_idx])
        ns_ph = np.stack([unrolled_next_state[i, :-1] for i in shuffled_idx])
        r_ph = np.stack([unrolled_reward[i, :-1] for i in shuffled_idx])
        d_ph = np.stack([unrolled_done[i, :-1] for i in shuffled_idx])
        b_ph = np.stack([unrolled_behavior_policy[i, :-1] for i in shuffled_idx])
        a_ph = np.stack([unrolled_action[i, :-1] for i in shuffled_idx])

        feed_dict = {
            self.s_ph: s_ph,
            self.ns_ph: ns_ph,
            self.r_ph: r_ph,
            self.d_ph: d_ph,
            self.a_ph: a_ph,
            self.behavior_policy: b_ph}

        vs, rho, value = self.sess.run(
            [self.vs, self.rho, self.value],
            feed_dict)

        pg_advantage = rho * (r_ph + self.discount_factor * (1 - d_ph) * vs_plus_1 - value)

        feed_dict = {
            self.s_ph: s_ph,
            self.ns_ph: ns_ph,
            self.r_ph: r_ph,
            self.d_ph: d_ph,
            self.a_ph: a_ph,
            self.behavior_policy: b_ph,
            self.vs_ph: vs,
            self.pg_advantage_ph: pg_advantage}

        pi_loss, value_loss, entropy, _ = self.sess.run(
            [self.pi_loss, self.value_loss, self.entropy, self.train_op],
            feed_dict)

        return pi_loss, value_loss, entropy

    def variable_to_network(self, variable):
        feed_dict = {i: j for i, j in zip(self.from_list, variable)}
        self.sess.run(self.write_main_parameter, feed_dict=feed_dict)

    def get_parameter(self):
        variable = self.sess.run(self.variable)
        return variable

    def get_policy_and_action(self, state):
        state = [state for i in range(self.unroll)]
        policy = self.sess.run(self.policy, feed_dict={self.s_ph: [state]})
        policy = policy[0][0]
        action = np.random.choice(self.output_size, p=policy)
        return action, policy, max(policy)

    def test(self, state, action, reward, done, behavior_policy):
        feed_dict = {
            self.s_ph: state,
            self.a_ph: action,
            self.d_ph: done,
            self.behavior_policy: behavior_policy,
            self.r_ph: reward}