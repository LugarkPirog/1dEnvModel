import numpy as np
import pickle
import tensorflow as tf
from ddpg.ddpg import ReplayBuffer


class Env:
    def __init__(self):
        self.actions = np.array([[1,0,0],
                                 [0,1,0],
                                 [0,0,1]])
        self.state_range = (0., 150.)
        self.dx = 1.
        self.std_0 = .2
        self.std_1 = .9
        self.std_2 = .9
        self.state_from = np.random.rand() * self.state_range[1]
        self.state_to = np.random.rand() * self.state_range[1]

    def reset(self):
        self.state_from = np.random.rand() * self.state_range[1]
        self.state_to = np.random.rand() * self.state_range[1]

    def set_manual_game(self, state_from, state_to):
        self.state_from = np.float32(state_from)
        self.state_to = np.float32(state_to)

    def step(self, action):
        self.state_from += self.move_func(action)
        dist = self.state_from - self.state_to
        if np.abs(dist) <= self.dx:
            done = 1
        else:
            done = 0
        return self.state_from, dist, done

    def get_state(self):
        return self.state_from

    def get_target(self):
        return self.state_to

    def move_func(self, action):
        if action == 0:
            return np.random.randn()*self.std_0
        elif action == 1:
            return -3. + np.random.randn()*self.std_1
        elif action == 2:
            return 3. + np.random.randn() * self.std_2
        else:
            raise ValueError('Cannot interp action', action, '\nPossible are: [0, 1, 2]')


class ModelNetwork:
    def __init__(self, state_dim=1, action_dim=3, out_dim=1, layer_sizes=(256,256),
                 activations=('relu','relu','id'), max_buffer_len=500000,
                 print_loss_every=1000, name='EnvModel'):
        self.name = name
        self.print_loss_every = print_loss_every
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.out_dim = out_dim
        self.buffer = ReplayBuffer(max_buffer_len)
        self.sess = tf.Session(graph=tf.get_default_graph())

        self.state_input,\
            self.action_input,\
            self.output,\
            self.net = self.create_net(layer_sizes, activations)

        self.target_input,\
            self.lr,\
            self._loss,\
            self._update_step = self.create_updater()

        self.sess.run(tf.global_variables_initializer())
        self.gs = 0
        self.losses = []

    def create_net(self, layers, activations):
        assert len(layers) == len(activations) - 1, 'You should provide an activation for each layer!'
        with tf.variable_scope(self.name):
            state_inp = tf.placeholder(tf.float32, [None, self.state_dim], name='net_state_input')
            action_inp = tf.placeholder(tf.float32, [None, self.action_dim], name='net_action_input')
            with tf.name_scope('layer_1'):
                w_1_state = tf.get_variable('w_1_state', [self.state_dim, layers[0]])
                w_1_action = tf.get_variable('w_1_action', [self.action_dim, layers[0]])
                b_1 = tf.Variable(tf.zeros([layers[0], ]), name='b_1')
                l1 = self._parse_activations(activations[0])(
                        tf.matmul(state_inp, w_1_state) +
                        tf.matmul(action_inp, w_1_action) + b_1
                    )

            with tf.name_scope('layer_2'):
                w_2 = tf.get_variable('w_2', [layers[0], layers[1]])
                b_2 = tf.Variable(tf.zeros([layers[1], ]), name='b_2')
                l2 = self._parse_activations(activations[1])(tf.matmul(l1, w_2) + b_2)

            with tf.name_scope('layer_3'):
                w_3 = tf.get_variable('w_3', [layers[1], self.out_dim])
                b_3 = tf.Variable(tf.zeros([self.out_dim, ]), name='b_3')
                l3 = self._parse_activations(activations[2])(tf.matmul(l2, w_3) + b_3)

            vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, self.name)
            return state_inp, action_inp, l3, vars

    def create_updater(self):
        target = tf.placeholder(tf.float32, [None, self.out_dim], name='target_input')
        lr = tf.placeholder(tf.float32, name='lr')
        loss = tf.square(target - self.output)
        updater = tf.train.AdamOptimizer(lr)
        update = updater.minimize(loss)
        return target, lr, loss, update

    def create_target_net(self):
        pass

    def _parse_activations(self, act):
        if act == 'sigmoid':
            out = tf.nn.sigmoid
        elif act == 'tanh':
            out = tf.nn.tanh
        elif act == 'softmax':
            out = tf.nn.softmax
        elif act == 'elu':
            out = tf.nn.elu
        elif act == 'selu':
            out = tf.nn.selu
        elif act in ['linear', 'id', 'identity']:
            out = tf.identity
        else:
            out = tf.nn.relu
        return out

    def save(self):
        pass

    def load(self):
        pass

    def predict(self, states, actions):
        return self.sess.run(self.output, {self.state_input:states,
                                           self.action_input:actions})

    def train(self, iters, batch_size):
        if self.buffer.count == 0:
            raise ValueError('No data in buffer! Fulfill it somehow!')
        for i in range(iters):
            data = np.array(self.buffer.sample(batch_size))
            states = data[:,0]
            actions = data[:,1]
            next_states = data[:,2]

            self._one_train_step(states, actions, next_states)

            self.gs += 1

    def _one_train_step(self, states, actions, target):
        feed_dict = {self.state_input:states,
                     self.action_input:actions,
                     self.target_input:target}
        _, loss = self.sess.run([self._update_step, self._loss, feed_dict])
        self.losses.append(loss)
        if self.gs % self.print_loss_every == self.print_loss_every - 1:
            print('GS {:6g}, Loss: {:.5f}'.format(self.gs+1, np.mean(self.losses[-100:])))

    def load_buffer(self, path):
        with open(path, 'wb') as f:
            self.buffer = pickle.load(f)

    def add_observation(self, state, action, new_state):
        self.buffer.add(state, action, new_state)


if __name__ == '__main__':
    env = Env()
    agent = ModelNetwork()
    state = env.get_state()

    for i in range(3):
        action = np.random.randint(0,3)
        action_ohe = env.actions[action]

        new_state, reward, done = env.step(action)
        agent.add_observation([state], [action_ohe], [new_state])
        state = new_state

    print(agent.buffer.buffer)
