import numpy as np
import pickle
import tensorflow as tf
from ddpg.ddpg import ReplayBuffer


def movav(arr, w):
    return [sum(arr[i:i+w])/w for i in range(len(arr)-w)]


class Env:
    def __init__(self, std_0=.05, std_1=.1, std_2=.1, max_steps=100):
        self.max_steps = max_steps
        self.actions = np.array([[1,0,0],
                                 [0,1,0],
                                 [0,0,1]])
        self.state_range = (0., 150.)
        self.dx = 1.
        self.std_0 = std_0
        self.std_1 = std_1
        self.std_2 = std_2
        self.state_from = np.random.rand() * self.state_range[1]
        self.state_to = np.random.rand() * self.state_range[1]
        self.gs = 0

    def reset(self):
        self.gs = 0
        self.state_from = np.random.rand() * self.state_range[1]
        self.state_to = np.random.rand() * self.state_range[1]

    def set_manual_game(self, state_from, state_to):
        self.state_from = np.float32(state_from)
        self.state_to = np.float32(state_to)

    def step(self, action):
        self.gs += 1
        self.state_from += self.move_func(action)
        dist = self.state_from - self.state_to
        if np.abs(dist) <= self.dx or self.gs >= self.max_steps:
            ret = self.state_from, dist, 1
            self.reset()
        else:
            ret = self.state_from, dist, 0
        return ret

    def get_state(self):
        return self.state_from

    def get_target(self):
        return self.state_to

    def move_func(self, action):
        if action == 0:
            return np.random.randn()*self.std_0
        elif action == 1:
            return -.3 - np.exp(self.state_from/30.)/50. + np.random.randn()*self.std_1
            #return -3 + np.random.randn()*self.std_1
        elif action == 2:
            return .3 + np.exp((150. - self.state_from)/30.)/50. + np.random.randn()*self.std_2
            #return 3 + np.random.randn()*self.std_2
        else:
            raise ValueError('Cannot interp action', action, '\nPossible are: [0, 1, 2]')


class ModelNetwork:
    def __init__(self, state_dim=1, action_dim=3, out_dim=1, layer_sizes=(256,256), l2_rate=1e-4,
                 activations=('relu','relu','id'), max_buffer_len=500000, name='BoxModel',
                 savedir='home/user/Desktop/py/BoxModel/model/', logdir='home/user/Desktop/py/BoxModel/logs/'):
        self.name = name
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
            self._update_step = self.create_updater(l2_rate)

        self.sess.run(tf.global_variables_initializer())

        self.summary_writer = tf.summary.FileWriter(logdir, self.sess.graph, session=self.sess)
        self.saver = tf.train.Saver(self.net)
        self.merged_summary = tf.summary.merge_all()

        self.savedir = savedir + self.name
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
            for var in  vars:
                tf.summary.histogram(var.name[:-2], var)
            return state_inp, action_inp, l3, vars

    def create_updater(self, l2_rate):
        target = tf.placeholder(tf.float32, [None, self.out_dim], name='target_input')
        lr = tf.placeholder(tf.float32, name='lr')
        loss = tf.reduce_mean(tf.square(target - self.output))
        tf.summary.scalar('Loss', loss)
        l2 = 0.
        for var in tf.trainable_variables():
            l2 += tf.reduce_sum(tf.square(var)/2)
        updater = tf.train.AdamOptimizer(lr)
        update = updater.minimize(loss + l2*l2_rate)
        return target, lr, loss, update

    @staticmethod
    def _parse_activations(act):
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
        self.saver.save(self.sess, self.savedir)

    def load(self):
        self.saver.restore(self.sess, self.savedir)

    def predict(self, states, actions):
        return self.sess.run(self.output, {self.state_input:states,
                                           self.action_input:actions})

    def train(self, iters, batch_size, lr=1e-3, verbose=1000):
        if self.buffer.count == 0:
            raise ValueError('No data in buffer! Fill it with something!')
        for i in range(iters):
            data = self.buffer.sample(batch_size)
            states = np.array([d[0] for d in data])
            actions = np.array([d[1] for d in data])
            next_states = np.array([d[2] for d in data])

            self._one_train_step(states, actions, next_states, lr, verbose)

            self.gs += 1

    def _one_train_step(self, states, actions, target, lr, verbose):
        feed_dict = {self.state_input:states.reshape((-1,self.state_dim)),
                     self.action_input:actions.reshape((-1,self.action_dim)),
                     self.target_input:target.reshape((-1, self.state_dim)),
                     self.lr:lr}
        _, loss = self.sess.run([self._update_step, self._loss], feed_dict)
        self.losses.append(loss)
        if self.gs % verbose == verbose - 1:
            self.write_summary()
            print('GS {:6g}, Loss: {:.5f}'.format(self.gs+1, np.mean(self.losses[-verbose:])))

    def load_buffer(self, path):
        with open(path, 'wb') as f:
            self.buffer = pickle.load(f)

    def save_buffer(self, path):
        with open(path, 'wb') as f:
            pickle.dump(self.buffer, f)

    def add_observation(self, state, action, new_state):
        self.buffer.add(state, action, new_state)

    def write_summary(self):
        s = self.sess.run(self.merged_summary, feed_dict={self._loss:self.losses[-1]})
        self.summary_writer.add_summary(s, self.gs)


if __name__ == '__main__':
    import matplotlib.pyplot as plt
    env = Env(std_0=0.1, std_1=.2, std_2=.2)
    agent = ModelNetwork()
    state = env.get_state()

    for i in range(500000):
        action = np.random.randint(0,3)
        action_ohe = env.actions[action]

        new_state, reward, done = env.step(action)
        agent.add_observation([state], [action_ohe], [new_state])
        state = new_state
        if done:
            state = env.get_state()

    print('Buffer Ready!')

    plt.plot(np.array([d[0] for d in agent.buffer.buffer]), 'r-')
    plt.show()

    agent.train(60000, batch_size=16)

    st = np.arange(0.,150., 1e-1).reshape((-1,1))
    new_states0 = agent.predict(st, [env.actions[0]]*len(st))
    new_states1 = agent.predict(st, [env.actions[1]] * len(st))
    new_states2 = agent.predict(st, [env.actions[2]] * len(st))

    plt.plot(st, st - new_states0, label='0')
    plt.plot(st, st - new_states1, label='1')
    plt.plot(st, st - new_states2, label='2')
    plt.legend()
    plt.show()