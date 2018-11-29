import gym
import numpy as np
import tensorflow as tf
import keras
from keras import Sequential, Model
from keras.layers import Dense, LSTM, Input, concatenate, Reshape, Flatten


class Agent():
    def __init__(self, input_space, output_space):
        self.input_space = input_space
        self.output_space = output_space

        self.input = Input(shape=(4,), batch_shape=(2, 4))
        self.x = Dense(units=16, activation='tanh')(self.input)
        # self.x = Reshape((1, 16))(self.x)
        # self.x = LSTM(4, stateful=True)(self.x)
        # self.x = Reshape((1,4))(self.x)
        self.action = Dense(units=2, activation="relu")(self.x)
        self.model = Model(inputs=self.input, outputs=self.action)
        self.model.compile(optimizer='adam', loss='mean_squared_error')

    def get_action(self, s):
        return self.model.predict(s.reshape(1, -1))

    def train(self, s, t):
        self.model.fit(s.reshape((2, -1)), t.reshape((2, -1)),verbose=False)


def exp_rate(factor, base=1.1, upper=1, lower=0.1, param={'eps_w': 0.01, 'rAll_w': 1}, function=float.__pow__):
    eps, rAll = factor
    eps_w = param['eps_w']
    rAll_w = param['rAll_w']
    if function is float.__pow__:
        return max(upper * function(base, -(eps * eps_w + rAll * rAll_w)), lower)


if __name__ == '__main__':
    DISCOUNT = 0.9
    Noise0 = 1
    env = gym.make('CartPole-v0')
    agent = Agent(env.observation_space, env.action_space)

    eps = 15000
    rAll = 0
    replay = []
    for i in range(eps):
        done = False
        rEps = 0
        j = 0
        s = np.array(env.reset())
        while not done:
            j += 1
            action_predict = agent.get_action(s)[0]
            action = np.argmax(action_predict)

            #noise = exp_rate((i, rAll), 1.25, 1, 0.1, param={'eps_w': 0.01, 'rAll_w': 1})
            a_noise = np.random.rand() <= 0.5
            action = a_noise * np.random.randint(0, env.action_space.n) + (1 - a_noise) * action
            print(action_predict)
            obs, r, done, info = env.step(action)
            replay.append((s, action, action_predict, r, obs))

            rEps += r

            if j % 5 == 4:
                s_train, action_train, action_predict_train, r_train, obs_train = replay.pop(
                    np.random.randint(0, len(replay)))
                target = action_predict_train
                target[action_train] = r_train + DISCOUNT * np.amax(agent.get_action(obs_train)[0])
                target = target.reshape((1, -1))

                s_train2, action_train, action_predict_train, r_train, obs_train = replay.pop(
                    np.random.randint(0, len(replay)))
                target2 = action_predict_train
                target2[action_train] = r_train + DISCOUNT * np.amax(agent.get_action(obs_train)[0])
                target2 = target2.reshape((1, -1))

                s_train = np.stack((s_train, s_train2), axis=0)
                target = np.concatenate((target, target2), axis=0)
                agent.train(s_train, target)
            s = obs
        print("Eps: {}, reward: {}".format(i, rEps))
        rAll += rEps
