#import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.layers import Dense
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import optimizers
import random
import numpy as np
from collections import deque
from scipy import special
from tensorflow.keras.callbacks import History
history = History()

'''
Deep Q_learning Agent. Initialize state size, action size,
discount factor, learning rate and exploration decay, number of hidden layer, layer size
'''
# Deep Q-learning Agent
class DQNAgent:
    def __init__(self, state_size=1, action_size=4, discount_factor=0, learning_rate=0.0001,
                expl_decay=0.9, nhl=1, sl_f=1, client_number=3):
        self.state_size = state_size            # state size
        self.action_size = action_size          # action size
        self.client_number = client_number
        self.memory_all = [0 for i in range(client_number)]
        # self.memory1 = deque(maxlen=500)         # size of memory
        # self.memory2 = deque(maxlen=500)
        # self.memory3 = deque(maxlen=500)
        for i in range(client_number):
            self.memory_all[i] = deque(maxlen=500)

        self.gamma = discount_factor   # discount rate
        self.epsilon = 1.0  # exploration rate
        self.epsilon_min = 0.1
        self.epsilon_decay = expl_decay             # exploration reduction factor
        self.learning_rate = learning_rate         # learning rate

        ### design of network  ###
        self.number_hidden_layers = nhl         # number of hidden layers
        self.size_layers = sl_f*self.state_size     # size of one layer

        ###  build networks ###
        self.model = self._build_model()        # build neural network
        #self.target_model = self._build_target_model()    # target model, same architecutre
        self.t_update_thresh = 10               # update factor tau
        self.t_update_count = self.t_update_thresh

        # Network performance
        self.loss = []
        self.loss_avg = 0
        # self.q_values = 0
        # self.q_val_arr

    def _build_model(self):
        '''
        Create network
        :return:
        '''
        # Neural Net for Deep-Q learning Model
        model = tf.keras.Sequential()
        model.add(tf.keras.layers.Dense(self.state_size, input_shape=[self.state_size], activation='relu'))
        for i in range(self.number_hidden_layers):
            model.add(tf.keras.layers.Dense(16, activation='relu'))

        # model.add(Dense(20, activation='relu'))
        # model.add(Dense(20, activation='relu'))
        # model.add(Dense(8, activation='relu'))
        # model.add(Dense(32, activation='tanh'))
        model.add(tf.keras.layers.Dense(self.action_size, activation='linear'))                         # linear
        # sgd = optimizers.SGD(lr=self.learning_rate, decay=self.epsilon_decay, momentum=0.9, nesterov=True)
        # model.compile(loss='mse', optimizer=sgd)
        #model.compile(loss=tf.keras.losses.MeanSquaredError(), optimizer=tf.keras.optimizers.SGD(lr=self.learning_rate))
        return model

    # def _build_target_model(self):
    #     '''
    #     Create network
    #     :return:
    #     '''
    #     # Neural Net for Deep-Q learning Model
    #     model = tf.keras.Sequential()
    #     model.add(tf.keras.layers.Dense(self.state_size, input_shape=[self.state_size], activation='relu'))
    #     for i in range(self.number_hidden_layers):
    #         model.add(tf.keras.layers.Dense(16, activation='relu'))
    #
    #     # model.add(Dense(20, activation='relu'))
    #     # model.add(Dense(20, activation='relu'))
    #     # model.add(Dense(8, activation='relu'))
    #     # model.add(Dense(32, activation='tanh'))
    #     model.add(tf.keras.layers.Dense(self.action_size, activation='linear'))                         # linear
    #     # sgd = optimizers.SGD(lr=self.learning_rate, decay=self.epsilon_decay, momentum=0.9, nesterov=True)
    #     # model.compile(loss='mse', optimizer=sgd)
    #     model.compile(loss=tf.keras.losses.MeanSquaredError(), optimizer=tf.keras.optimizers.Adam(lr=self.learning_rate))
    #     return model

    # def update_target_model(self):
    #     '''
    #     update target network
    #     :return:
    #     '''
    #     #model_weights = self.model.get_weights()
    #     model_weights = tf.keras.Model.get_weights(self.model)
    #     #self.target_model.set_weights(model_weights)
    #     tf.keras.Model.set_weights(self.target_model, model_weights)
    #     self.t_update_count = 0

    def remember(self, state, action, reward, next_state, c):
        '''
        save transitions
        :return:
        '''

        self.memory_all[c].append((state, action, reward, next_state))

        #print('memory_all[s]:', self.memory_all[s])
        #print('memory_len:', len(self.memory_all[c]))
        return

    # def act(self, state):
    #     '''
    #     with probability epsilon choose random action
    #     else: choose best known action
    #     :param state:
    #     :return:
    #     '''
    #     # if np.random.rand() <= self.epsilon:
    #     #     return random.randrange(self.action_size)
    #     # #act_values = self.model.predict(state)
    #     # act_values = tf.keras.Model.predict(self.model, state)
    #     # #print(act_values)
    #     # return np.argmax(act_values[0])  # returns action
    #     if np.random.rand() <= self.epsilon:
    #         return random.randrange(self.action_size)
    #     else:
    #         #act_values = self.model.predict(state)
    #         act_values = tf.keras.Model.predict(self.model, state)
    #         #print(act_values)
    #         return np.argmax(act_values[0])  # returns action
    #
    # def act_test(self, state):
    #     '''
    #     decide during testing phase solely on best known action
    #     :param state:
    #     :return:
    #     '''
    #     #act_values = self.model.predict(state)
    #     act_values = tf.keras.Model.predict(self.model, state)
    #     #print(act_values)
    #     return np.argmax(act_values[0])  # returns action

    # def replay(self, batch_size):
    #     '''
    #     train network
    #     :param batch_size:
    #     :return:
    #     '''
    #     minibatch = random.sample(self.memory, batch_size)      # load minibatch
    #     for state, action, reward, next_state in minibatch:
    #         target = reward + self.gamma * \
    #                    np.amax(self.target_model.predict(next_state)[0])        # compute target Q-value
    #         target_f = self.model.predict(state)                                # compute action for state
    #         target_f[0][action] = target                                        # set target value to best action
    #         self.model.fit(state, target_f, epochs=1, verbose=0, callbacks=[history])       # adjust netowrk
    #     self.t_update_count += 1
    #     self.loss = np.append(self.loss, history.history['loss'][0])
    #     self.loss_avg += history.history['loss'][0]
    #
    #     if self.t_update_count >= self.t_update_thresh:         # only copy to target network every tau steps
    #         self.update_target_model()
    #     if self.epsilon > self.epsilon_min:
    #         self.epsilon *= self.epsilon_decay
    # def replay(self, batch_size, s):
    #
    #     '''
    #     train network
    #     :param batch_size:
    #     :return:
    #     '''
    #
    #     train_data_state_s = []
    #     train_data_target_s = []
    #     minibatch = random.sample(self.memory_all[s], batch_size)  # load minibatch
    #     for state_f, action, reward, next_state in minibatch:
    #         # target = reward + self.gamma * \
    #         #          np.amax(self.target_model.predict(next_state)[0])  # compute target Q-value
    #         target = reward + self.gamma * \
    #                      np.amax(tf.keras.Model.predict(self.target_model, next_state)[0])  # compute target Q-value
    #         #target_f = self.model.predict(state_f)  # compute action for state
    #         target_f = tf.keras.Model.predict(self.model, state_f)
    #         target_f[0][action] = target  # set target value to best action
    #         # self.model.fit(state, target_f, epochs=1, verbose=0, callbacks=[history])       # adjust netowrk
    #
    #
    #         train_data_state_s = np.append(train_data_state_s, state_f)
    #         train_data_target_s = np.append(train_data_target_s, target_f)
    #         #print('train_data_state_s', train_data_state_s)
    #         #print('train_data_target_s', train_data_target_s)
    #
    #
    #     return train_data_state_s, train_data_target_s

    # def replaycontinue(self):
    #     self.t_update_count += 1
    #     # self.loss = np.append(self.loss, history.history['loss'][0])
    #     # self.loss_avg += history.history['loss'][0]
    #
    #     if self.t_update_count >= self.t_update_thresh:         # only copy to target network every tau steps
    #         self.update_target_model()
    #     if self.epsilon > self.epsilon_min:
    #         self.epsilon *= self.epsilon_decay

    # def save(self, path):               # save network weights
    #     # self.model.save_weights(name)
    #     self.model.save(path + '_weights.h5')
    #
    # def load(self, path):           # load network weights , needs to know architecture of network
    #     self.model.load_weights(path + '_weights.h5')
    #
    # def random_act(self):
    #     return random.randrange(self.action_size)