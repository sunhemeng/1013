import collections

import tensorflow as tf
import tensorflow_federated as tff
from keras.layers import Dense
from keras.models import Sequential
from keras.optimizers import Adam
from keras import optimizers
import random
import numpy as np
from collections import deque
from scipy import special
from keras.callbacks import History
history = History()


from fl_DQN import DQNAgent
from fl_environment_3server import Simulation


tf.compat.v1.enable_v2_behavior()

tff.framework.set_default_context(tff.test.ReferenceExecutor())



#dataset
NUM_CLIENTS = 1
NUM_EPOCHS = 150
episodes = 100
testing_number = 1000
test_sever_number = 1

hist_timeslots = 1
avg_SNR = 10
sim_env = Simulation(number_of_servers=3, number_of_users=1, historic_time=hist_timeslots,
                         snr_set=avg_SNR, csi=1, channel=0.99)



#model

state_size = ((sim_env.features-sim_env.CSI) * (sim_env.W - 1 + sim_env.CSI) * sim_env.S)
action_size = len(sim_env.action)
batch_size = 16

QN = DQNAgent(state_size, action_size, discount_factor=0, learning_rate=0.0001, expl_decay=0.9, nhl=3, sl_f=1, client_number=NUM_CLIENTS)

# Input_spec = collections.OrderedDict(
#             x=tf.TensorSpec([state_size], tf.float32),
#             y=tf.TensorSpec([action_size], tf.float32))
# Input_type = tff.to_type(Input_spec)

def model_fn():
    model = QN._build_model()
    return tff.learning.from_keras_model(model,
                                         #input_spec=tff.simulation.datasets.emnist.element_spec,
                                         #input_spec=tf.nest.map_structure(lambda x: tf.TensorSpec(x.shape, x.dtype),
                                         #                                 data),
                                         input_spec=collections.OrderedDict(
                                             x=tf.TensorSpec([None, state_size], tf.float32),
                                             y=tf.TensorSpec([action_size], tf.float32)),
                                         loss=tf.keras.losses.MeanSquaredError())





#training process

iterative_process = tff.learning.build_federated_averaging_process(
    model_fn,
    client_optimizer_fn=lambda: tf.keras.optimizers.Adam(learning_rate=0.0001),
    server_optimizer_fn=lambda: tf.keras.optimizers.Adam(learning_rate=0.0001))



# @tff.federated_computation(tff.SequenceType(tf.float32),tf.int32)
# def replay(state, batch_size):
#     '''
#     train network
#     :param batch_size:
#     :return:
#     '''
#     minibatch = random.sample(QN.memory, batch_size)  # load minibatch
#     for state_f, action, reward, next_state in minibatch:
#         target = reward + QN.gamma * \
#                  np.amax(QN.target_model.predict(next_state)[0])  # compute target Q-value
#         target_f = QN.model.predict(state_f)  # compute action for state
#         target_f[0][action] = target  # set target value to best action
#         # self.model.fit(state, target_f, epochs=1, verbose=0, callbacks=[history])       # adjust netowrk
#         federated_train_data = [state_f, target_f]
#         print('federated_train_data', federated_train_data)
#         state, metrics = iterative_process.next(state, federated_train_data)  # 1model
#         print('state:', state)
#         print('metrics:', metrics)
#     QN.t_update_count += 1
#     QN.loss = np.append(QN.loss, history.history['loss'][0])
#     QN.loss_avg += history.history['loss'][0]
#     print('loss_avg:', QN.loss_avg)
#
#     if QN.t_update_count >= QN.t_update_thresh:  # only copy to target network every tau steps
#         QN.update_target_model()
#     if QN.epsilon > QN.epsilon_min:
#         QN.epsilon *= QN.epsilon_decay

#tf.keras.Model.compile(QN.target_model, loss=tf.keras.losses.MeanSquaredError(), optimizer=tf.keras.optimizers.SGD(lr=0.0001))


state = iterative_process.initialize()     #sever state

#states = sim_env.state
#print('states:', states)


loss_all = []
loss_avg = []
error_avg = [0 for i in range(test_sever_number)]
#err = [0 for i in range(test_sever_number)]
metric = []
error = [0 for i in range(test_sever_number)]

eval_model = None
pre_model = None
target_model = None

for e in range(episodes):
    sim_env.reset()
    states1 = sim_env.state
    loss_sum = 0

    for round_num in range(NUM_EPOCHS):
        train_data_state = []
        train_data_target = []

        pre_model = QN._build_model()
        pre_model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),
                           loss=tf.keras.losses.MeanSquaredError(),
                           )

        tff.learning.assign_weights_to_keras_model(pre_model, state.model)


        target_model = QN._build_model()
        target_model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),
                          loss=tf.keras.losses.MeanSquaredError(),
                          )



        # 1st server
        #print('states1:', states1)
        #action = QN.act(states1)
        if np.random.rand() <= QN.epsilon:
            action = random.randrange(QN.action_size)
        else:
            act_values = pre_model.predict(states1)
            #act_values = tf.keras.Model.predict(state.model, states1)
            #print(act_values)
            action = np.argmax(act_values[0])  # returns action
        #print('action:', action)
        next_state, rewards, overall_err = sim_env.Assign_Cores(action, round_num, 1)
        #print('overall_err', overall_err)
        #print('rewards', rewards)
                # next_state = np.reshape(next_state, (1, state_size))
        QN.remember(states1, action, rewards, next_state, 0)
                # memory_dataset = np.concatenate((memory_dataset, memory_s), axis=0)
        states1 = next_state
        if len(QN.memory_all[0]) > batch_size:
            #train_data_state_s, train_data_target_s = QN.replay(batch_size, 0)

            train_data_state_1 = []
            train_data_target_1 = []
            minibatch = random.sample(QN.memory_all[0], batch_size)  # load minibatch
            for state_f, action, reward, next_state in minibatch:
                # target = reward + self.gamma * \
                #          np.amax(self.target_model.predict(next_state)[0])  # compute target Q-value
                target = reward + QN.gamma * \
                         np.amax(target_model.predict(next_state)[0])  # compute target Q-value
                # target_f = self.model.predict(state_f)  # compute action for state
                target_f = pre_model.predict(state_f)
                target_f[0][action] = target  # set target value to best action
                # self.model.fit(state, target_f, epochs=1, verbose=0, callbacks=[history])       # adjust netowrk

                train_data_state_1 = np.append(train_data_state_1, state_f)
                train_data_target_1 = np.append(train_data_target_1, target_f)
                # print('train_data_state_s', train_data_state_s)
                # print('train_data_target_s', train_data_target_s)


            train_data_state = np.concatenate((train_data_state, train_data_state_1), axis=0)
                    # print('train_data_state',train_data_state)
            train_data_target = np.concatenate((train_data_target, train_data_target_1), axis=0)
                    # print('train_data_target', train_data_target)






        if len(QN.memory_all[0]) > batch_size:
            train_data_state = np.reshape(train_data_state, (NUM_CLIENTS, state_size*batch_size))
            train_data_target = np.reshape(train_data_target, (NUM_CLIENTS, action_size*batch_size))
            #memory_dataset = np.reshape(memory_dataset, (NUM_CLIENTS, -1))
            #print('train_data_state', train_data_state)
            #print('train_data_target', train_data_target)
            federated_train_data = [[0 for j in range(batch_size)] for i in range(NUM_CLIENTS)]
            for i in range(NUM_CLIENTS):
                for j in range(batch_size):
                    federated_train_data[i][j] = {
                                                 'x':
                                                    np.array([train_data_state[i][(j*state_size):(j*state_size+state_size)]], dtype=np.float32),
                                                 'y':
                                                    np.array(train_data_target[i][(j*action_size):(j*action_size+action_size)], dtype=np.float32)}

            #print('federated_train_data', federated_train_data)
            state, metrics = iterative_process.next(state, federated_train_data)  # 1model
            # print('state:', state)
            #print('metrics:', metrics)
            #metric = np.append(metric, metrics)
            # print('loss', loss)
            # loss_all = np.append(loss_all, loss)
            loss_all = np.append(loss_all, metrics['train']['loss'])
            loss_sum += metrics['train']['loss']

            #QN.replaycontinue()
            QN.t_update_count += 1
            # self.loss = np.append(self.loss, history.history['loss'][0])
            # self.loss_avg += history.history['loss'][0]

            if QN.t_update_count >= QN.t_update_thresh:  # only copy to target network every tau steps
                #self.update_target_model()
                tff.learning.assign_weights_to_keras_model(target_model, state.model)
                # # model_weights = self.model.get_weights()
                # model_weights = tf.keras.Model.get_weights(self.model)
                # # self.target_model.set_weights(model_weights)
                # tf.keras.Model.set_weights(self.target_model, model_weights)
                QN.t_update_count = 0
            if QN.epsilon > QN.epsilon_min:
                QN.epsilon *= QN.epsilon_decay
                # QN.t_update_count += 1
                # QN.loss = np.append(QN.loss, history.history['loss'][0])
                # QN.loss_avg += history.history['loss'][0]
                # print('loss_avg:', QN.loss_avg)
                #
                # if QN.t_update_count >= QN.t_update_thresh:  # only copy to target network every tau steps
                #     QN.update_target_model()
                # if QN.epsilon > QN.epsilon_min:
                #     QN.epsilon *= QN.epsilon_decay


    #testing process

    # hist_timeslots = 1
    # avg_SNR = 10
    # # sim_env = Simulation(number_of_servers=1, number_of_users=1, historic_time=hist_timeslots,
    # #                          snr_set=avg_SNR, csi=1, channel=0.99)

    loss_avg = np.append(loss_avg, loss_sum / NUM_EPOCHS)

    eval_model = QN._build_model()
    eval_model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),
                       loss=tf.keras.losses.MeanSquaredError(),
                       )

    tff.learning.assign_weights_to_keras_model(eval_model, state.model)

    #states = [0 for i in range(test_sever_number)]
    sim_env.reset()

    states_test_1 = sim_env.state

    # for s in range(test_sever_number):
    #     states[s] = sim_env.state

    for i in range(testing_number):
        #err = [0 for i in range(test_sever_number)]
        # first sever choose workload first
        states_test_1 = np.reshape(states_test_1, (1, state_size))
        #print('states_test_1', states_test_1)
        #action = QN.act_test(states_test_1)

        #act_values = tf.keras.Model.predict(eval_model, states_test_1)
        act_values = eval_model.predict(states_test_1)
        action = np.argmax(act_values[0])

        #print('action', action)
        next_state, rewards, overall_err = sim_env.Assign_Cores(action, i, 0)
        #print('overall_err', overall_err)
        #print('rewards', rewards)
        next_state = np.reshape(next_state, [1, state_size])
        states_test_1 = next_state
        error[0] = np.append(error[0], overall_err)
        # states[0] = np.append(states[0], next_state)
        # states[0] = np.reshape(states[0], [-1, state_size])
        #print('states0', states[0])
        #err[0] += - np.log10(overall_err)






    print('##################test####################### episode %d' % (e))
        # action =
        # print('action', action)
        # next_state, rewards, overall_err, c3 = sim_env.Assign_Cores_test(action, i, 2, workload3)
        # next_state = np.reshape(next_state, [1, state_size])
        #states[2] = np.append(states[1], next_state)
    #for s in range(test_sever_number):
    error_avg[0] = np.append(error_avg[0], np.power(10, -sim_env.error/testing_number))

    print(error_avg)
        # for s in range(test_sever_number):
        #     action = QN.act_test(states)
        #     print('action', action)
        #     next_state, rewards, overall_err = sim_env.Assign_Cores(action, i, 0, s)
        #     error[s] = np.append(error[s], overall_err)
        #     #next_state = np.reshape(next_state, [1, state_size])
        #     states = next_state


    #error_avg = np.append(error_avg, np.power(10, -sim_env.error / testing))
parameters = '_DQN__S{}_rho{}_SNR{}_PS{}_W{}_lr{}_df{}_sl{}_nhl{}_ef{}_7'. \
    format(sim_env.S, sim_env.p, sim_env.SNR_avg[0], sim_env.pi, sim_env.W, QN.learning_rate,
           QN.gamma, QN.size_layers, QN.number_hidden_layers, QN.epsilon_decay)

np.savetxt(sim_env.channel_type + '_Error' + parameters + '.csv', np.transpose(error), header='error', fmt='0%13.11f')
np.savetxt('error_avg.csv', np.transpose(error_avg), header='Error[sum(-log10(e))]', fmt='0%30.28f')
    #np.savetxt('metric.csv', np.transpose(metric), header='metric', fmt='0%13.11f')
    #np.savetxt('loss.csv', np.transpose(loss_all), header='loss', fmt='0%13.11f')
np.savetxt('loss.csv', np.transpose(loss_all), header='loss', fmt='0%13.11f')
np.savetxt('loss_avg.csv', np.transpose(loss_avg), header='loss_avg', fmt='0%13.11f')