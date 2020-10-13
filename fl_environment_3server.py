#import matplotlib.pyplot as plt
import random
import numpy as np
from collections import deque
from scipy import special
from scipy.stats import rayleigh
from itertools import product
#import Sim_Optimal_Offloading_V2

#from Sim_Optimal_Offloading_V2 import Optimal_Offloading

'''
Simulation Environment for partitioned workload. Needs to include the DDPGAgent and OUNoise. 
'''

class Simulation:
    def __init__(self, number_of_servers=3, number_of_users=1, historic_time=1, snr_set=10, csi=1, channel=0.99):  # initialize number of servers, number of users,
                                                                                                                  # historic timeslots, avg_snr, csi knowledge, channel correlation
        # define Simulation parameters
        self.S = number_of_servers  # number of servers
        self.N = number_of_users  # number of users
        self.W = historic_time     # historic time slots
        self.features = 2  # number of feature(SNR and workload in state)
        self.action = self.compute_action_space()  # action set
        self.server_selection_size = len(self.action)
        self.CSI = csi        # 0 = information of previous time slot, 1 = perfect CSI
        self.p = channel      # channel correlation between time slots

        print(self.action)

        # Communication & Computation constants
        # self.area_width = 50
        # self.area_length = 50
        # self.poisson_variable = np.random.poisson(0.5 * self.area_width, 3)
        # self.A = self.area_length * self.area_width  # area in square meter
        # self.B = 5 * 10 ** 6  # bandwidth
        # self.F = 2.4 * 10 ** 9  # carrier frequency
        # self.r = []
        # self.r = [110, 110, 110, 20, 20, 20, 20, 20, 20, 20, 20, 20]  # distance between UE and server
        # self.No_dBm = -174  # noise power level in dbm/Hz
        # self.No_dB = self.No_dBm - 30
        self.TS = 0.025 * 10 ** -3  # symbol time 0.025 *10**-3
        # self.P_dBm = 20  # in dbm
        # self.P_dB = self.P_dBm - 30
        self.pi = 320  # bits
        self.co = 24 * 10 ** 6  # total workload in cycles 24
        self.f = 3 * 10 ** 9  # computation power
        self.lam = 3 * 10 ** 6  # poisson arriving rate
        self.xi = -0.0214  # shape parameter
        self.sigma = 3.4955 * 10 ** 6  # scale parameter
        self.d = 2.0384 * 10 ** 7  # threshold
        self.T = 0.025  # delay tolerance in s 0.025
        self.eps_max = 0.001
        self.threshold_computation = 0.999  # computation threshold
        self.comp_correlation = 4*10**6    # computation correlation

        # Feedback
        self.reward = 0
        self.error = 0
        # self.rnd_channel = np.random.seed(2236391566)

        #print(np.random.get_state())

        self.channel_type = self.channel_type_name()        # Create Channel type string for saving data in right path
        print(self.channel_type)

        self.h = []             # channel gain
        for i in range(self.S):
            self.h = np.append(self.h, 1)
        #     self.h = np.append(self.h, 1)


        self.channel_sequence = self.create_random_channel()        # create fixed channel sequence
        print('random:', self.channel_sequence[0][0])

        self.chh = []

        ### Initialize SNR array ###
        self.SNR_s = []                                 # initialize SNR of channel
        self.SNR_avg = [np.power(10, snr_set/10)]*self.S
        self.SNR = []
        for i in range(self.S):
            self.SNR_s = np.append(self.SNR_s, self.SNR_avg[i])         # set average snr
        for i in range(self.W):                                         # initialize whole array
            self.SNR = np.concatenate((self.SNR, self.SNR_s), axis=0)
        self.SNR = self.SNR.reshape((self.W, self.S))
        print(self.SNR)

        ### Initiliaze previous task sizes ###
        self.tasks_prev = []        # previous task size
        self.tasks_prev_s = []
        for i in range(self.S):
            self.tasks_prev_s = np.append(self.tasks_prev_s, 0)
        for i in range(self.W - 1 + self.CSI):
            self.tasks_prev = np.concatenate((self.tasks_prev, self.tasks_prev_s), axis=0)      # initialize whole array
        self.tasks_prev = self.tasks_prev.reshape((self.W - 1 + self.CSI), self.S)
        print(self.tasks_prev)

        # Initialize state, concatenate snr + previous tasks sizes dependent on knowledge of environment
        if self.CSI == 1:       # perfect CSI
            self.state = np.log10(self.SNR) / 10    # initialize state
           # self.state = np.log10(self.SNR)
        else:                   # outdated CSI
            self.state = np.concatenate((np.log10(self.SNR[:-1])/10, self.tasks_prev/(10**9)), axis=0)    # initialize state

        print(self.state)

    def reset(self):
        '''
        resets reward, error variables to zero. Sets SNR and workload back.
        :return:
        '''
        # Feedback
        self.reward = 0
        self.error = 0

        self.h = []             # channel gain
        for i in range(self.S):
            self.h = np.append(self.h, 1)

        ### Initialize SNR array ###
        self.SNR_s = []  # initialize SNR of channel
        self.SNR = []
        for i in range(self.S):
            self.SNR_s = np.append(self.SNR_s, self.SNR_avg[i])
        for i in range(self.W):
            self.SNR = np.concatenate((self.SNR, self.SNR_s), axis=0)
        self.SNR = self.SNR.reshape((self.W, self.S))

        self.tasks_prev = []  # previous task size
        self.tasks_prev_s = []
        for i in range(self.S):
            self.tasks_prev_s = np.append(self.tasks_prev_s, 0)
        for i in range(self.W - 1 + self.CSI):
            self.tasks_prev = np.concatenate((self.tasks_prev, self.tasks_prev_s), axis=0)
        self.tasks_prev = self.tasks_prev.reshape((self.W - 1 + self.CSI, self.S))
        # print(self.tasks_prev)

        ### Initialize state, concatenate snr + previous tasks sizes ###
        if self.CSI == 1:       # perfect CSI
            self.state = np.log10(self.SNR) / 10
           # self.state = np.log10(self.SNR)
        else:                   # outdated CSI
            self.state = np.concatenate((np.log10(self.SNR[:-1])/10, self.tasks_prev/(10**9)), axis=0)

    def channel_type_name(self):

        if self.p == 1:
            channel = 'StaticChannel'
        else:
            channel = 'FadingChannel'

        if self.CSI == 1:
            channel = channel + '_PerfectCSI'
        else:
            channel = channel + '_OutdatedCSI'
        return channel

    def create_random_channel(self):
        '''
        create channel sequence h^
        :return: returns h^ for every server with length 1000
        '''
        h_c = []


        for c in range(self.S):     # for every channel
            h_c.append([])

        for s in range(self.S):
            np.random.seed(s)
            for i in range(1500):           # length of channel
                h_c[s].append(complex(np.random.randn(), np.random.randn()) / np.sqrt(2))
        return h_c

    def compute_action_space(self):
        # l = list(product(range(2), repeat=3))
        dck = [0, 1/6, 1/3, 2/3, 1]
        l = list(product(dck, repeat=self.S))
        # l = list(product([0, 4, 8, 16, 24], repeat=self.S))
        # act_all = np.asarray(l)
        l = np.round(np.asarray(l), 4)
        act_all = l[1:]
        # print(act_all)
        act_space = []
        for i in range(len(act_all)):
            if np.sum(act_all[i]) < 0.99:
                continue
            if round(np.sum(act_all[i]) - 0.01 + 0.5) == 1:
                act_space.append(act_all[i])
        #print(np.asarray(act_space))
        return np.asarray(act_space)

    def compute_action_space_V1(self):
        '''
        :return: action space regarding server selection
        '''
        l = list(product(range(2), repeat=self.S))
        action_space = np.asarray(l)
        action_space = action_space[1:]
        return action_space

    def total_error(self, c, n):
        '''
        computes overall error depending on set parameters and arguments n and c
        :param n: blocklength
        :param c: workload array
        :return: overall error
        '''
        t1 = self.TS * n
        ck = c * self.co
        t2 = self.T - t1
        r = self.pi / n  # coding rate
        totall = 1

        for k in range(self.S):
            if c[k] != 0:
                # Communication
                shannon = np.log2(1 + self.SNR[-1][k])  # shannon
                channel_disp = 1 - (1 / (1 + self.SNR[-1][k]) ** 2)  # channel dispersion
                Q_arg = np.sqrt(n / channel_disp) * (shannon - r) * np.log(2)
                comm_error = 0.5 * special.erfc(Q_arg / np.sqrt(2))
                # print([t2 - ((ck + self.d) + self.tasks_prev - self.time_slot_const) / self.f, 0])
                #print('comm:', comm_error)

                # Computation
                if self.CSI == 1:
                    comp_depend = 0
                else:
                    comp_depend = self.tasks_prev[-1][k] - self.comp_correlation
                    if comp_depend < 0:
                        comp_depend = 0
                m = t2 - (((ck[k] + self.d) + comp_depend) / self.f)
                if m < 0:
                    comp_err = 1
                else:
                    comp_err = (1 - self.threshold_computation) * (1 + ((self.xi * m) / (self.sigma / self.f))) ** (
                        -1 / self.xi)
                #print('comp:', comp_err)

                # Link k error
                total_k = c[k] * (comm_error + comp_err - (
                        comm_error * comp_err))  # compute error of single channel + computation
            else:
                total_k = 0
           # print('total_k:', total_k)
            totall = totall * (1 - total_k)
            #print('totall:', totall)

        # for i in range(self.S):  # total error
        #     if total_k[i] > 0.01:
        #         total_k[i] = 1

        total = 1 - totall  # overall error
        # self.tasks_prev = np.append(self.tasks_prev[1:], a * ck)
        self.tasks_prev = np.append(self.tasks_prev[1:], ck)
        self.tasks_prev = self.tasks_prev.reshape((self.W - 1 + self.CSI, self.S))

        #print('SNR:', self.SNR[-1])

        #print('total:', total)
        return total

        # t1 = self.TS * n
        # ck = c * self.co
        # t2 = self.T - t1
        # # Communication error
        # r = self.pi / n  # coding rate
        # shannon = np.log2(1 + self.SNR[-1])  # shannon
        # channel_disp = 1 - (1 / (1 + self.SNR[-1]) ** 2)  # channel dispersion
        # Q_arg = np.sqrt(n / channel_disp) * (shannon - r) * np.log(2)
        # comm_error = 0.5 * special.erfc(Q_arg / np.sqrt(2))
        # # print([t2 - ((ck + self.d) + self.tasks_prev - self.time_slot_const) / self.f, 0])
        #
        # # Computation error
        # if self.CSI==1:
        #     comp_depend = 0
        # else:
        #     comp_depend = self.tasks_prev[-1] - self.comp_correlation
        #     comp_depend[comp_depend < 0] = 0
        #
        # #used = np.clip(ck, 0, 1)
        # used = np.clip(c, 0, 1)
        # m = t2 - (((ck + self.d) + comp_depend) / self.f)
        # #m[m < 0] = 0
        # # print('m:', m)
        # #comp_err = (1 - self.threshold_computation) * (1 + ((self.xi * m) / (self.sigma / self.f))) ** (
        # #                -1 / self.xi)
        # if m < 0:
        #     comp_err = 1
        # else:
        #     comp_err = (1 - self.threshold_computation) * (1 + ((self.xi * m) / (self.sigma / self.f))) ** (
        #                 -1 / self.xi)
        #
        #
        # total_k = used * (comm_error + comp_err - (
        #                 comm_error * comp_err))  # compute error of single channel + computation
        #
        # # for i in range(self.S):  # total error
        # #     if total_k[i] > 0.01:
        # #         total_k[i] = 1
        #
        # total = 1 - np.prod(1 - total_k)        # compute overall error
        #
        # # Update workload array
        # self.tasks_prev = np.append(self.tasks_prev[1:], ck)
        # self.tasks_prev = self.tasks_prev.reshape((self.W -1 + self.CSI, self.S))
        # return total

    def update_channel_state(self):
        '''
        Update channel using total random channel variable
        :return:
        '''
        SNR = []
        for i in range(self.S):          # compute new SNR for every channel
            h_bar = complex(np.random.randn(), np.random.randn()) / np.sqrt(2)
            hdl = self.p * self.h[i] + (np.sqrt(1 - np.square(self.p)) * h_bar)
            SNR = np.append(SNR, self.SNR_avg[i] * abs(np.square(hdl)))
            self.h[i] = hdl
        self.SNR = np.append(self.SNR[1:], SNR)
        self.SNR = self.SNR.reshape((self.W, self.S))

    def update_channel_state_sequence(self, count):
        '''
        Updates channel based on pseudo-random channel sequence
        :param count: Cycle counter
        :return: update of SNR array
        '''
        SNR = []
        ch = []

        for i in range(self.S):          # compute new SNR for every channel
            h_bar = self.channel_sequence[i][count]     # using fixed channel sequence
            hdl = self.p * self.h[i] + (np.sqrt(1 - np.square(self.p)) * h_bar)
            SNR = np.append(SNR, self.SNR_avg[i] * abs(np.square(hdl)))
            self.h[i] = hdl

            ch = np.append(ch, abs(np.square(hdl)))

        #print('ch:', ch)
        self.chh = np.append(self.chh, ch)
        #print('chh:', self.chh)
        self.SNR = np.append(self.SNR[1:], SNR)
        self.SNR = self.SNR.reshape((self.W, self.S))


    def create_state(self):
        '''
        Creates state dependent on buffer and CSI
        :return: state (m+1)
        '''

        # division of 10 is for regulating input values in same range (can be changed)
        if self.CSI == 1:       # perfect CSI
            state = np.log10(self.SNR) / 10
            #state = np.log10(self.SNR)

        else:                   # outdated CSI
            #state = np.concatenate((np.log10(self.SNR[:-1]), self.tasks_prev/(10**9)), axis=0)
            state = np.concatenate((np.log10(self.SNR[:-1])/10, self.tasks_prev/(10**9)), axis=0)
        #print(state)
        return state

    def compute_rewards(self, error_probability):
        '''
        Computes reward depending on error
        :param error_probability:
        :return: reward r
        '''
        e = error_probability

        if e == 0:
            self.reward = 1
            return

        # if e == 1:
        #     self.reward = -1
        #     return

        self.reward = - np.log10(e)/100    # use 100 as factor (can be changed to best outcome)

    def calculate_blocklength_perfCSI(self, used):
        T = self.T
        t1 = 1
        n_lower_bound = 100
        n_upper_bound = int(np.floor(self.T / self.TS) - 100)
        lowest = 1
        t1_lowest = 0
        ck = self.co / np.count_nonzero(used)
        # time_slot_const = np.minself.comp_correlation * self.tasks_prev

        for n in range(n_lower_bound, n_upper_bound, 10):
            t1 = n * self.TS  # compute t1 from blocklength
            t2 = T - t1  # t2
            r = self.pi / n  # coding rate
            shannon = np.log2(1 + self.SNR[-1])  # shannon
            channel_disp = 1 - (1 / (1 + self.SNR[-1]) ** 2)  # channel dispersion
            Q_arg = np.sqrt(n / channel_disp) * (shannon - r) * np.log(2)
            comm_error = 0.5 * special.erfc(Q_arg / np.sqrt(2))

            # comp_depend = np.amax(self.tasks_prev[-2] - self.comp_correlation)
            # print('comp_depend:', comp_depend)
            # m = (t2 - ((ck + self.d) + comp_depend) / self.f)
            # print([t2 - ((ck + self.d) + self.tasks_prev - self.time_slot_const) / self.f, 0])
            m = np.asarray([t2 - (((ck*used) + self.d) / self.f), [0] * self.S]).max(axis=0, initial=0)
            # m = np.asarray([m, [0] * self.S]).max(axis=0, initial=0)
            comp_err = (1 - self.threshold_computation) * (1 + ((self.xi * m) / (self.sigma / self.f))) ** (
                        -1 / self.xi)

            total_k = used * (comm_error + comp_err - (
                        comm_error * comp_err))  # compute error of single channel + computation

            # for i in range(self.S):  # total error
            #     if total_k[i] > 0.01:
            #         total_k[i] = 1
            total = 1 - np.prod(1 - total_k)

            if total <= lowest:
                n_lowest = t1/self.TS
                lowest = total

        self.tasks_prev = np.append(self.tasks_prev[1:], ck*used)
        self.tasks_prev = self.tasks_prev.reshape((self.W-1+ self.CSI, self.S))
        return n_lowest

    def Assign_Cores(self, actions, count, train):
        '''
        :param actions: workload index
        :param count: Cycle counter for pseudo-random channel sequence
        :param train: Training or testing (1 for training)=> which channel update should be taken?
        :return: next state, reward, error
        '''

        ac = self.action[actions]
        #print('ac', ac)

        #action_blkl = int(round((actions[0]) * (self.T/self.TS)))   # map value to blocklength
       # action_selection = self.action[int(round(actions[1]*(len(self.action) -1)))]    # map value to index in workload array
        self.reward = 0                 # set reward to 0
        non_zeros = np.count_nonzero(ac)        # check if any server is choosen

        if non_zeros > 0:
            blkl = self.calculate_blocklength_perfCSI(ac)
            total_error = self.total_error(ac, blkl)      # computes the error probability for optimal t
            self.compute_rewards(total_error)         # assigns reward
            if total_error < 10**-100:
                total_error = 10**-100
            self.error += - np.log10(total_error)

        if train == 1:          # training or testing?
            self.update_channel_state()            # update channel SNR for next state
        else:
            self.update_channel_state_sequence(count)

        self.state = self.create_state()       # updates next state
        #print(self.state)
        return self.state, self.reward, total_error
