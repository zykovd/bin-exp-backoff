import math
import random
from enum import Enum
from queue import PriorityQueue

import matplotlib.pyplot as plt
import numpy as np
from termcolor import cprint


def logger(func):
    def wrapper(*args, **kwargs):
        cprint(">> {} started".format(func.__name__), "green")
        value = func(*args, **kwargs)
        cprint(">> {} finished".format(func.__name__), "green")
        return value

    return wrapper


class SimulationResult:
    _algorithm = None
    _list_intense = None
    _list_delay = None
    _list_clients = None
    _legend = None
    _p_max = None
    _p_min = None
    _tau = None
    _interval_border = None
    _is_first_send = None
    _M = None
    _param1 = None
    _param2 = None
    _param3 = None
    _param4 = None

    def __init__(self, algorithm, list_intense, list_delay, list_clients, legend=None, p_max=None, p_min=None, tau=None,
                 interval_border=None, is_first_send=None, M=None, param1=None, param2=None, param3=None, param4=None):
        self._algorithm = algorithm
        self._list_intense = list_intense
        self._list_delay = list_delay
        self._list_clients = list_clients
        self._legend = legend
        self._p_max = p_max
        self._p_min = p_min
        self._tau = tau
        self._interval_border = interval_border
        self._is_first_send = is_first_send
        self._M = M
        self._param1 = param1
        self._param2 = param2
        self._param3 = param3
        self._param4 = param4

    @property
    def algorithm(self):
        return self._algorithm

    @property
    def list_intense(self):
        return self._list_intense

    @property
    def list_delay(self):
        return self._list_delay

    @property
    def list_clients(self):
        return self._list_clients

    @property
    def legend(self):
        return self._legend

    @property
    def p_max(self):
        return self._p_max

    @property
    def p_min(self):
        return self._p_min

    @property
    def tau(self):
        return self._tau

    @property
    def interval_border(self):
        return self._interval_border

    @property
    def is_first_send(self):
        return self._is_first_send

    @property
    def M(self):
        return self._M

    @property
    def param1(self):
        return self._param1

    @property
    def param2(self):
        return self._param2

    @property
    def param3(self):
        return self._param3

    @property
    def param4(self):
        return self._param4


class AlgorithmEnum(Enum):
    BINARY_EXP = 1
    ALOHA = 2
    ADAPTIVE_ALOHA = 3
    INTERVAL_BINARY_EXP = 4


class MultipleAccess:
    _helping_interval = []
    _DEBUG = False
    DELAY = 'd'
    NUM_CLIENTS = 'N'
    PARAM1 = 'p1'
    PARAM2 = 'p2'
    PARAM3 = 'p3'
    PARAM4 = 'p4'
    """
    Implementation of multiple access algorithms.
    """

    def __init__(self, total_time=10000, num_messages=10000, debug=False):
        self.total_time = total_time
        self.num_messages = num_messages
        self._DEBUG = debug

    def get_alg_name(self, algorithm):
        """
        Returns string representing algorithm name.

        :param algorithm:
        :return:
        """
        if algorithm == AlgorithmEnum.BINARY_EXP:
            return "BinExp"
        if algorithm == AlgorithmEnum.ALOHA:
            return "Aloha"
        if algorithm == AlgorithmEnum.ADAPTIVE_ALOHA:
            return "AdaptiveAloha"
        if algorithm == AlgorithmEnum.INTERVAL_BINARY_EXP:
            return "IntervalBinExp"

    def get_alg_name_list(self, list_algorithms):
        alg_name_list = []
        for algorithm in list_algorithms:
            alg_name_list.append(self.get_alg_name(algorithm))
        return alg_name_list

    def get_legend(self, algorithm, p_max=1, p_min=0.5, M=1, w_max=5, w_min=1):
        if algorithm == AlgorithmEnum.BINARY_EXP:
            return "{} | p_max = {} | p_min = {} | M = {}".format(self.get_alg_name(algorithm), p_max, p_min, M)
        if algorithm == AlgorithmEnum.ALOHA:
            return "{} | p = {} | M = {}".format(self.get_alg_name(algorithm), p_max, M)
        if algorithm == AlgorithmEnum.ADAPTIVE_ALOHA:
            return "{} | p = {} | M = {}".format(self.get_alg_name(algorithm), p_max, M)
        if algorithm == AlgorithmEnum.INTERVAL_BINARY_EXP:
            return "{} | W_max = {} | W_min = {} | M = {}".format(self.get_alg_name(algorithm), w_max, w_min, M)

    def _simulate(self, list_intense, algorithm, M, p_max=1, p_min=0.5, w_max=5, w_min=1):
        list_clients, list_delay = [], []
        list_p1, list_p2, list_p3, list_p4 = [], [], [], []
        for intense in list_intense:
            res = self.run(algorithm, intense, M, p_max, p_min, w_max, w_min)
            list_delay.append(res[self.DELAY])
            list_clients.append(res[self.NUM_CLIENTS])
            list_p1.append(res[self.PARAM1])
            list_p2.append(res[self.PARAM2])
            list_p3.append(res[self.PARAM3])
            list_p4.append(res[self.PARAM4])
        return SimulationResult(algorithm=algorithm, list_intense=list_intense, list_delay=list_delay,
                                list_clients=list_clients,
                                p_min=p_min, p_max=p_max, M=M,
                                legend=self.get_legend(algorithm, p_max, p_min, M, w_max, w_min),
                                param1=list_p1, param2=list_p2, param3=list_p3, param4=list_p4)

    @logger
    def simulate_system(self, algorithm, list_intense, list_M, list_p_max=[1], list_p_min=[0.5], list_w_max=[5],
                        list_w_min=[1]):
        cprint("simulate_system | Lambda: {} | M: {}".format(list_intense, list_M), "cyan")
        list_results = []
        if algorithm == AlgorithmEnum.INTERVAL_BINARY_EXP:
            for M in list_M:
                for w_max in list_w_max:
                    for w_min in list_w_min:
                        res = self._simulate(list_intense=list_intense, algorithm=algorithm, w_max=w_max, w_min=w_min,
                                             M=M)
                        list_results.append(res)
        else:
            for M in list_M:
                for p_max in list_p_max:
                    for p_min in list_p_min:
                        res = self._simulate(list_intense=list_intense, algorithm=algorithm, p_max=p_max, p_min=p_min,
                                             M=M)
                        list_results.append(res)
        return list_results

    @logger
    def run_bin_exp_save_probs(self, intense=0.85, M=4, p_max=1, p_min=0.015625):
        cprint("run_bin_exp_draw_prob | Lambda: {} | M: {}".format(intense, M), "cyan")
        temp_res = self.run_bin_exp(intense, M, p_max, p_min, True)
        result = SimulationResult(algorithm=AlgorithmEnum.BINARY_EXP, list_intense=[intense],
                                  list_clients=temp_res[self.NUM_CLIENTS], list_delay=temp_res[self.DELAY],
                                  param2=temp_res[self.PARAM2], p_max=p_max, p_min=p_min, M=M)
        return result

    @logger
    def run_bin_exp_interval_save_probs(self, intense=0.85, M=4, w_max=10, w_min=4):
        cprint("run_bin_exp_interval_save_probs | Lambda: {} | M: {}".format(intense, M), "cyan")
        temp_res = self.run_interval_bin_exp(intense, M, w_max, w_min, True)
        result = SimulationResult(algorithm=AlgorithmEnum.INTERVAL_BINARY_EXP, list_intense=[intense],
                                  list_clients=temp_res[self.NUM_CLIENTS], list_delay=temp_res[self.DELAY],
                                  param2=temp_res[self.PARAM2], p_max=w_max, p_min=w_min, M=M)
        return result

    @logger
    def simulate_channel_capture(self, intense=0.85, M=4, p_max=1, p_min=0.015625):
        cprint("simulate_system | Lambda: {} | M: {}".format(intense, M), "cyan")
        _, list_results = self.run_channel_capture(intense, M, p_max, p_min)
        return list_results

    def run_channel_capture(self, intense, M, p_max, p_min):
        list_results = [[] for _ in range(M + 1)]
        result = {self.DELAY: 0, self.NUM_CLIENTS: 0, self.PARAM1: 0, self.PARAM2: 0, self.PARAM3: 0, self.PARAM4: 0}
        self.generate_helping_interval(intense=intense / M)

        _p_max = _p_min = 1 / M

        time, total_sends, total_delay, clients = 0, 0, 0.0, 0
        message_ready = [False for _ in range(M)]
        appear_time = [[] for _ in range(M)]
        probs = [p_max for _ in range(M)]

        init_num_of_messages = 50
        for i in range(M):
            for _ in range(init_num_of_messages):
                appear_time[i].append(random.random())
                message_ready[i] = True

        total_messages = 0
        channel_capture = False

        while time < self.total_time:

            is_sending = [False for _ in range(M)]
            for i in range(M):
                if message_ready[i] and random.random() <= probs[i]:
                    is_sending[i] = True
            cur_clients = is_sending.count(True)
            clients += cur_clients
            if cur_clients > 1:
                if not channel_capture:
                    for i in range(M):
                        if is_sending[i]:
                            probs[i] = max(probs[i] / 2, _p_min)
                else:
                    for i in range(M - 1):
                        if is_sending[i]:
                            probs[i] = max(probs[i] / 2, p_min)

            elif cur_clients == 1:
                idx = is_sending.index(True)
                if not channel_capture:
                    probs[idx] = _p_max
                else:
                    probs[idx] = p_max
                total_sends += 1
                delay = time + 1 - appear_time[idx].pop(0)
                total_delay += delay
                if len(appear_time[idx]) == 0:
                    message_ready[idx] = False
            for i in range(M):
                if len(appear_time[i]) == 0:
                    message_ready[i] = False
                num_of_messages = self.generate_num_of_events(intense=intense / M)
                total_messages += num_of_messages
                for _ in range(num_of_messages):
                    appear_time[i].append(time + random.random())
                    message_ready[i] = True
            time += 1

            if time >= self.total_time // 100 and not channel_capture:
                channel_capture = True
                if self._DEBUG:
                    cprint('Channel capture on time = {}'.format(time), 'red')
                message_ready.append(True)
                appear_time.append([time + random.random()])
                probs.append(p_max)
                M += 1
                for _ in range(init_num_of_messages ** 2):
                    appear_time[M - 1].append(random.random())
                for i in range(M):
                    list_results[i].append(probs[i])
            elif time >= self.total_time // 100 and channel_capture:
                for i in range(M):
                    list_results[i].append(probs[i])
            elif not channel_capture:
                for i in range(M):
                    list_results[i].append(probs[i])
                list_results[M].append(0)

        for i in range(M):
            if len(appear_time[i]) > 0:
                delay = time + 1 - appear_time[i].pop(0)
                total_delay += delay
                total_sends += 1

        if self._DEBUG:
            cprint('Remaining messages in queue =  {}'.format(sum([len(x) for x in appear_time])), 'red')
            cprint('Generated messages / Time = {}'.format(total_messages / self.total_time), 'red')

        if total_sends != 0:
            result[self.DELAY] = total_delay / total_sends
        result[self.NUM_CLIENTS] = clients / time
        result[self.PARAM1] = total_sends / time
        return result, list_results

    def run(self, algorithm, intense, M, p_max, p_min, w_max, w_min):
        res = {}
        if algorithm == AlgorithmEnum.BINARY_EXP:
            res = self.run_bin_exp(intense, M, p_max, p_min)
            if self._DEBUG:
                print("{} | Lambda {} | M {} | p_max {} | p_min {} | d = {}\n".format(
                    self.get_alg_name(algorithm), intense, M,
                    p_max, p_min, res[self.DELAY]))
        if algorithm == AlgorithmEnum.ALOHA:
            res = self.run_aloha(intense, M, p_max)
            if self._DEBUG:
                print("{} | Lambda {} | M {} | p {} | d = {}\n".format(
                    self.get_alg_name(algorithm), intense, M,
                    p_max, res[self.DELAY]))
        if algorithm == AlgorithmEnum.ADAPTIVE_ALOHA:
            res = self.run_adaptive_aloha(intense, M, p_max)
            if self._DEBUG:
                print("{} | Lambda {} | M {} | p {} | d = {}\n".format(
                    self.get_alg_name(algorithm), intense, M,
                    p_max, res[self.DELAY]))
        if algorithm == AlgorithmEnum.INTERVAL_BINARY_EXP:
            res = self.run_interval_bin_exp(intense, M, w_max, w_min)
            if self._DEBUG:
                print("{} | Lambda {} | M {} | W_max {} | W_min {} | d = {}\n".format(
                    self.get_alg_name(algorithm), intense, M,
                    w_max, w_min, res[self.DELAY]))
        return res

    def run_adaptive_aloha(self, intense, M, p_max):
        result = {self.DELAY: 0, self.NUM_CLIENTS: 0, self.PARAM1: 0, self.PARAM2: 0, self.PARAM3: 0, self.PARAM4: 0}
        # self.generate_helping_interval(intense=intense)
        self.generate_helping_interval(intense=intense / M)
        # print(self._helping_interval)

        time, total_sends, total_delay, clients = 0, 0, 0.0, 0
        message_ready = [False for _ in range(M)]
        appear_time = [[] for _ in range(M)]
        cur_prob = p_max

        total_messages = 0
        while time < self.total_time:
            is_sending = [False for _ in range(M)]
            for i in range(M):
                if message_ready[i] and random.random() <= cur_prob:
                    is_sending[i] = True
            cur_clients = is_sending.count(True)
            clients += cur_clients
            if cur_clients > 1:
                cur_prob = max(1 / M, cur_prob / 2)
            elif cur_clients == 0:
                cur_prob = min(1, 2 * cur_prob)
            elif cur_clients == 1:
                idx = is_sending.index(True)
                total_sends += 1
                delay = time + 1 - appear_time[idx].pop(0)
                total_delay += delay
                if len(appear_time[idx]) == 0:
                    message_ready[idx] = False
            for i in range(M):
                if len(appear_time[i]) == 0:
                    message_ready[i] = False
                num_of_messages = self.generate_num_of_events(intense=intense / M)
                total_messages += num_of_messages
                for _ in range(num_of_messages):
                    appear_time[i].append(time + random.random())
                    message_ready[i] = True
            time += 1

        for i in range(M):
            if len(appear_time[i]) > 0:
                delay = time + 1 - appear_time[i].pop(0)
                total_delay += delay
                total_sends += 1

        if self._DEBUG:
            cprint('Remaining messages in queue =  {}'.format(sum([len(x) for x in appear_time])), 'red')
            cprint('Generated messages / Time = {}'.format(total_messages / self.total_time), 'red')

        if total_sends != 0:
            result[self.DELAY] = total_delay / total_sends
        result[self.NUM_CLIENTS] = clients / time
        result[self.PARAM1] = total_sends / time
        return result

    def run_aloha(self, intense, M, p_max):
        result = {self.DELAY: 0, self.NUM_CLIENTS: 0, self.PARAM1: 0, self.PARAM2: 0, self.PARAM3: 0, self.PARAM4: 0}
        # self.generate_helping_interval(intense=intense)
        self.generate_helping_interval(intense=intense / M)
        # print(self._helping_interval)

        time, total_sends, total_delay, clients = 0, 0, 0.0, 0
        message_ready = [False for _ in range(M)]
        appear_time = [[] for _ in range(M)]

        total_messages = 0

        while time < self.total_time:

            is_sending = [False for _ in range(M)]
            for i in range(M):
                if message_ready[i] and random.random() <= p_max:
                    is_sending[i] = True
            cur_clients = is_sending.count(True)
            clients += cur_clients
            if cur_clients == 1:
                idx = is_sending.index(True)
                total_sends += 1
                delay = time + 1 - appear_time[idx].pop(0)
                total_delay += delay
                if len(appear_time[idx]) == 0:
                    message_ready[idx] = False
            for i in range(M):
                if len(appear_time[i]) == 0:
                    message_ready[i] = False
                num_of_messages = self.generate_num_of_events(intense=intense / M)
                total_messages += num_of_messages
                for _ in range(num_of_messages):
                    appear_time[i].append(time + random.random())
                    message_ready[i] = True
            time += 1

        for i in range(M):
            if len(appear_time[i]) > 0:
                delay = time + 1 - appear_time[i].pop(0)
                total_delay += delay
                total_sends += 1

        if self._DEBUG:
            cprint('Remaining messages in queue =  {}'.format(sum([len(x) for x in appear_time])), 'red')
            cprint('Generated messages / Time = {}'.format(total_messages / self.total_time), 'red')

        if total_sends != 0:
            result[self.DELAY] = total_delay / total_sends
        result[self.NUM_CLIENTS] = clients / time
        result[self.PARAM1] = total_sends / time
        return result

    def run_bin_exp(self, intense, M, p_max, p_min, plot_probs=False):
        result = {self.DELAY: 0, self.NUM_CLIENTS: 0, self.PARAM1: 0, self.PARAM2: 0, self.PARAM3: 0, self.PARAM4: 0}
        # self.generate_helping_interval(intense=intense)
        self.generate_helping_interval(intense=intense / M)
        # print(self._helping_interval)

        time, total_sends, total_delay, clients = 0, 0, 0.0, 0
        message_ready = [False for _ in range(M)]
        appear_time = [[] for _ in range(M)]
        probs = [p_max for _ in range(M)]

        if plot_probs:
            result[self.PARAM2] = [[p_max] for _ in range(M)]

        total_messages = 0

        while time < self.total_time:
            is_sending = [False for _ in range(M)]
            for i in range(M):
                if message_ready[i] and random.random() <= probs[i]:
                    is_sending[i] = True
            cur_clients = is_sending.count(True)
            clients += cur_clients
            if cur_clients > 1:
                for i in range(M):
                    if is_sending[i]:
                        probs[i] = max(probs[i] / 2, p_min)
            elif cur_clients == 1:
                idx = is_sending.index(True)
                probs[idx] = p_max
                total_sends += 1
                delay = time + 1 - appear_time[idx].pop(0)
                total_delay += delay
                if len(appear_time[idx]) == 0:
                    message_ready[idx] = False
            for i in range(M):
                if len(appear_time[i]) == 0:
                    message_ready[i] = False
                num_of_messages = self.generate_num_of_events(intense=intense / M)
                total_messages += num_of_messages
                for _ in range(num_of_messages):
                    appear_time[i].append(time + random.random())
                    message_ready[i] = True
                if plot_probs:
                    result[self.PARAM2][i].append(probs[i])
            time += 1

        for i in range(M):
            if len(appear_time[i]) > 0:
                delay = time + 1 - appear_time[i].pop(0)
                total_delay += delay
                total_sends += 1

        if self._DEBUG:
            cprint('Remaining messages in queue =  {}'.format(sum([len(x) for x in appear_time])), 'red')
            cprint('Generated messages / Time = {}'.format(total_messages / self.total_time), 'red')

        if total_sends != 0:
            result[self.DELAY] = total_delay / total_sends
        result[self.NUM_CLIENTS] = clients / time
        result[self.PARAM1] = total_sends / time
        return result

    def run_interval_bin_exp(self, intense, M, w_max, w_min, plot_w=False):
        result = {self.DELAY: 0, self.NUM_CLIENTS: 0, self.PARAM1: 0, self.PARAM2: 0, self.PARAM3: 0, self.PARAM4: 0}
        self.generate_helping_interval(intense=intense / M)

        time, total_sends, total_delay, clients = 0, 0, 0.0, 0
        message_ready = [False for _ in range(M)]
        appear_time = [[] for _ in range(M)]
        intervals = [0 for _ in range(M)]
        intervals_border = [w_min for _ in range(M)]

        if plot_w:
            result[self.PARAM2] = [[intervals_border[i]] for i in range(M)]

        total_messages = 0

        while time < self.total_time:

            is_sending = [False for _ in range(M)]
            for i in range(M):
                if message_ready[i] and intervals[i] == 0:
                    is_sending[i] = True
            cur_clients = is_sending.count(True)
            clients += cur_clients
            if cur_clients > 1:
                for i in range(M):
                    if is_sending[i]:
                        intervals_border[i] = min(intervals_border[i] * 2, w_max)
                        intervals[i] = random.randint(0, intervals_border[i])
            elif cur_clients == 1:
                idx = is_sending.index(True)
                intervals_border[idx] = w_min
                intervals[idx] = random.randint(0, intervals_border[idx])
                total_sends += 1
                delay = time + 1 - appear_time[idx].pop(0)
                total_delay += delay
                if len(appear_time[idx]) == 0:
                    message_ready[idx] = False
            for i in range(M):
                if len(appear_time[i]) == 0:
                    message_ready[i] = False
                num_of_messages = self.generate_num_of_events(intense=intense / M)
                # if num_of_messages != 0:
                #     print('here')
                total_messages += num_of_messages
                for _ in range(num_of_messages):
                    appear_time[i].append(time + random.random())
                    message_ready[i] = True
                if plot_w:
                    result[self.PARAM2][i].append(intervals_border[i])
            time += 1
            for i in range(M):
                if intervals[i] > 0:
                    intervals[i] -= 1

        for i in range(M):
            if len(appear_time[i]) > 0:
                delay = time + 1 - appear_time[i].pop(0)
                total_delay += delay
                total_sends += 1

        if self._DEBUG:
            cprint('Remaining messages in queue =  {}'.format(sum([len(x) for x in appear_time])), 'red')
            cprint('Generated messages / Time = {}'.format(total_messages / self.total_time), 'red')

        if total_sends != 0:
            result[self.DELAY] = total_delay / total_sends
        result[self.NUM_CLIENTS] = clients / time
        result[self.PARAM1] = total_sends / time
        return result

    def _run_bin_exp(self, intense, M, p_max, p_min):
        result = {self.DELAY: 0, self.NUM_CLIENTS: 0}

        appear_time = self.generate_intervals(intense, M)

        time, total_sends, total_delay = 0, 0, 0.0
        message_ready = [False for _ in range(M)]
        probs = [1 for _ in range(M)]

        while True:
            is_sending = [False for _ in range(M)]
            for i in range(M):
                if len(appear_time[i]) > 0 and time > appear_time[i][0]:
                    message_ready[i] = True
            for i in range(M):
                if message_ready[i] and random.random() <= probs[i]:
                    is_sending[i] = True
            if is_sending.count(True) > 1:
                for i in range(M):
                    if is_sending[i]:
                        probs[i] = max(probs[i] / 2, p_min)
            elif is_sending.count(True) == 1:
                idx = is_sending.index(True)
                probs[idx] = p_max
                total_sends += 1
                message_ready[idx] = False
                total_delay += time + 1 - appear_time[idx][0]
                appear_time[idx].pop(0)
            time += 1
            is_break = True
            for i in range(M):
                if len(appear_time[i]) != 0:
                    is_break = False
            if is_break or time > self.total_time:
                break

        if total_sends != 0:
            result[self.DELAY] = total_delay / total_sends
        result[self.NUM_CLIENTS] = total_sends / time
        return result

    def generate_intervals(self, intense, M=1):
        tau = np.random.exponential(1 / (intense / M), self.num_messages // 2)
        t = np.cumsum(tau)
        appear_time = [[] for i in range(M)]
        for time in t:
            client = random.randrange(0, M)
            appear_time[client].append(float(time))
        return appear_time

    def generate_num_of_events(self, intense):
        rand = random.uniform(0, 1)
        num_of_events = -1
        for i in range(len(self._helping_interval)):
            if i == 0:
                if rand < self._helping_interval[i]:
                    num_of_events = 0
            else:
                if self._helping_interval[i - 1] < rand <= self._helping_interval[i]:
                    num_of_events = i
        if num_of_events == -1:
            num_of_events = len(self._helping_interval)
        return num_of_events

    def generate_helping_interval(self, intense, epsilon=10e-7):
        self._helping_interval = []
        cur_interval = 1
        num_of_events = 0
        while cur_interval > epsilon:
            cur_interval = math.exp(-intense) * math.pow(intense, num_of_events) / math.factorial(num_of_events)
            self._helping_interval.append(cur_interval)
            num_of_events += 1
        self._helping_interval = np.cumsum(self._helping_interval)

    @staticmethod
    def plot_results(list_results, title=''):
        fig, (ax1, ax2, ax3) = plt.subplots(1, 3)
        plt.suptitle(title)
        ax1.set_xlabel('Lambda')
        ax1.set_ylabel('N')
        ax2.set_xlabel('Lambda')
        ax2.set_ylabel('d')
        ax3.set_xlabel('Lambda')
        ax3.set_ylabel('In/Out')
        list_legend = ['w_min = 1 | w_max = 8', 'w_min = 1 | w_max = 1000']
        for result in list_results:
            ax1.plot(result.list_intense, result.list_clients)
            ax2.plot(result.list_intense, result.list_delay)
            ax3.plot(result.list_intense, result.param1)
            # list_legend.append(result.legend)
        # ax1.set_xlim(0, 1.25)
        # ax2.set_xlim(0, 1.25)
        ax2.set_ylim(0, 20)
        ax1.legend(list_legend)
        ax2.legend(list_legend)
        ax3.legend(list_legend)
        return fig

    @staticmethod
    def plot_probs(list_results):
        fig = None
        for result in list_results:
            fig, (ax1) = plt.subplots(1, 1)
            plt.suptitle("M {} | Lambda {} | p_max {} | p_min {}".format(result.M, result.list_intense[0], result.p_max,
                                                                         result.p_min))
            ax1.set_xlabel('t')
            ax1.set_ylabel('P')
            list_legend = []
            for i in range(result.M):
                ax1.plot([t for t in range(len(result.param2[i]))], result.param2[i])
                list_legend.append("client{}".format(i))
            med = sum([sum(result.param2[i]) for i in range(result.M)]) / (result.M * len(result.param2[0]))
            ax1.plot([t for t in range(len(result.param2[0]))], [med for _ in range(len(result.param2[0]))], '--')
            list_legend.append("mean")
            # ax1.set_xlim(0, 1.25)
            # ax2.set_xlim(0, 1.25)
            ax1.legend(list_legend)
        return fig

    @staticmethod
    def plot_w(list_results):
        fig = None
        for result in list_results:
            fig, (ax1) = plt.subplots(1, 1)
            # plt.suptitle("M {} | Lambda {} | w_max {} | w_min {}".format(result.M, result.list_intense[0], result.p_max,
            #                                                              result.p_min))
            plt.suptitle("M {} | Lambda {} | w_max {} | w_min {}".format(result.M, result.list_intense[0], 1000,
                                                                         result.p_min))
            ax1.set_xlabel('t')
            ax1.set_ylabel('W')
            list_legend = []
            for i in range(result.M):
                ax1.plot([t for t in range(len(result.param2[i]))], result.param2[i])
                list_legend.append("client{}".format(i))
            med = sum([sum(result.param2[i]) for i in range(result.M)]) / (result.M * len(result.param2[0]))
            ax1.plot([t for t in range(len(result.param2[0]))], [med for _ in range(len(result.param2[0]))], '--')
            list_legend.append("mean")
            ax1.set_ylim(1, med+3)
            ax1.set_xlim(0, 1000)
            ax1.legend(list_legend)
        return fig

    @staticmethod
    def plot_channel_capture(list_results, total_time, title=''):
        fig, (ax1) = plt.subplots(1, 1)
        plt.suptitle(title)
        ax1.set_xlabel('Time')
        ax1.set_ylabel('Probability')
        for result in list_results:
            ax1.plot([i for i in range(total_time)], result)
        ax1.set_xlim(total_time // 100 - 400, total_time // 100 + 400)
        # ax2.set_xlim(0, 1.25)
        return fig


if __name__ == '__main__':
    total_time = 100000
    simulation = MultipleAccess(total_time=total_time, num_messages=100000, debug=True)

    # list_lambda = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    # list_lambda = [0.01, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45,
    #                0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 1]
    list_lambda = [0.01, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]

    list_M = [5]

    list_w_max = [8, 15]
    list_w_min = [1]

    list_results = []

    # Вариант 7 Двоичная экспоненциальная отсрочка (Интервальный вариант)

    list_results.extend(
        simulation.simulate_system(algorithm=AlgorithmEnum.INTERVAL_BINARY_EXP, list_intense=list_lambda, list_M=list_M,
                                   list_w_max=list_w_max, list_w_min=list_w_min)
    )

    MultipleAccess.plot_results(list_results)

    list_results = []

    list_results.append(simulation.run_bin_exp_interval_save_probs(intense=0.1, M=5, w_max=1000, w_min=1))
    list_results.append(simulation.run_bin_exp_interval_save_probs(intense=0.3, M=5, w_max=1000, w_min=1))
    list_results.append(simulation.run_bin_exp_interval_save_probs(intense=0.5, M=5, w_max=1000, w_min=1))
    list_results.append(simulation.run_bin_exp_interval_save_probs(intense=0.8, M=5, w_max=1000, w_min=1))

    fig = MultipleAccess.plot_w(list_results=list_results)

    plt.show()
