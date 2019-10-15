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

    def __init__(self, algorithm, list_intense, list_delay, list_clients, legend=None, p_max=None, p_min=None, tau=None,
                 interval_border=None, is_first_send=None, M=None):
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


class AlgorithmEnum(Enum):
    BINARY_EXP = 1


class MultipleAccess:
    _helping_interval = []
    _DEBUG = False
    DELAY = 'd'
    NUM_CLIENTS = 'N'
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

    def get_alg_name_list(self, list_algorithms):
        alg_name_list = []
        for algorithm in list_algorithms:
            alg_name_list.append(self.get_alg_name(algorithm))
        return alg_name_list

    def get_legend(self, algorithm, p_max=1, p_min=0.5, M=1):
        if algorithm == AlgorithmEnum.BINARY_EXP:
            return "{} | p_max = {} | p_min = {} | M = {}".format(self.get_alg_name(algorithm), p_max, p_min, M)

    def _simulate(self, list_intense, algorithm, p_max, p_min, M):
        list_clients, list_delay = [], []
        for intense in list_intense:
            res = self.run(algorithm, intense, M, p_max, p_min)
            list_delay.append(res[self.DELAY])
            list_clients.append(res[self.NUM_CLIENTS])
        return SimulationResult(algorithm=algorithm, list_intense=list_intense, list_delay=list_delay,
                                list_clients=list_clients,
                                p_min=p_min, p_max=p_max, M=M,
                                legend=self.get_legend(algorithm, p_max, p_min, M))

    @logger
    def simulate_system(self, algorithm, list_intense, list_M, list_p_max=[0.9], list_p_min=[0.3]):
        cprint("simulate_system | Lambda: {} | M: {}".format(list_intense, list_M), "cyan")
        list_results = []
        for M in list_M:
            for p_max in list_p_max:
                for p_min in list_p_min:
                    res = self._simulate(list_intense, algorithm, p_max, p_min, M)
                    list_results.append(res)
        return list_results

    def run(self, algorithm, intense, M, p_max, p_min):
        res = {}
        if algorithm == AlgorithmEnum.BINARY_EXP:
            res = self.run_bin_exp(intense, M, p_max, p_min)
            if self._DEBUG:
                print("{} | Lambda {} | M {} | p_max {} | p_min {} | d = {}".format(
                    self.get_alg_name(algorithm), intense, M,
                    p_max, p_min, res[self.DELAY]))
        return res

    def run_bin_exp(self, intense, M, p_max, p_min):
        result = {self.DELAY: 0, self.NUM_CLIENTS: 0}
        # if intense == 1:
        #     intense -= 0.01
        # result[self.NUM_CLIENTS] = intense * (2 - intense) / (2 * (1 - intense))

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
            if is_break:
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

    def generate_num_of_events(self, intense, M=1):
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
        fig, (ax1, ax2) = plt.subplots(1, 2)
        plt.suptitle(title)
        ax1.set_xlabel('Lambda')
        ax1.set_ylabel('N')
        ax2.set_xlabel('Lambda')
        ax2.set_ylabel('d')
        list_legend = []
        for result in list_results:
            ax1.plot(result.list_intense, result.list_clients)
            ax2.plot(result.list_intense, result.list_delay)
            list_legend.append(result.legend)
        ax1.set_xlim(0, 1.25)
        ax2.set_xlim(0, 1.25)
        ax2.set_ylim(0, 20)
        ax1.legend(list_legend)
        ax2.legend(list_legend)
        return fig


if __name__ == '__main__':
    simulation = MultipleAccess(total_time=100000, num_messages=100000, debug=True)

    list_lambda = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    # list_lambda = [0.01, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45,
    #                0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 1]
    # list_lambda = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]
    # list_med_evets = []
    # for l in list_lambda:
    #     simulation.generate_helping_interval(intense=l)
    #     s = 0
    #     for i in range(1000):
    #         s += simulation.generate_num_of_events(intense=l, M=2)
    #     s = s / 1000
    #     list_med_evets.append(s)
    # fig, (ax1, ax2) = plt.subplots(1, 2)
    # ax1.plot(list_lambda, list_med_evets)
    # plt.show()

    list_M = [2]
    list_p_max = [1, 0.8]
    list_p_min = [0.2, 0.4]

    list_results = []

    # TODO Интенсивность входного потока от выходного
    # Вариант 6 Двоичная экспоненциальная отсрочка

    # list_results.extend(
    #     simulation.simulate_system(algorithm=AlgorithmEnum.TIME_DIVISION, list_intense=list_lambda, list_M=list_M))
    # list_results.extend(
    #     simulation.simulate_system(algorithm=AlgorithmEnum.REQUEST_ACCESS, list_intense=list_lambda, list_M=list_M,
    #                                list_tau=list_tau))
    # list_results.extend(
    #     simulation.simulate_system(algorithm=AlgorithmEnum.ALOHA, list_intense=list_lambda, list_M=list_M,
    #                                list_p=list_p, is_first_send=False))
    # list_results.extend(
    #     simulation.simulate_system(algorithm=AlgorithmEnum.ALOHA, list_intense=list_lambda, list_M=list_M,
    #                                list_p=list_p, is_first_send=True))
    list_results.extend(
        simulation.simulate_system(algorithm=AlgorithmEnum.BINARY_EXP, list_intense=list_lambda, list_M=list_M,
                                   list_p_max=list_p_max, list_p_min=list_p_min))

    fig = MultipleAccess.plot_results(list_results, "Bin Exp algorithm")
    plt.show()
