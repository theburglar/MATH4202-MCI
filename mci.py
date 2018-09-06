from gurobipy import *
import random

class MCI:
    PES = 0
    MOD = 1
    OPT = 2

    I = 0
    D = 1

    SURVIVAL_RATES = {
        PES: ((0.09, 17, 1.10), (0.57, 61, 2.03)),
        MOD: ((0.24, 47, 1.30), (0.76, 138, 2.17)),
        OPT: ((0.56, 91, 1.58), (0.81, 160, 2.41))
    }

    def __init__(self):
        self.set_scenario(MCI.OPT)

    def set_scenario(self, scenario):
        self._scenario = scenario
        self._betas = MCI.SURVIVAL_RATES[scenario]

    def generate_data(self, seed):
        random.seed(seed)
        #number of immediate patients
        self._n_i = random.randint(10, 30)
        #number of delayed patients
        self._n_d = random.randint(10, 30)
        #number of ambulances
        self._n_a = random.randint(10, 30)
        #resources required for immediate patients
        self._w_i = random.randint(3, 6)
        #resources required for delayed patients
        self._w_d = random.randint(1, 3)
        required_resources = self._n_i * self._w_i + self._n_d * self._w_d

        #resources available per hospital
        self._c_h = []
        while sum(self._c_h) < required_resources:
            self._c_h.append(random.randint(10, 30))

        #number of hospitals
        self._n_h = len(self._c_h)

        #time to hospital
        t_h = [random.randint(5,50) for h in c_h]

    def generate_schedules(self):
        schedules = []

        used_h = [0 for h in self._n_h]


    def f_p(self, p, t):
        """
        :param t: time
        :param p: type of patient (I: immediate, D: delayed)
        :return:
        """
        b_0, b_1, b_2 = self._betas[MCI.I] if p == MCI.I else self._betas[MCI.D]
        return b_0 / (((t / b_1) ** b_2) + 1)

if __name__ == '__main__':
    mci = MCI()
    mci.set_scenario(MCI.PES)

    for t in range(100):
        print('Immediate', t, mci.f_p(MCI.I, t))
