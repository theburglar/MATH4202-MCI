from gurobipy import *
from pprint import pprint
import random

n_i = 2
n_d = 2
n_a = 30
w_i = 3
w_d = 3
c = [5 for i in range(3)]
times = [10, 10, 5]

I = 0
D = 1
n = [n_i, n_d]
P = [I, D]
w = [w_i, w_d]

PES = 0
MOD = 1
OPT = 2

SURVIVAL_RATES = {
    PES: ((0.09, 17, 1.10), (0.57, 61, 2.03)),
    MOD: ((0.24, 47, 1.30), (0.76, 138, 2.17)),
    OPT: ((0.56, 91, 1.58), (0.81, 160, 2.41))
}

START_PATIENT = -1
START_HOSPITAL = -1
START_NODE = -1

FINAL_PATIENT = 0
FINAL_HOSPITAL = 0
FINAL_NODE = None


def generate_schedules(schedule, resources_h, p_i, p_d):
    schedules = []
    more = False
    for h in H:
        if resources_h[h] >= w_i and p_i > 0:
            new_schedule = schedule[:] + [(I, h)]
            new_resources = resources_h[:]
            new_resources[h] -= w_i
            schedules.extend(generate_schedules(new_schedule, new_resources, p_i-1, p_d))
            more = True
        if resources_h[h] >= w_d and p_d > 0:
            new_schedule = schedule[:] + [(D, h)]
            new_resources = resources_h[:]
            new_resources[h] -= w_d
            schedules.extend(generate_schedules(new_schedule, new_resources, p_i, p_d-1))
            more = True
    if not more:
        return [schedule]
    return schedules
    
def all_schedules(schedule, resources_h, p_i, p_d):
    schedules = []
    for i in range(p_i):
        for d in range(p_d):
            schedules.extend(generate_schedules([], resources_h, i, d))
    return schedules

def get_resources_used(schedule):
    b_h = [0 for h in range(len(c))]
    for p, h in schedule:
         b_h[h] += w_i if p == I else w_d
    return b_h    
    
    
def get_people(schedule):
    a = [0, 0]
    for p, h in schedule:
        a[p] += 1
    return a
    
def f_p(t, p):
    """
    :param t: time
    :param p: type of patient (I: immediate, D: delayed)
    :return:
    """
    b_0, b_1, b_2 = SURVIVAL_RATES[MOD][I] if p == I else SURVIVAL_RATES[MOD][D]
    return b_0 / (((t / b_1) ** b_2) + 1)
    
def get_gs(schedule):
    g = 0
    t = 0
    for p, h in schedule:
        t += times[h]
        g += f_p(t,p)
        t += times[h]
    return g

######################################################################
#        SUB-PROBLEM
######################################################################

def subproblem():

    def F(W_, R, i, j, T):

        W_ = list(W_)
        # previous trip
        p_i, h_i = vertices[i] if i != START_NODE else (0, 0)
        p_j, h_j = vertices[j]

        # increase number of patients p
        W_[p_j] = W_[p_j] + 1 if i != FINAL_NODE else W_[p_j]
        # increase resources used as hospital h
        W_[h_j + 2] = W_[h_j + 2] + w[p_j] if i != FINAL_NODE else W_[h_j + 2]

        # dual variables
        pi = MaxPeople[p_j].pi
        rho = ResourceCapacity[h_j].pi
        sigma = MaxAmbulances.pi
        print('IT\s TIME!', pi, rho, sigma)

        R_j = R + f_p(T, p_j) - pi - w[p_j]*rho if i is not FINAL_NODE else R - sigma
        # print('R_j', R_j)
        T = T + times[h_i] + times[h_j] if i is not START_NODE else times[h_j]

        return tuple(W_), R_j, T

    def is_dominated(W, R, W_, R_):
        """returns whether the label W_,R_ is dominated by label W,R"""
        for i in P:
            # people count
            if W_ [i] > W[i]:
                return False
        for i in range(2, len(W)):
            # hospital resources
            if W_ [i] < W[i]:
                return False
        if R_ > R:
            return False

        return True

    def is_dominated_by_set(W_, R_, labels):
        """returns whether the label W_, R_ is dominated by any vector in labels"""
        for i, W, R, t, s in labels:
            if is_dominated(W, R, W_, R_):
                # W_,R_ is dominated by W,R
                return True
        return False

    def EFF(W, R, labels):
        """returns the set of labels from labels that is not dominated by W, R"""
        eff = set()
        for i, W_, R_, t, s in labels:
            if not(all(W[i] <= W_[i] for i in range(len(W))) and R >= R_):
                # W_,R_ is not dominated by W,R
                eff.add((i, W_, R_, t, s))
        return eff

    vertices = [(p,h) for p in P for h in H] + [(FINAL_PATIENT, FINAL_HOSPITAL)]
    V = range(len(vertices))
    FINAL_NODE = len(V) - 1

    #Step 0 Initialisation
    T = 0   #current time
    W_i = (0, 0) + tuple([0 for h in H])
    R_i = 0
    s = tuple()

    E = [set() for v in V]
    L = [(START_NODE, W_i, R_i, T, s)]

    while len(L) > 0:
        #Step 1
        i, W_i, R_i, T_i, s = L.pop(0)

        if i == len(V):
            # node v_f
            continue

        #Step 2
        for j, v in enumerate(vertices):
            W_j, R_j, T_j = F(W_i, R_i, i, j, T_i)

            if not any(res > res_max for  res, res_max in zip(W_j, n + c)):
                if not is_dominated_by_set(W_j, R_j, E[j]):
                    s_j = s + (vertices[j],)
                    label = (j, W_j, R_j, T_j, s_j)
                    L.append(label)

                    E[j] = EFF(W_j, R_j, E[j])
                    E[j].add(label)

    # Good boi labels
    # pprint(E[FINAL_NODE])
    return max(E[FINAL_NODE], key=lambda x: x[2])

######################################################################
#              MASTER PROBLEM
######################################################################

H = range(len(c))
schedules = all_schedules([], c, n_i, n_d)
schedules = [[(0,2)]]
S = range(len(schedules))

master = Model('Master Problem')
master.setParam('OutputFlag', 0)

# Variables
lambda_s = {s: master.addVar() #vtype=GRB.BINARY
            for s in S}

# Objective
master.setObjective(quicksum(get_gs(schedules[s]) * lambda_s[s] for s in S), GRB.MAXIMIZE)

# Constraints
MaxPeople = {p: master.addConstr(
                    quicksum(get_people(schedules[s])[p] * lambda_s[s]
                             for s in S) <= n[p])
             for p in P}
ResourceCapacity = {h: master.addConstr(
                        quicksum(get_resources_used(schedules[s])[h] * lambda_s[s]
                                 for s in S) <= c[h])
                    for h in H}
MaxAmbulances = master.addConstr(quicksum(lambda_s[s] for s in S) <= n_a)


# for s in lambda_s:
#     if lambda_s[s].x==1:
#         print(schedules[s])
#         print(get_gs(schedules[s]))

while True:
    master.optimize()
    # print(','.join([str(MaxPeople[p].pi) for p in P]))
    # print(','.join([str(ResourceCapacity[h].pi) for h in H]))
    # print(MaxAmbulances.pi)
    # print('LAMBDAS', lambda_s)

    pprint(master.getConstrs())

    # print('#' * 80)
    solution = subproblem()
    print('#' * 80)
    pprint(solution)
    pprint(schedules)


    schedules.append(list(solution[-1]))
    S = range(len(schedules))

    lambda_s[len(S)-1] = master.addVar()

    # remove all constraints
    master.remove(master.getConstrs())

    # new constraints
    MaxPeople = {p: master.addConstr(
        quicksum(get_people(schedules[s])[p] * lambda_s[s]
                 for s in S) <= n[p])
        for p in P}
    ResourceCapacity = {h: master.addConstr(
                        quicksum(get_resources_used(schedules[s])[h] * lambda_s[s] for s in S) <= c[h])
                        for h in H}
    MaxAmbulances = master.addConstr(quicksum(lambda_s[s] for s in S) <= n_a)

    # new objective
    master.setObjective(quicksum(get_gs(schedules[s]) * lambda_s[s] for s in S), GRB.MAXIMIZE)