from gurobipy import *
import random

n_i = 2
n_d = 2
n_a = 2
w_i = 3
w_d = 3
c_h = [10 for i in range(3)]
t_h = [10,10,5]

I = 0
D = 1
n = [n_i, n_d]
P = [I, D]

PES = 0
MOD = 1
OPT = 2

SURVIVAL_RATES = {
    PES: ((0.09, 17, 1.10), (0.57, 61, 2.03)),
    MOD: ((0.24, 47, 1.30), (0.76, 138, 2.17)),
    OPT: ((0.56, 91, 1.58), (0.81, 160, 2.41))
}


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
    b_h = [0 for h in range(len(c_h))]
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
        t += t_h[h]
        g += f_p(t,p)
        t += t_h[h]
    return g
    

H = range(len(c_h))
schedules = all_schedules([], c_h, n_i, n_d)
S = range(len(schedules))

master = Model('Master Problem')
master.setParam('OutputFlag',1)

# Variables
lambda_s = {s: master.addVar(vtype=GRB.BINARY)
            for s in S}
                      
master.update()


# Objective
master.setObjective(quicksum(get_gs(schedules[s]) * lambda_s[s] for s in S), GRB.MAXIMIZE)

# Constraints
MaxPeople = {p: master.addConstr(quicksum(get_people(schedules[s])[p] * lambda_s[s] for s in S) <= n[p]) 
                    for p in P}
ResourceCapacity = {h: master.addConstr(quicksum(get_resources_used(schedules[s])[h] * lambda_s[s] for s in S) <= c_h[h])
                    for h in H}
MaxAmbulances = master.addConstr(quicksum(lambda_s[s] for s in S) <= n_a)

master.optimize()

for s in lambda_s:
    if lambda_s[s].x==1:
        print(schedules[s])
        print(get_gs(schedules[s]))

nodes = [(p,h) for p in P for h in H]
        